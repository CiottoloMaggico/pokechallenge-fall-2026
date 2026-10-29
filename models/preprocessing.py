"""Preprocessing steps shared by the concrete models."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.utils.validation import check_is_fitted

from model import EstimatorModel, Model


class DropEmptyColumns(TransformerMixin, BaseEstimator):
    """Drop the columns that are entirely NaN in the training data.

    Tree models handle missing values, but some (HistGradientBoosting)
    fail on a column with no value at all, which happens when a feature
    errors on every battle under ``ErrorPolicy.NAN``.
    """

    def fit(self, x: Any, y: Any = None) -> DropEmptyColumns:
        self.keep_ = ~np.isnan(np.asarray(x, dtype=float)).all(axis=0)
        return self

    def transform(self, x: Any) -> Any:
        check_is_fitted(self, "keep_")
        if hasattr(x, "iloc"):
            return x.iloc[:, self.keep_]
        return np.asarray(x)[:, self.keep_]


def tree_model(name: str, estimator: Any) -> Model:
    """Tree model behind a step that drops all-NaN columns."""
    return EstimatorModel(name, make_pipeline(DropEmptyColumns(), estimator))


class MissingDependencyError(ImportError):
    """The library behind a model is not installed."""


def require(module_name: str) -> Any:
    """Import an optional library, with an install hint if it is missing."""
    import importlib

    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise MissingDependencyError(
            f"{module_name} is not installed: pip install {module_name}"
        ) from exc
