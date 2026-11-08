"""A model fitted on all data, saved with what it needs to predict again."""

from __future__ import annotations

import json
import pickle
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from interfaces import FeatureSource, FittableFeatureSource, Model, is_stateful

RUN_MODEL_FILE = "model.pkl"
RUN_INFO_FILE = "run.json"
FEATURE_STATE_FILE = "feature_state.pkl"


class FeatureMismatchError(ValueError):
    """The feature source no longer produces the columns seen in fit."""


@dataclass(frozen=True)
class TrainedRun:
    """A model fitted on all data, bundled with its feature source.

    ``columns`` are the feature names seen at fit time. Every prediction
    checks that the feature source still produces exactly those columns,
    in the same order, so a change to the features cannot go unnoticed.

    ``repair`` is the manifest of the repair applied to the training battles
    (``TimelineRepair.describe()``); None in runs saved before it was recorded.
    """

    source: FeatureSource
    model: Model
    columns: tuple[str, ...]
    repair: Mapping[str, Any] | None = None

    @property
    def feature_set(self) -> str:
        """Name of the feature set the model was trained on."""
        return self.source.name

    @property
    def model_name(self) -> str:
        return self.model.name

    @property
    def name(self) -> str:
        """``<feature set>__<model>``, also the default folder name."""
        return f"{self.feature_set}__{self.model_name}"

    def predict_proba(self, battles: Sequence[Any]) -> np.ndarray:
        return self.model.predict_proba(self._features(battles))

    def predict(self, battles: Sequence[Any], threshold: float = 0.5) -> np.ndarray:
        return self.model.predict(self._features(battles), threshold)

    def check_columns(self, battles: Sequence[Any]) -> None:
        """Raise FeatureMismatchError if the source's columns changed."""
        self._features(battles)

    def save(self, directory: str | Path) -> Path:
        """Save the fitted model, the fitted feature state, the columns and the repair.

        The feature code is not saved: pass the rebuilt feature set to
        :meth:`load`, which restores the learned state onto it.
        """
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        self.model.save(out / RUN_MODEL_FILE)
        stateful = is_stateful(self.source)
        if stateful:
            source = cast(FittableFeatureSource, self.source)
            with (out / FEATURE_STATE_FILE).open("wb") as fh:
                pickle.dump(source.get_state(), fh)
        info = {
            "feature_set": self.feature_set,
            "columns": list(self.columns),
            "model_name": self.model.name,
            "model_class": type(self.model).__name__,
            "stateful_features": stateful,
            "repair": self.repair,
            "saved": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        (out / RUN_INFO_FILE).write_text(json.dumps(info, indent=2))
        return out

    @staticmethod
    def saved_feature_set(directory: str | Path) -> str:
        """Name of the feature set a saved run needs, to rebuild it."""
        info = json.loads((Path(directory) / RUN_INFO_FILE).read_text())
        return str(info["feature_set"])

    @classmethod
    def load(
        cls,
        directory: str | Path,
        feature_set: FeatureSource,
        sample_battles: Sequence[Any] | None = None,
    ) -> TrainedRun:
        """Load a run saved with :meth:`save`.

        ``feature_set`` is the feature source rebuilt from code; its name
        must match the saved one, and its learned state is restored on a
        clone (the object passed in is not modified). With
        ``sample_battles`` the columns are checked immediately; otherwise
        at the first prediction. Load only files you created (pickle).
        """
        src = Path(directory)
        info = json.loads((src / RUN_INFO_FILE).read_text())
        if feature_set.name != info["feature_set"]:
            raise FeatureMismatchError(
                f"model in {src} was trained on feature set "
                f"{info['feature_set']!r}, got {feature_set.name!r}"
            )
        model = Model.load(src / RUN_MODEL_FILE)
        if model.name != info["model_name"]:
            raise ValueError(
                f"{src}: model file is {model.name!r}, run.json says {info['model_name']!r}"
            )
        source = _restore_source(feature_set, src, bool(info["stateful_features"]))
        run = cls(source, model, tuple(info["columns"]), info.get("repair"))
        if sample_battles is not None:
            run.check_columns(sample_battles)
        return run

    def _features(self, battles: Sequence[Any]) -> pd.DataFrame:
        x = self.source.to_frame(battles)
        found = tuple(str(c) for c in x.columns)
        if found != self.columns:
            raise FeatureMismatchError(
                f"feature set {self.feature_set!r} changed since fit: "
                + _describe_column_diff(self.columns, found)
            )
        return x


def _restore_source(feature_set: FeatureSource, directory: Path, stateful: bool) -> FeatureSource:
    """Clone ``feature_set`` and restore the saved state onto it, if any."""
    if not stateful:
        if is_stateful(feature_set):
            raise FeatureMismatchError(
                f"{directory}: saved without feature state, but "
                f"{feature_set.name!r} has stateful features"
            )
        return feature_set
    if not isinstance(feature_set, FittableFeatureSource):
        raise FeatureMismatchError(
            f"{directory}: saved with feature state, but {feature_set.name!r} cannot restore it"
        )
    with (directory / FEATURE_STATE_FILE).open("rb") as fh:
        state = pickle.load(fh)
    restored = feature_set.clone()
    restored.set_state(state)
    return restored


def _describe_column_diff(expected: Sequence[str], found: Sequence[str]) -> str:
    missing = [c for c in expected if c not in found]
    extra = [c for c in found if c not in expected]
    if not missing and not extra:
        return "same columns in a different order"
    parts: list[str] = []
    if missing:
        parts.append(f"missing {missing[:5]}")
    if extra:
        parts.append(f"unexpected {extra[:5]}")
    return ", ".join(parts)
