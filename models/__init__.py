"""Imports every model file, which registers its builder, and exposes the registry.

::

    models = [build("logreg"), build("hgb", learning_rate=0.05), build("ensemble")]
    available()   # names whose library is installed

Keyword arguments override a model's default parameters. A model whose library is not
installed (``lightgbm``, ``xgboost``, ``catboost``) raises ``MissingDependencyError`` when built.
"""

from models import (  # noqa: F401  (imported for the registration)
    catboost,
    catboost_lr,
    gb_lr,
    hgb,
    hgb_ko,
    lightgbm,
    lightgbm_lr,
    logreg,
    xgboost,
    xgboost_lr,
)

# isort: split
from models import ensembles  # noqa: F401  (its members must be registered first)
from models.core.preprocessing import MissingDependencyError
from registry import ENSEMBLES, MODELS, available, build, build_tuned, search_spaces

__all__ = [
    "ENSEMBLES",
    "MODELS",
    "MissingDependencyError",
    "available",
    "build",
    "build_tuned",
    "search_spaces",
]
