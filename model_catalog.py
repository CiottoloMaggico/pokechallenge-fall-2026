"""Registry of the concrete models in ``models/``, built by name.

::

    models = [build("logreg"), build("hgb", learning_rate=0.05), build("ensemble")]
    available()   # names whose library is installed

* ``logreg``    median imputation, scaling, L1 logistic regression
* ``hgb``       scikit-learn HistGradientBoosting
* ``lightgbm``  LightGBM          (optional: ``pip install lightgbm``)
* ``xgboost``   XGBoost           (optional: ``pip install xgboost``)
* ``catboost``  CatBoost          (optional: ``pip install catboost``)
* ``ensemble``  average of other catalog models (default logreg + hgb)
* ``ensemble_boost``  logreg + lightgbm + xgboost + catboost
* ``ensemble_all``    logreg + hgb + lightgbm + xgboost + catboost
* ``gb_lr`` / ``catboost_lr`` / ``lightgbm_lr`` / ``xgboost_lr``
                gradient boosting that starts from the logistic regression's
                log-odds and learns only a correction (``models/from_logreg.py``)
* ``ensemble_from_lr`` gb_lr + catboost_lr + lightgbm_lr + xgboost_lr
* ``blend_v11``  logit blend of a catalog model (default ensemble_from_lr)
                with the ``v11_proba`` column of the ``*_v11`` feature sets
                (``models/logit_blend.py``; w from ``tools/v11_blend_weight.py``)

Keyword arguments override a model's default parameters.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import registry
from model import Model
from models import (
    catboost_model,
    from_logreg,
    hgb,
    lightgbm_model,
    logit_blend,
    logreg,
    xgboost_model,
)
from models.ensemble import DEFAULT_MEMBERS
from models.ensemble import build as build_ensemble
from models.preprocessing import MissingDependencyError
from models.space import Trial
from registry import ENSEMBLES, MODELS

__all__ = [
    "ENSEMBLES",
    "MODELS",
    "SEARCH_SPACES",
    "MissingDependencyError",
    "available",
    "build",
    "ensemble",
]


def ensemble(
    name: str = "ensemble",
    members: Sequence[str | Model] = DEFAULT_MEMBERS,
    weights: Sequence[float] | None = None,
) -> Model:
    """Average of catalog models (given by name) or Model objects."""
    built = [m if isinstance(m, Model) else build(m) for m in members]
    return build_ensemble(name, built, weights)


def _named_ensemble(key: str) -> Callable[..., Model]:
    def builder(name: str = key, weights: Sequence[float] | None = None) -> Model:
        return ensemble(name, ENSEMBLES[key], weights)

    return builder


registry.model("logreg", logreg.search_space)(logreg.build)
registry.model("hgb", hgb.search_space)(hgb.build)
registry.model("lightgbm", lightgbm_model.search_space, (lightgbm_model.LIBRARY,))(
    lightgbm_model.build
)
registry.model("xgboost", xgboost_model.search_space, (xgboost_model.LIBRARY,))(xgboost_model.build)
registry.model("catboost", catboost_model.search_space, (catboost_model.LIBRARY,))(
    catboost_model.build
)
registry.model("gb_lr", from_logreg.search_space_sklearn)(from_logreg.build_sklearn)
registry.model("catboost_lr", from_logreg.search_space_catboost, (catboost_model.LIBRARY,))(
    from_logreg.build_catboost
)
registry.model("lightgbm_lr", from_logreg.search_space_lightgbm, (lightgbm_model.LIBRARY,))(
    from_logreg.build_lightgbm
)
registry.model("xgboost_lr", from_logreg.search_space_xgboost, (xgboost_model.LIBRARY,))(
    from_logreg.build_xgboost
)
registry.model("blend_v11")(logit_blend.build)
registry.model("ensemble")(ensemble)
for _name, _members in (
    ("ensemble_boost", ("logreg", "lightgbm", "xgboost", "catboost")),
    ("ensemble_all", ("logreg", "hgb", "lightgbm", "xgboost", "catboost")),
    ("ensemble_from_lr", ("gb_lr", "catboost_lr", "lightgbm_lr", "xgboost_lr")),
):
    registry.ensemble(_name, _members)(_named_ensemble(_name))

SEARCH_SPACES: dict[str, Callable[[Trial], dict[str, Any]]] = {
    spec.name: spec.search_space for spec in MODELS.values() if spec.search_space is not None
}


def build(model: str, /, **params: Any) -> Model:
    """Build the catalog model ``model``; ``params`` override its defaults.

    ``name=`` among ``params`` renames the built model (e.g. "hgb-tuned").
    """
    return MODELS[model].build(**params)


def build_tuned(model: str, tuned: Mapping[str, Mapping[str, Any]]) -> Model:
    """Build ``model`` with tuned parameters (``tuned``: model -> params).

    A single model found in ``tuned`` gets its parameters and is named
    ``<model>-tuned``; an ensemble is rebuilt from its members, each with
    its tuned parameters when available, and named ``<ensemble>-tuned``
    if at least one member was tuned. Anything else is built as usual.
    """
    if model in ENSEMBLES or model == "ensemble":
        members = ENSEMBLES.get(model, DEFAULT_MEMBERS)
        if not any(m in tuned for m in members):
            return build(model)
        built = [build_tuned(m, tuned) for m in members]
        return ensemble(f"{model}-tuned", built)
    if model in tuned:
        return build(model, name=f"{model}-tuned", **tuned[model])
    return build(model)


def available() -> list[str]:
    """Catalog names whose libraries are all installed."""
    return [spec.name for spec in MODELS.values() if all(map(_installed, spec.requires))]


def _installed(module: str) -> bool:
    return importlib.util.find_spec(module) is not None
