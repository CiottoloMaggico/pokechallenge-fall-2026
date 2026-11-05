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

__all__ = [
    "BUILDERS",
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


# named ensembles: name -> members (all averaged with equal weights)
ENSEMBLES: dict[str, tuple[str, ...]] = {
    "ensemble_boost": ("logreg", "lightgbm", "xgboost", "catboost"),
    "ensemble_all": ("logreg", "hgb", "lightgbm", "xgboost", "catboost"),
    "ensemble_from_lr": ("gb_lr", "catboost_lr", "lightgbm_lr", "xgboost_lr"),
}


def _named_ensemble(key: str) -> Callable[..., Model]:
    def builder(name: str = key, weights: Sequence[float] | None = None) -> Model:
        return ensemble(name, ENSEMBLES[key], weights)

    return builder


BUILDERS: dict[str, Callable[..., Model]] = {
    "logreg": logreg.build,
    "hgb": hgb.build,
    "lightgbm": lightgbm_model.build,
    "xgboost": xgboost_model.build,
    "catboost": catboost_model.build,
    "gb_lr": from_logreg.build_sklearn,
    "catboost_lr": from_logreg.build_catboost,
    "lightgbm_lr": from_logreg.build_lightgbm,
    "xgboost_lr": from_logreg.build_xgboost,
    "blend_v11": logit_blend.build,
    "ensemble": ensemble,
    **{key: _named_ensemble(key) for key in ENSEMBLES},
}

# model name -> hyperparameter search space (see models/space.py)
SEARCH_SPACES: dict[str, Callable[[Trial], dict[str, Any]]] = {
    "logreg": logreg.search_space,
    "hgb": hgb.search_space,
    "lightgbm": lightgbm_model.search_space,
    "xgboost": xgboost_model.search_space,
    "catboost": catboost_model.search_space,
    "gb_lr": from_logreg.search_space_sklearn,
    "catboost_lr": from_logreg.search_space_catboost,
    "lightgbm_lr": from_logreg.search_space_lightgbm,
    "xgboost_lr": from_logreg.search_space_xgboost,
}

# model name -> optional library it needs
REQUIRED_MODULE = {
    "lightgbm": lightgbm_model.LIBRARY,
    "xgboost": xgboost_model.LIBRARY,
    "catboost": catboost_model.LIBRARY,
    "catboost_lr": catboost_model.LIBRARY,
    "lightgbm_lr": lightgbm_model.LIBRARY,
    "xgboost_lr": xgboost_model.LIBRARY,
}


def build(model: str, /, **params: Any) -> Model:
    """Build the catalog model ``model``; ``params`` override its defaults.

    ``name=`` among ``params`` renames the built model (e.g. "hgb-tuned").
    """
    try:
        builder = BUILDERS[model]
    except KeyError:
        choices = sorted(BUILDERS)
        raise KeyError(f"unknown model {model!r}; choose from {choices}") from None
    return builder(**params)


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
    return [name for name in BUILDERS if all(map(_installed, _needs(name)))]


def _needs(name: str) -> tuple[str, ...]:
    members = ENSEMBLES.get(name, (name,))
    return tuple(REQUIRED_MODULE[m] for m in members if m in REQUIRED_MODULE)


def _installed(module: str) -> bool:
    return importlib.util.find_spec(module) is not None
