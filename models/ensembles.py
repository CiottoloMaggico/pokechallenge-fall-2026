"""Averages of other registered models: ``ensemble`` and the ones that need extra libraries.

* ``ensemble``         logreg + hgb
* ``ensemble_boost``   logreg + lightgbm + xgboost + catboost
* ``ensemble_all``     logreg + hgb + lightgbm + xgboost + catboost
* ``ensemble_from_lr`` gb_lr + catboost_lr + lightgbm_lr + xgboost_lr
"""

from __future__ import annotations

from collections.abc import Sequence

from interfaces import Model
from models.core.ensemble import EnsembleModel
from registry import ENSEMBLES, build, ensemble


def average(name: str, weights: Sequence[float] | None = None) -> Model:
    """The registered ensemble ``name``, its members built with their default parameters."""
    return EnsembleModel(name, [build(member) for member in ENSEMBLES[name]], weights)


@ensemble("ensemble", members=("logreg", "hgb"))
def build_ensemble(name: str = "ensemble", weights: Sequence[float] | None = None) -> Model:
    return average(name, weights)


@ensemble("ensemble_boost", members=("logreg", "lightgbm", "xgboost", "catboost"))
def build_ensemble_boost(
    name: str = "ensemble_boost", weights: Sequence[float] | None = None
) -> Model:
    return average(name, weights)


@ensemble("ensemble_all", members=("logreg", "hgb", "lightgbm", "xgboost", "catboost"))
def build_ensemble_all(name: str = "ensemble_all", weights: Sequence[float] | None = None) -> Model:
    return average(name, weights)


@ensemble("ensemble_from_lr", members=("gb_lr", "catboost_lr", "lightgbm_lr", "xgboost_lr"))
def build_ensemble_from_lr(
    name: str = "ensemble_from_lr", weights: Sequence[float] | None = None
) -> Model:
    return average(name, weights)
