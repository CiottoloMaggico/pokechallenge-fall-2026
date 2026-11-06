"""Helpers shared by the feature set files."""

from __future__ import annotations

from collections.abc import Iterable
from functools import cache
from pathlib import Path

from domain.battle import BattleContext
from domain.typechart import TypeChart
from feature_set import FeatureSet
from interfaces import Feature
from registry import set_name

TYPE_CHART_PATH = Path(__file__).resolve().parent.parent / "assets" / "type_chart_gen2.csv"


@cache
def type_chart() -> TypeChart:
    """The Gen 2 type chart, loaded once."""
    return TypeChart.from_csv(TYPE_CHART_PATH)


def make_set(
    base: str, turn: int | None, features: Iterable[Feature[BattleContext]]
) -> FeatureSet[BattleContext]:
    """A feature set called ``set_name(base, turn)`` holding ``features``."""
    built = FeatureSet[BattleContext](set_name(base, turn))
    built.register_all(features)
    return built
