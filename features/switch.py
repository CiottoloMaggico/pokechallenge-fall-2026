"""``switch``: next-switch probability of both players (2 columns)."""

from __future__ import annotations

from domain.battle import BattleContext
from feature_set import FeatureSet
from features.common import make_set
from features.core.switching import NextSwitchProbability
from registry import feature_set


@feature_set("switch")
def switch(turn: int | None = None) -> FeatureSet[BattleContext]:
    """The next-switch probability of both players."""
    return make_set("switch", turn, [NextSwitchProbability(turn)])
