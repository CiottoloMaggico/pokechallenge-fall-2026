"""``hp30``: only the HP difference at the last recorded turn (fixed turn)."""

from __future__ import annotations

from domain.battle import BattleContext
from feature_set import FeatureSet
from interfaces import FeatureValue
from registry import feature_set


def hp_diff_t30(context: BattleContext) -> FeatureValue:
    """P1 total HP minus P2 total HP at the last recorded turn.

    The timelines are cut at turn 30, so for most battles this is the
    situation at turn 30; shorter battles use their final turn.
    """
    last = context.last
    return "hp_diff_t30", last.p1.hp_total - last.p2.hp_total


@feature_set("hp30")
def hp30(turn: int | None = None) -> FeatureSet[BattleContext]:
    """Only the HP difference at the last recorded turn."""
    if turn is not None:
        raise ValueError("hp30 is fixed at the last turn: use 'material' instead")
    features = FeatureSet[BattleContext]("hp30")
    features.register(hp_diff_t30)
    return features
