"""``material``: HP, alive, statused, healthy, sleep, revealed (8 columns)."""

from __future__ import annotations

from domain.battle import BattleContext
from feature_set import FeatureSet
from features.common import make_set
from features.core.material import material_features
from registry import feature_set


@feature_set("material")
def material(turn: int | None = None) -> FeatureSet[BattleContext]:
    """The material features of ``features.core.material``."""
    return make_set("material", turn, material_features(turn))
