"""``hazards``: Spikes placed / spun / on field, phazing (4 columns)."""

from __future__ import annotations

from domain.battle import BattleContext
from feature_set import FeatureSet
from features.common import make_set
from features.core.hazards import hazard_features
from registry import feature_set


@feature_set("hazards")
def hazards(turn: int | None = None) -> FeatureSet[BattleContext]:
    """The hazard features of ``features.core.hazards``."""
    return make_set("hazards", turn, hazard_features(turn))
