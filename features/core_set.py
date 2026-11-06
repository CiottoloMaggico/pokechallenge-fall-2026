"""``core``: material + hazards + switch.

Named ``core_set`` because ``features/core/`` is the package of the concrete features.
"""

from __future__ import annotations

from domain.battle import BattleContext
from feature_set import FeatureSet
from features.common import make_set
from features.core.hazards import hazard_features
from features.core.material import material_features
from features.core.switching import NextSwitchProbability
from registry import feature_set


@feature_set("core")
def core(turn: int | None = None) -> FeatureSet[BattleContext]:
    """Material, hazard and next-switch features."""
    features = [
        *material_features(turn),
        *hazard_features(turn),
        NextSwitchProbability(turn),
    ]
    return make_set("core", turn, features)
