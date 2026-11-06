"""``compact``: the features kept by the incremental selection (see the selection report).

``compact_orig`` has 14 features. The multi-seed revalidation (6 seeds x 5 folds,
corrected repeated k-fold t-test, one-sided corrected p >= 0.20, removed one at a
time, least harmful first) dropped ``mv_res``, ``hidden_bst`` and ``te_p2_mean``,
leaving 11 columns with accuracy and log loss equal to the original within noise.
"""

from __future__ import annotations

from domain.battle import BattleContext
from feature_set import FeatureSet
from features.common import make_set, type_chart
from features.core.encoding import OpponentSpeciesEncoding
from features.core.field import boost_sum, substitute_up, trapped
from features.core.hazards import spikes_on_field
from features.core.material import (
    hp_share,
    p1_mean_hp_alive,
    rest_sleep_share,
    statused_share,
)
from features.core.moves import move_features
from features.core.switching import SwitchPressure
from features.core.team import HiddenTeamStrength, hpw_spd
from features.core.turn import AtTurn, side_difference
from interfaces import Feature
from registry import feature_set, set_name

REVALIDATION_DROPS = ("hidden_team_strength", "opponent_species_encoding", "mv_res")


@feature_set("compact_orig")
def compact_orig(turn: int | None = None) -> FeatureSet[BattleContext]:
    """The 14 features kept by the incremental selection."""
    features: list[Feature[BattleContext]] = [
        AtTurn("material_hp", side_difference(hp_share), turn),
        AtTurn("material_statused", side_difference(statused_share, bad=True), turn),
        AtTurn("p1_mean_hp_alive", p1_mean_hp_alive, turn),
        AtTurn("material_asleep_rest", side_difference(rest_sleep_share, bad=True), turn),
        AtTurn("spikes_on_field", spikes_on_field, turn),
        AtTurn("bst_sum", boost_sum, turn),
        AtTurn("hpw_spd", hpw_spd, turn),
        AtTurn("eff_substitute", substitute_up, turn),
        AtTurn("eff_trapped", trapped, turn),
        *move_features(type_chart(), turn),
        SwitchPressure(turn),
        HiddenTeamStrength(turn),
        OpponentSpeciesEncoding(turn),
    ]
    return make_set("compact_orig", turn, features)


@feature_set("compact")
def compact(turn: int | None = None) -> FeatureSet[BattleContext]:
    """compact after the multi-seed revalidation."""
    return compact_orig(turn).without(REVALIDATION_DROPS, set_name("compact", turn))
