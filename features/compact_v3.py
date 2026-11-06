"""``compact_v3``: compact_v2 with the learned detector's expected knockouts.

``p2_hidden_ko_det`` replaces ``p2_hidden_ko_soft`` (19 columns in
``compact_v3_orig``). The multi-seed revalidation dropped ``p2_hidden_ko_det``
(the repair applied to every set already carries the detector), ``ghost_ko_50``,
``act_duel`` and ``mv_res``: 15 columns, the same as ``compact_v2``.
"""

from __future__ import annotations

from domain.battle import BattleContext
from feature_set import FeatureSet
from features.common import make_set, type_chart
from features.core.encoding import TeamEncoding
from features.core.field import boost_sum, substitute_up, trapped
from features.core.hazards import spikes_on_field
from features.core.knockouts import expected_hidden_knockouts, ghost_knockouts
from features.core.matchup import active_duel
from features.core.material import healthy_share, hp_healed, hp_share
from features.core.moves import move_features
from features.core.simulation import BattleSimulation
from features.core.switching import SwitchPressure
from features.core.team import HiddenTeamStrength, hpw_spd, snorlax_diff
from features.core.turn import AtTurn, side_difference
from interfaces import Feature
from registry import feature_set, set_name

REVALIDATION_DROPS = ("p2_hidden_ko_det", "ghost_ko_50", "act_duel", "mv_res")


def compact_v3_features(turn: int | None) -> list[Feature[BattleContext]]:
    """The 19 features of ``compact_v3_orig``, the base of v4 and ``selection_pool``."""
    chart = type_chart()
    return [
        AtTurn("material_hp", side_difference(hp_share), turn),
        AtTurn("material_healthy", side_difference(healthy_share), turn),
        AtTurn("spikes_on_field", spikes_on_field, turn),
        AtTurn("bst_sum", boost_sum, turn),
        AtTurn("hpw_spd", hpw_spd, turn),
        AtTurn("eff_substitute", substitute_up, turn),
        AtTurn("eff_trapped", trapped, turn),
        AtTurn("hp_healed", hp_healed, turn),
        AtTurn("act_duel", active_duel(chart), turn),
        AtTurn("ghost_ko_50", ghost_knockouts, turn),
        AtTurn("snorlax_diff", snorlax_diff, turn),
        AtTurn("p2_hidden_ko_det", expected_hidden_knockouts, turn),
        *move_features(chart, turn),
        SwitchPressure(turn, outputs=("p_switch_diff", "switch_excess_all")),
        HiddenTeamStrength(turn),
        TeamEncoding(turn),
        BattleSimulation(chart, turn, outputs=("sim_seen_left",)),
    ]


@feature_set("compact_v3_orig")
def compact_v3_orig(turn: int | None = None) -> FeatureSet[BattleContext]:
    """compact_v2 with the learned detector's expected knockouts (19)."""
    return make_set("compact_v3_orig", turn, compact_v3_features(turn))


@feature_set("compact_v3")
def compact_v3(turn: int | None = None) -> FeatureSet[BattleContext]:
    """compact_v3 after the multi-seed revalidation."""
    return compact_v3_orig(turn).without(REVALIDATION_DROPS, set_name("compact_v3", turn))
