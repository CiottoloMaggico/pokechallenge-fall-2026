"""``compact_v2``: compact_ko re-selected on the corrected replay.

The replay fixed stale moves and per-turn knockouts. ``compact_v2_orig`` (19
columns) adds ``p2_hidden_ko_soft``, ``ghost_ko_50``, ``snorlax_diff`` and
``switch_excess_all``. The multi-seed revalidation dropped the hidden knockout
probability (the repair applied to every set already carries the detector),
``ghost_ko_50``, ``act_duel`` and ``mv_res``: 15 columns.
"""

from __future__ import annotations

from domain.battle import BattleContext
from feature_set import FeatureSet
from features.common import make_set, type_chart
from features.core.encoding import TeamEncoding
from features.core.field import boost_sum, substitute_up, trapped
from features.core.hazards import spikes_on_field
from features.core.knockouts import HiddenKnockoutProbability, ghost_knockouts
from features.core.matchup import active_duel
from features.core.material import healthy_share, hp_healed, hp_share
from features.core.moves import move_features
from features.core.simulation import BattleSimulation
from features.core.switching import SwitchPressure
from features.core.team import HiddenTeamStrength, hpw_spd, snorlax_diff
from features.core.turn import AtTurn, side_difference
from interfaces import Feature
from registry import feature_set, set_name

REVALIDATION_DROPS = ("hidden_knockout_probability", "ghost_ko_50", "act_duel", "mv_res")


@feature_set("compact_v2_orig")
def compact_v2_orig(turn: int | None = None) -> FeatureSet[BattleContext]:
    """compact_ko re-selected on the corrected replay (19 features)."""
    chart = type_chart()
    features: list[Feature[BattleContext]] = [
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
        *move_features(chart, turn),
        SwitchPressure(turn, outputs=("p_switch_diff", "switch_excess_all")),
        HiddenTeamStrength(turn),
        TeamEncoding(turn),
        BattleSimulation(chart, turn, outputs=("sim_seen_left",)),
        HiddenKnockoutProbability(chart, turn),
    ]
    return make_set("compact_v2_orig", turn, features)


@feature_set("compact_v2")
def compact_v2(turn: int | None = None) -> FeatureSet[BattleContext]:
    """compact_v2 after the multi-seed revalidation."""
    return compact_v2_orig(turn).without(REVALIDATION_DROPS, set_name("compact_v2", turn))
