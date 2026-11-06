"""``compact_ko``: the features selected on the knockout-repaired replay.

Selected on the replay repaired by ``domain.transitions``; use it with repaired
datasets (``main.py`` always repairs). Superseded by ``compact_v2``.
"""

from __future__ import annotations

from domain.battle import BattleContext
from feature_set import FeatureSet
from features.common import make_set, type_chart
from features.core.encoding import TeamEncoding
from features.core.field import boost_sum, substitute_up, trapped
from features.core.hazards import spikes_on_field
from features.core.matchup import active_duel
from features.core.material import alive_share, healthy_share, hp_healed, hp_share
from features.core.moves import move_features
from features.core.simulation import BattleSimulation
from features.core.switching import SwitchPressure, switch_count_feature
from features.core.team import HiddenTeamStrength, hpw_def, hpw_spd
from features.core.turn import AtTurn, side_difference
from interfaces import Feature
from registry import feature_set


@feature_set("compact_ko")
def compact_ko(turn: int | None = None) -> FeatureSet[BattleContext]:
    """The 18 features on the knockout-repaired replay."""
    chart = type_chart()
    features: list[Feature[BattleContext]] = [
        AtTurn("material_hp", side_difference(hp_share), turn),
        AtTurn("material_healthy", side_difference(healthy_share), turn),
        AtTurn("spikes_on_field", spikes_on_field, turn),
        AtTurn("bst_sum", boost_sum, turn),
        AtTurn("hpw_spd", hpw_spd, turn),
        AtTurn("hpw_def", hpw_def, turn),
        AtTurn("eff_substitute", substitute_up, turn),
        AtTurn("eff_trapped", trapped, turn),
        AtTurn("hp_healed", hp_healed, turn),
        AtTurn("act_duel", active_duel(chart), turn),
        AtTurn("material_alive", side_difference(alive_share), turn),
        *move_features(chart, turn),
        SwitchPressure(turn),
        switch_count_feature(turn),
        HiddenTeamStrength(turn),
        TeamEncoding(turn),
        BattleSimulation(chart, turn, outputs=("sim_seen_left",)),
    ]
    return make_set("compact_ko", turn, features)
