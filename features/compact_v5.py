"""``compact_v5``: compact_v4 re-selected with logistic regression on the pool features.

After adding the team-pool features (``features.core.pool``): team ratings
``tr_p1`` / ``tr_p2`` in place of ``te_p1_team``, ``phpw_spd`` in place of
``hpw_spd``, plus ``smat_hp``, ``hpw_spd_t20``, ``b_stat_diff``,
``alive_gain_10_30``, ``p2_fainted`` (26 columns in ``compact_v5_orig``). The
multi-seed revalidation dropped ``p2_hidden_ko_det``, ``material_hp``
(``smat_hp`` replaces it), ``mv_res``, ``hp_healed``, ``ghost_ko_50``,
``act_duel``, ``alive_gain_10_30`` and ``eff_trapped``: 18 columns, accuracy and
log loss equal to the original within noise.
"""

from __future__ import annotations

from domain.battle import BattleContext
from feature_set import FeatureSet
from features.common import make_set, type_chart
from features.core.field import boost_sum, substitute_up, trapped
from features.core.hazards import spikes_on_field
from features.core.knockouts import (
    expected_hidden_knockouts,
    ghost_knockouts,
    unsure_hidden_knockouts,
)
from features.core.matchup import active_duel
from features.core.material import healthy_share, hp_healed, hp_share
from features.core.moves import move_features
from features.core.pool import (
    PairEncoding,
    PoolBulk,
    TeamRating,
    alive_gain_10_30,
    p2_fainted,
    soft_material_hp,
    status_move_share_diff,
)
from features.core.simulation import BattleSimulation
from features.core.switching import SwitchPressure
from features.core.team import HiddenTeamStrength, hpw_spd, snorlax_diff
from features.core.turn import AtTurn, side_difference
from interfaces import Feature
from registry import feature_set, set_name

REVALIDATION_DROPS = (
    "p2_hidden_ko_det",
    "material_hp",
    "mv_res",
    "hp_healed",
    "ghost_ko_50",
    "act_duel",
    "alive_gain_10_30",
    "eff_trapped",
)


@feature_set("compact_v5_orig")
def compact_v5_orig(turn: int | None = None) -> FeatureSet[BattleContext]:
    """compact_v4 re-selected with logistic regression on the pool features (26)."""
    chart = type_chart()
    hpw_turn = 20 if turn is None else min(20, turn)
    features: list[Feature[BattleContext]] = [
        AtTurn("material_hp", side_difference(hp_share), turn),
        AtTurn("material_healthy", side_difference(healthy_share), turn),
        AtTurn("spikes_on_field", spikes_on_field, turn),
        AtTurn("bst_sum", boost_sum, turn),
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
        PairEncoding(turn),
        BattleSimulation(chart, turn, outputs=("sim_seen_left",)),
        AtTurn("p2_hidden_ko_unsure", unsure_hidden_knockouts, turn),
        TeamRating(turn),
        PoolBulk(turn),
        AtTurn("hpw_spd_t20", hpw_spd, hpw_turn),
        AtTurn("smat_hp", soft_material_hp, turn),
        AtTurn("b_stat_diff", status_move_share_diff, turn),
        AtTurn("alive_gain_10_30", alive_gain_10_30, turn),
        AtTurn("p2_fainted", p2_fainted, turn),
    ]
    return make_set("compact_v5_orig", turn, features)


@feature_set("compact_v5")
def compact_v5(turn: int | None = None) -> FeatureSet[BattleContext]:
    """compact_v5 after the multi-seed revalidation."""
    return compact_v5_orig(turn).without(REVALIDATION_DROPS, set_name("compact_v5", turn))
