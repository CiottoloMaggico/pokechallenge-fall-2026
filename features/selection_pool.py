"""``selection_pool``: compact_v3 plus every other candidate (37 columns).

The starting pool of ``selection.py``. Features with several outputs (switch
pressure, simulation) compute all of them here; ``selection.py`` picks columns.
"""

from __future__ import annotations

from domain.battle import BattleContext
from feature_set import FeatureSet
from features.common import make_set, type_chart
from features.compact_v3 import compact_v3_features
from features.core.encoding import OpponentSpeciesEncoding
from features.core.hazards import phazing_uses, spikes_placed, spikes_spun
from features.core.knockouts import HiddenKnockoutProbability, unsure_hidden_knockouts
from features.core.material import (
    alive_share,
    other_sleep_share,
    p1_mean_hp_alive,
    p2_revealed_share,
    rest_sleep_share,
    statused_share,
)
from features.core.simulation import BattleSimulation
from features.core.switching import NextSwitchProbability, SwitchPressure, switch_count_feature
from features.core.team import hpw_def
from features.core.turn import AtTurn, share_difference, side_difference
from interfaces import Feature
from registry import feature_set


@feature_set("selection_pool")
def selection_pool(turn: int | None = None) -> FeatureSet[BattleContext]:
    """compact_v3 and the other candidate features, for ``selection.py``."""
    chart = type_chart()
    core = [
        f for f in compact_v3_features(turn) if not isinstance(f, SwitchPressure | BattleSimulation)
    ]
    extra: list[Feature[BattleContext]] = [
        SwitchPressure(turn, outputs=SwitchPressure.OUTPUTS),
        BattleSimulation(chart, turn),
        AtTurn("p2_hidden_ko_unsure", unsure_hidden_knockouts, turn),
        HiddenKnockoutProbability(chart, turn),
        AtTurn("hpw_def", hpw_def, turn),
        AtTurn("material_alive", side_difference(alive_share), turn),
        AtTurn("material_statused", side_difference(statused_share, bad=True), turn),
        AtTurn("p1_mean_hp_alive", p1_mean_hp_alive, turn),
        AtTurn("material_asleep_rest", side_difference(rest_sleep_share, bad=True), turn),
        AtTurn("material_asleep_other", side_difference(other_sleep_share, bad=True), turn),
        AtTurn("p2_revealed", p2_revealed_share, turn),
        AtTurn("spikes_placed", share_difference(spikes_placed), turn),
        AtTurn("spikes_spun", share_difference(spikes_spun), turn),
        AtTurn("phazing", share_difference(phazing_uses), turn),
        switch_count_feature(turn),
        NextSwitchProbability(turn),
        OpponentSpeciesEncoding(turn),
    ]
    return make_set("selection_pool", turn, [*core, *extra])
