"""``compact_plus``: compact + team encoding and battle simulation (17 columns)."""

from __future__ import annotations

from domain.battle import BattleContext
from feature_set import FeatureSet
from features.common import make_set, type_chart
from features.compact import compact_orig
from features.core.encoding import TeamEncoding
from features.core.simulation import BattleSimulation
from registry import feature_set, set_name


@feature_set("compact_plus")
def compact_plus(turn: int | None = None) -> FeatureSet[BattleContext]:
    """``compact`` plus ``te_p1_team``, ``te_pair_mean``, ``sim_seen_left``."""
    extra = make_set(
        "extra",
        turn,
        [
            TeamEncoding(turn),
            BattleSimulation(type_chart(), turn, outputs=("sim_seen_left",)),
        ],
    )
    return compact_orig(turn).merge(extra, name=set_name("compact_plus", turn))
