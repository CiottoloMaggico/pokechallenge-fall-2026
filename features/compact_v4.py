"""``compact_v4``: compact_v3 + ``p2_hidden_ko_unsure``.

The only change kept by ``selection.py`` on the detector-repaired replay: the
knockouts the repair left unmarked (20 columns in ``compact_v4_orig``). The
multi-seed revalidation dropped the same four columns as v3: 16 columns.
"""

from __future__ import annotations

from domain.battle import BattleContext
from feature_set import FeatureSet
from features.common import make_set
from features.compact_v3 import compact_v3_features
from features.core.knockouts import unsure_hidden_knockouts
from features.core.turn import AtTurn
from registry import feature_set, set_name

REVALIDATION_DROPS = ("p2_hidden_ko_det", "ghost_ko_50", "act_duel", "mv_res")


@feature_set("compact_v4_orig")
def compact_v4_orig(turn: int | None = None) -> FeatureSet[BattleContext]:
    """compact_v3 plus the knockouts the repair left unmarked (20)."""
    features = [
        *compact_v3_features(turn),
        AtTurn("p2_hidden_ko_unsure", unsure_hidden_knockouts, turn),
    ]
    return make_set("compact_v4_orig", turn, features)


@feature_set("compact_v4")
def compact_v4(turn: int | None = None) -> FeatureSet[BattleContext]:
    """compact_v4 after the multi-seed revalidation."""
    return compact_v4_orig(turn).without(REVALIDATION_DROPS, set_name("compact_v4", turn))
