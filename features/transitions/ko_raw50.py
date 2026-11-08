"""``ko_raw50``: the 50 columns of ``domain.transitions`` the detector was first trained on.

Every column of ``ALL_COLUMNS`` except the three later codes that replace groups of them
(``a_prio_phaze``, ``x_status``, ``y_code``). Out of fold on P1's switches (3 seeds) log loss
0.0984, AUC 0.9915; superseded by ``ko_domain30``, equal within noise with 20 fewer columns.
"""

from __future__ import annotations

from domain.transitions import ALL_COLUMNS, Transition
from feature_set import FeatureSet
from features.transitions.columns import transition_set

LATER_CODES = ("a_prio_phaze", "x_status", "y_code")
COLUMNS = tuple(c for c in ALL_COLUMNS if c not in LATER_CODES)


def build() -> FeatureSet[Transition]:
    """The 50 original columns."""
    return transition_set("ko_raw50", COLUMNS)
