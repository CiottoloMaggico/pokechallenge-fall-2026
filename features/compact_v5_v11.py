"""``compact_v5_v11``: compact_v5 + ``v11_proba`` (``features/v11_bridge.py``).

For the ``blend_v11`` model only: v11 is trained inside each fold, the column is
NaN on training battles. Needs the v11 repository (``V11_REPO``).
"""

from __future__ import annotations

from domain.battle import BattleContext
from feature_set import FeatureSet
from features.common import make_set
from features.compact_v5 import compact_v5
from registry import feature_set


@feature_set("compact_v5_v11", requires=("catboost", "lightgbm", "xgboost"))
def compact_v5_v11(turn: int | None = None) -> FeatureSet[BattleContext]:
    """compact_v5 + ``v11_proba``; only defined at the last turn."""
    from features.v11_bridge import V11Probability

    if turn is not None:
        raise ValueError("compact_v5_v11 is only defined at the last turn")
    extra = make_set("v11", None, [V11Probability()])
    return compact_v5(None).merge(extra, name="compact_v5_v11")
