"""``v11``: the features of the FDS repository's ``blend_v11``, ported natively.

One feature (``features.core.v11.feature.V11Features``) runs v11's whole pipeline: its own
Pokédex and hidden-faint detector, the feature groups of the four pipelines of ``blend_v11``
(static, state, trajectory, actives, events, matchup, win conditions, sleep, species / HP / move
bags, team ids, simulations, behaviour, condition) and the learned forecast. The columns are
known after ``fit`` and follow the original matrix order. Fixed at the last recorded turn (the
forecast already looks at truncated timelines).
"""

from __future__ import annotations

from domain.battle import BattleContext
from feature_set import FeatureSet
from features.common import make_set
from registry import feature_set


@feature_set("v11", requires=("lightgbm",))
def v11(turn: int | None = None) -> FeatureSet[BattleContext]:
    """v11's columns (the ``v11_*`` models read them)."""
    if turn is not None:
        raise ValueError("v11 is fixed at the last turn: drop the @t suffix")
    # Imported here so that listing the feature sets does not need LightGBM.
    from features.core.v11.feature import V11Features

    return make_set("v11", None, [V11Features()])
