"""``v11``: the features of the FDS repository's ``blend_v11``, on this repository's constructs.

One ``Feature`` per group of the original's matrix, in its order (``features.core.v11.groups``):
static, state and trajectory, actives, events, matchup, win conditions, sleep, species / HP /
move bags, team ids, duels, behaviour, second simulation, condition; then the learned forecast of
the next turns (``features.core.v11.forecast``). Each reads the replayed ``BattleContext`` and
the P2 faint probabilities of its repair (the reference is ``--repair v11=…``, any repair
works); every feature is label-free and fits on the training battles only (``train`` and
``evaluate`` refuse ``--test`` with v11). Fixed at the last recorded turn (the forecast already
looks at truncated timelines).
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
    from features.core.v11 import groups as g
    from features.core.v11.forecast import V11Forecast

    return make_set(
        "v11",
        None,
        [
            g.V11Static(),
            g.V11State(),
            g.V11Active(),
            g.V11Events(),
            g.V11Matchup(),
            g.V11WinConditions(),
            g.V11Sleep(),
            g.V11Bags(),
            g.V11TeamKeys(),
            g.V11Duel(),
            g.V11Behaviour(),
            g.V11Simulation(),
            g.V11Condition(),
            V11Forecast(),
        ],
    )
