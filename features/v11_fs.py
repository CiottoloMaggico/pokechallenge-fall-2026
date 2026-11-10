"""``v11_fs``: the living line of v11, with the label-aware steps of its members as features.

The label-free groups of ``v11`` (from the package ``features.core.v11_fs``, a copy that owns its
code; the frozen ``v11`` set is never modified), in v11's order without the bags and team keys;
then the splines of the spline member's keys (``V11Splines``), the logit of the bags
(``V11BagLogit``, ``stack_logit_0``) and the team effect against a base logit of every non-key
column of v11 (``V11TeamEffect``, ``tm*``), the last two label-aware (out of fold on their
training rows). The generic models (``models/stacked.py``) read it with no column name. Fixed at
the last turn; ``train`` / ``evaluate`` refuse ``--test`` (the label-free features fit on the
training battles only).
"""

from __future__ import annotations

from domain.battle import BattleContext
from feature_set import FeatureSet
from features.common import make_set
from registry import feature_set


@feature_set("v11_fs", requires=("lightgbm",))
def v11_fs(turn: int | None = None) -> FeatureSet[BattleContext]:
    """v11's label-free columns, splines, bag logit and team effect."""
    if turn is not None:
        raise ValueError("v11_fs is fixed at the last turn: drop the @t suffix")
    # Imported here so that listing the feature sets does not need LightGBM.
    from features.core.v11_fs import groups as g
    from features.core.v11_fs import label_aware as la
    from features.core.v11_fs.forecast import V11Forecast

    return make_set(
        "v11_fs",
        None,
        [
            g.V11Static(),
            g.V11State(),
            g.V11Active(),
            g.V11Events(),
            g.V11Matchup(),
            g.V11WinConditions(),
            g.V11Sleep(),
            g.V11Duel(),
            g.V11Behaviour(),
            g.V11Simulation(),
            g.V11Condition(),
            V11Forecast(),
            la.V11Splines(),
            la.V11BagLogit(),
            la.V11TeamEffect(),
        ],
    )
