"""``v11_fs``: the living line of v11, with the label-aware steps of its members as features.

A two-stage set (``staged_feature_set``). First stage: the groups of ``v11`` (from the package
``features.core.v11_fs``, a copy that owns its code; the frozen ``v11`` set is never modified)
in the exact order of v11's matrix, bags and team keys included (the base logit of the team
effect reads the columns in this order). Second stage, on that table: the splines of the spline
member's keys (``V11Splines``), the logit of the bags (``V11BagLogit``, ``stack_logit_0``, which
consumes the bags) and the team effect against a base logit of every non-key column
(``V11TeamEffect``, ``tm*``, which consumes the keys), the last two label-aware (out of fold on
their training rows). The generic models (``models/stacked.py``) read it with no column name.
Fixed at the last turn; ``train`` / ``evaluate`` refuse ``--test`` (the label-free features fit
on the training battles only).
"""

from __future__ import annotations

from domain.battle import BattleContext
from features.common import make_set
from registry import feature_set
from staged_feature_set import StagedFeatureSet


@feature_set("v11_fs", requires=("lightgbm",))
def v11_fs(turn: int | None = None) -> StagedFeatureSet[BattleContext]:
    """v11's columns, then splines, bag logit and team effect on their table."""
    if turn is not None:
        raise ValueError("v11_fs is fixed at the last turn: drop the @t suffix")
    # Imported here so that listing the feature sets does not need LightGBM.
    from features.core.v11_fs import groups as g
    from features.core.v11_fs import label_aware as la
    from features.core.v11_fs.forecast import V11Forecast

    base = make_set(
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
            g.V11Bags(),
            g.V11TeamKeys(),
            g.V11Duel(),
            g.V11Behaviour(),
            g.V11Simulation(),
            g.V11Condition(),
            V11Forecast(),
        ],
    )
    return StagedFeatureSet("v11_fs", base, [la.V11Splines(), la.V11BagLogit(), la.V11TeamEffect()])
