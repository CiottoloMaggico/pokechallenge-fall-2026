"""``ko_domain30``: the 53 columns of ``domain.transitions`` compressed to 30 by Gen 2 rules.

No pruning on scores: lossless codes and quantities that contain the dropped columns. Every
dropped column is an exact function of the kept ones or is folded into a quantity that contains
it; out of fold (3 seeds) the log loss is 0.0988 vs 0.0984 with ``ko_raw50`` (corrected t 0.39,
p 0.70) and the P2 repairs agree on 99.2%. On P2 it repaired 10,130 switches before the Pokédex
lost its manual species (10,109 after), the number symmetry with P1's recorded faints predicts
(~10,100 detectable). The codes ``x_status`` and ``y_code`` are categorical: the ``hgb_ko``
model treats them so when a table has them.
"""

from __future__ import annotations

from domain.transitions import Transition
from feature_set import FeatureSet
from features.transitions.columns import transition_set

COLUMNS = (
    "turn",
    # leaver: HP, then margins that already include the Gen 2 damage
    "x_hp",
    # end-of-turn damage: psn / tox counter, Leech Seed, Curse, Nightmare, traps, sand, Leftovers
    "x_residual",
    "x_hp_after",  # x_hp - field-aware expected damage - x_residual (a_dmg_field is implied)
    "x_hp_minus_max",  # x_hp - worst critical roll (a_dmg_max is implied)
    "x_status",  # categorical code of slp / par / frz / psn / tox / brn
    "x_boost",
    "x_trapped",
    "x_sub",
    "x_conf",
    "x_perish_turns",  # 0 when not under Perish Song (replaces x_perish)
    "x_bhp",
    "x_bdef",
    "x_bspd",
    "x_spe_ratio",
    # incoming Pokémon
    "y_code",  # categorical: first appearance x (no move / new move / repeated move)
    "y_hp",
    "y_drop",  # y_prev_hp = y_hp + y_drop
    "drop_gap",  # a_dmg_in = drop_gap + y_drop
    "y_bst",
    "spikes",
    # opponent
    "a_drop",  # NaN when the attacker changed (replaces a_same)
    "a_drop_any",
    "a_mv_changed",
    "a_bp",
    "a_status_mv",
    "a_eff",
    "a_prio_phaze",  # priority, -2 for Roar / Whirlwind (replaces a_priority, a_phaze)
    "a_boost",
    "a_disabled",
)


def build() -> FeatureSet[Transition]:
    """The 30 domain-compressed columns."""
    return transition_set("ko_domain30", COLUMNS)
