"""The columns of :mod:`domain.transitions` the knockout detector is trained on.

``DETECTOR_FEATURES``: the 53 compressed to 30 by Gen 2 rules, without pruning
on scores (lossless codes and quantities that contain the dropped columns,
exp/ko_domain.py). Every dropped column is an exact function of the kept ones
or is folded into a quantity that contains it; out of fold (3 seeds) the log
loss is 0.0988 vs 0.0984 with the 50 (corrected t 0.39, p 0.70) and the P2
repairs agree on 99.2%. On P2 it repairs 10,130 switches, the number symmetry
with P1's recorded faints predicts (~10,100 detectable).
"""

from __future__ import annotations

DETECTOR_FEATURES = (
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
CATEGORICAL_FEATURES = frozenset({"x_status", "y_code"})
