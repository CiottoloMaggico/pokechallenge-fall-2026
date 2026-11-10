"""v11's features of the two active Pokémon at the last decision point: HP, status, boosts,
volatile effects, speed order and Gen 2 damage and turns-to-KO estimates in both directions.
A copy of the original's ``features/active.py``."""

from __future__ import annotations

import math
from typing import Any

from domain import expected_damage as ed
from domain.pokedex import PokemonGen2
from domain.timeline import SideTimeline as Side
from features.core.v11.battle import NAN, STATUSES, VOLATILE, Battle, log_ratio
from features.core.v11.moves import BOOSTS, CHART, PARTIAL_TRAP

BASE = ("hp", "atk", "def", "spa", "spd", "spe", "bst")
TTK_CAP = 20.0


def _ttk(hp: float, dmg: float) -> float:
    return min(TTK_CAP, hp / dmg) if dmg > 0 else TTK_CAP


def _base(d: PokemonGen2, k: str) -> float:
    return d.base_stat_total if k == "bst" else d.base(k)


def _stint(side: Side, i: int) -> int:
    k = i
    while k > 0 and side.names[k - 1] == side.names[i]:
        k -= 1
    return i - k + 1


def active_features(b: Battle) -> dict[str, Any]:
    """The ``a_`` columns of a battle."""
    i = b.n - 1
    f: dict[str, Any] = {}
    s1, s2 = b.sides["p1"], b.sides["p2"]
    n1, n2 = s1.names[i], s2.names[i]
    d1, d2 = b.species(n1), b.species(n2)
    for tag, side, d in (("a1", s1, d1), ("a2", s2, d2)):
        st, bo, ef = side.status[i], side.boosts[i], side.effects[i]
        f[f"a_{tag}_hp"] = side.hp[i]
        f[f"a_{tag}_fnt"] = st == "fnt"
        for s in STATUSES:
            f[f"a_{tag}_{s}"] = st == s
        for k in BOOSTS:
            f[f"a_{tag}_b_{k}"] = bo[k]
        f[f"a_{tag}_boost_pos"] = sum(max(0, v) for v in bo.values())
        f[f"a_{tag}_boost_neg"] = sum(min(0, v) for v in bo.values())
        for e in VOLATILE:
            f[f"a_{tag}_e_{e}"] = e in ef
        f[f"a_{tag}_e_partial"] = any(e in PARTIAL_TRAP for e in ef)
        for k in BASE:
            f[f"a_{tag}_base_{k}"] = _base(d, k) if d else NAN
        f[f"a_{tag}_stint"] = _stint(side, i)
        f[f"a_{tag}_n_moves"] = len(side.revealed.get(side.names[i], {}))

    lv1, lv2 = b.level("p1", n1), b.level("p2", n2)
    sp1 = ed.effective_speed(d1, s1.boosts[i], s1.status[i], lv1)
    sp2 = ed.effective_speed(d2, s2.boosts[i], s2.status[i], lv2)
    f["a_speed_logratio"] = log_ratio(sp1, sp2)
    faster = NAN if math.isnan(sp1) or math.isnan(sp2) else float(sp1 > sp2) - float(sp1 < sp2)
    f["a_p1_faster"] = faster

    m1 = list(s1.revealed.get(n1, {}).values())
    m2 = list(s2.revealed.get(n2, {}).values())
    dmg12 = ed.best_damage(
        CHART,
        m1,
        d1,
        d2,
        att_level=lv1,
        def_level=lv2,
        att_boosts=s1.boosts[i],
        def_boosts=s2.boosts[i],
        att_item=b.item("p1", n1),
        att_status=s1.status[i],
        att_name=n1,
    )
    dmg21 = ed.best_damage(
        CHART,
        m2,
        d2,
        d1,
        att_level=lv2,
        def_level=lv1,
        att_boosts=s2.boosts[i],
        def_boosts=s1.boosts[i],
        att_status=s2.status[i],
        att_name=n2,
    )
    f["a_dmg12"], f["a_dmg21"] = dmg12, dmg21
    ttk12, ttk21 = _ttk(s2.hp[i], dmg12), _ttk(s1.hp[i], dmg21)
    f["a_ttk12"], f["a_ttk21"] = ttk12, ttk21
    f["a_ttk_adv"] = ttk21 - ttk12 + (0 if math.isnan(faster) else 0.5 * faster)
    if d1 and d2:
        f["a_stab12"] = ed.stab_effectiveness(CHART, d1.types, d2.types)
        f["a_stab21"] = ed.stab_effectiveness(CHART, d2.types, d1.types)
    else:
        f["a_stab12"] = f["a_stab21"] = NAN
    return f
