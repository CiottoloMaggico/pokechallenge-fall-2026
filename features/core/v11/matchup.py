"""v11's matchup matrix between the remaining Pokémon of both players and win-condition features
(hazards, spinners, setup, phazers, healers, sleepers, explosions). A copy of the original's
``features/matchup.py``."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
from scipy.special import expit

from domain import expected_damage as ed
from features.core.v11 import moves as mx
from features.core.v11.battle import Battle
from features.core.v11.moves import CHART
from features.core.v11.state import Track

PHYSICAL_TYPES = {
    "normal",
    "fighting",
    "flying",
    "poison",
    "ground",
    "rock",
    "bug",
    "ghost",
    "steel",
}
DISABLE = {"slp": 0.5, "frz": 0.1}
STEEP = 1.5

Fighter = dict[str, Any]


def proxies(types: Iterable[str]) -> list[dict[str, Any]]:
    """One 80 base power STAB move per type (physical or special by the Gen 2 type split)."""
    return [
        {
            "name": f"stab_{t}",
            "type": t,
            "category": "physical" if t in PHYSICAL_TYPES else "special",
            "base_power": 80,
            "accuracy": 1.0,
        }
        for t in types
    ]


def _roster(
    b: Battle, side_key: str, track: Track, weight_fn: Callable[[str], float]
) -> list[Fighter]:
    side = b.sides[side_key]
    out: list[Fighter] = []
    for n, hp in track["hp"].items():
        w = weight_fn(n)
        d = b.species(n)
        if w <= 0.5 or d is None or hp <= 0:
            continue
        revealed = [m for m in side.revealed.get(n, {}).values() if m["category"] != "status"]
        st = track["status"][n]
        mult = 1.0
        if st in DISABLE and not (st == "slp" and "sleeptalk" in side.revealed.get(n, {})):
            mult = DISABLE[st]
        spe = ed.stat_value(d.base_spe, b.level(side_key, n)) * (0.25 if st == "par" else 1.0)
        out.append(
            {
                "name": n,
                "hp": hp,
                "w": w,
                "d": d,
                "moves": revealed + proxies(d.types),
                "mult": mult,
                "spe": spe,
                "level": b.level(side_key, n),
                "item": b.item(side_key, n),
                "status": st,
            }
        )
    return out


def _dmg(u: Fighter, v: Fighter) -> float:
    return u["mult"] * ed.best_damage(
        CHART,
        u["moves"],
        u["d"],
        v["d"],
        att_level=u["level"],
        def_level=v["level"],
        att_item=u["item"],
        att_status=u["status"],
        att_name=u["name"],
    )


def matchup_features(b: Battle, t1: Track, t2: Track) -> dict[str, Any]:
    """The ``m_`` columns of a battle."""
    A = _roster(b, "p1", t1, lambda n: 1 - t1["pf"][n])
    B = _roster(b, "p2", t2, lambda n: 1 - t2["pf"][n])
    f: dict[str, Any] = {"m_n1": len(A), "m_n2": len(B)}
    keys = (
        "m_win_mean",
        "m_p1_cover",
        "m_p2_cover",
        "m_p1_sweeper",
        "m_p2_sweeper",
        "m_p1_dmg_cov",
        "m_p2_dmg_cov",
        "m_p1_ohko",
        "m_p2_ohko",
        "m_p1_walls",
        "m_p2_walls",
    )
    if not A or not B:
        f.update({k: math.nan for k in keys})
        return f
    D12 = np.array([[_dmg(u, v) for v in B] for u in A])
    D21 = np.array([[_dmg(v, u) for v in B] for u in A])
    hp1 = np.array([u["hp"] for u in A])[:, None]
    hp2 = np.array([v["hp"] for v in B])[None, :]
    ttk12 = np.where(D12 > 0, hp2 / np.maximum(D12, 1e-9), 50.0)
    ttk21 = np.where(D21 > 0, hp1 / np.maximum(D21, 1e-9), 50.0)
    faster = np.sign(
        np.array([u["spe"] for u in A])[:, None] - np.array([v["spe"] for v in B])[None, :]
    )
    W = expit(STEEP * (np.ceil(ttk21) - np.ceil(ttk12) + 0.5 * faster))
    w2 = np.array([v["w"] for v in B])
    f["m_win_mean"] = float((W * w2).sum() / (w2.sum() * len(A)))
    f["m_p1_cover"] = float((W.max(0) * w2).sum() / w2.sum())
    f["m_p2_cover"] = float((1 - W).max(1).mean())
    f["m_p1_sweeper"] = float((W * w2).sum(1).max())
    f["m_p2_sweeper"] = float(((1 - W).sum(0) * w2).max())
    f["m_p1_dmg_cov"] = float((np.minimum(D12.max(0), 1) * w2).sum() / w2.sum())
    f["m_p2_dmg_cov"] = float(np.minimum(D21.max(1), 1).mean())
    f["m_p1_ohko"] = float(((D12.max(0) >= hp2[0]) * w2).sum())
    f["m_p2_ohko"] = float((D21.max(1) >= hp1[:, 0]).sum())
    f["m_p1_walls"] = float((D21.max(1) < 0.25).sum())
    f["m_p2_walls"] = float(((D12.max(0) < 0.25) * w2).sum())
    return f


def wincon_features(b: Battle, t1: Track, t2: Track) -> dict[str, Any]:
    """The ``w_`` columns of a battle."""
    f: dict[str, Any] = {}
    for key, track in (("p1", t1), ("p2", t2)):
        side = b.sides[key]
        alive = [n for n in track["hp"] if track["pf"][n] < 0.5]
        moves = {n: side.revealed.get(n, {}) for n in alive}
        f[f"w_{key}_spinners"] = sum("rapidspin" in m for m in moves.values())
        f[f"w_{key}_setup"] = sum(any(x in mx.SETUP for x in m) for m in moves.values())
        f[f"w_{key}_phazers"] = sum(any(x in mx.PHAZE for x in m) for m in moves.values())
        f[f"w_{key}_healers"] = sum(any(x in mx.HEAL for x in m) for m in moves.values())
        f[f"w_{key}_boomers"] = sum(any(x in mx.SELF_KO for x in m) for m in moves.values())
        f[f"w_{key}_sleepers"] = sum(any(x in mx.SLEEP for x in m) for m in moves.values())
        f[f"w_spikes_on_{key}"] = float(b.spikes[key][-1])
    f["w_spikes_adv"] = f["w_spikes_on_p2"] * (f["w_p2_spinners"] == 0) - f["w_spikes_on_p1"] * (
        f["w_p1_spinners"] == 0
    )
    i = b.n - 1
    for key in ("p1", "p2"):
        side = b.sides[key]
        trapped = "trapped" in side.effects[i] or any(e in mx.PARTIAL_TRAP for e in side.effects[i])
        f[f"w_{key}_trapped_perish"] = float(trapped and "perish" in side.effects[i])
        f[f"w_{key}_boosted_active"] = float(sum(max(0, v) for v in side.boosts[i].values()) >= 2)
    for k in ("spinners", "setup", "phazers", "healers", "boomers", "sleepers"):
        f[f"w_d_{k}"] = f[f"w_p1_{k}"] - f[f"w_p2_{k}"]
    return f
