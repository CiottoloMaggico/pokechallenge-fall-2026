"""v11's battle state at the last decision point and its trajectory: faints (with the estimated
hidden Player 2 faints), HP and status, and their differences at earlier snapshots.

A copy of the original's ``features/state.py``. A track is the per-Pokémon HP, status and faint
probability at the last entry plus the per-entry cumulative faints and total HP.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from domain.battle import BattleContext
from features.core.v11.battle import STATUSES, Battle, battle_of

SNAPSHOTS = (26, 21, 16, 11, 6)

Track = dict[str, Any]


def p1_track(b: Battle) -> Track:
    """Player 1's track: faints are always recorded."""
    side = b.sides["p1"]
    hp = {p["name"]: 1.0 for p in b.team}
    status = {p["name"]: "nostatus" for p in b.team}
    fainted: set[str] = set()
    faint_cum, hp_tot = np.zeros(b.n), np.zeros(b.n)
    for i in range(b.n):
        n = side.names[i]
        hp[n], status[n] = side.hp[i], side.status[i]
        if side.status[i] == "fnt":
            fainted.add(n)
        faint_cum[i] = len(fainted)
        hp_tot[i] = sum(v for k, v in hp.items() if k not in fainted)
    pf = {n: float(n in fainted) for n in hp}
    return {"hp": hp, "status": status, "pf": pf, "faint_cum": faint_cum, "hp_tot": hp_tot}


def p2_track(b: Battle) -> Track:
    """Player 2's track, with the detector's faint probabilities (``b.p2_faint``).

    The lead is known from the start, also when it switches out on turn 1 and never shows up again.
    """
    side = b.sides["p2"]
    lead = b.lead2["name"]
    hp, status, pf = {lead: 1.0}, {lead: "nostatus"}, {lead: 0.0}
    faint_cum, hp_tot, revealed = np.zeros(b.n), np.zeros(b.n), np.zeros(b.n)
    for i in range(b.n):
        if i > 0 and i in b.p2_faint:
            x = side.names[i - 1]
            pf[x] = max(pf.get(x, 0.0), b.p2_faint[i])
        n = side.names[i]
        hp[n], status[n] = side.hp[i], side.status[i]
        pf.setdefault(n, 0.0)
        if side.status[i] == "fnt":
            pf[n] = 1.0
        faint_cum[i] = sum(pf.values())
        hp_tot[i] = sum((1 - pf[k]) * v for k, v in hp.items()) + (6 - len(hp))
        revealed[i] = len(hp)
    return {
        "hp": hp,
        "status": status,
        "pf": pf,
        "faint_cum": faint_cum,
        "hp_tot": hp_tot,
        "revealed": revealed,
    }


def tracks_of(context: BattleContext) -> tuple[Track, Track]:
    """P1's and P2's tracks of ``context``, built once per replay (they only read the view)."""
    b = battle_of(context)
    return context.cached(("v11", "tracks"), lambda: (p1_track(b), p2_track(b)))


def _side_summary(prefix: str, tr: Track, b: Battle, side_key: str) -> dict[str, Any]:
    side = b.sides[side_key]
    hp, status, pf = tr["hp"], tr["status"], tr["pf"]
    dex = {n: d for n in hp if (d := b.species(n)) is not None}
    alive_w = {n: 1 - pf[n] for n in hp}
    f: dict[str, Any] = {
        f"{prefix}_fainted": tr["faint_cum"][-1],
        f"{prefix}_hp_sum": tr["hp_tot"][-1],
        f"{prefix}_hp_low": sum(w for n, w in alive_w.items() if hp[n] < 0.3),
        f"{prefix}_hp_mid": sum(w for n, w in alive_w.items() if hp[n] < 0.6),
        f"{prefix}_strength": sum(
            w * hp[n] * dex[n].base_stat_total / 100 for n, w in alive_w.items() if n in dex
        ),
        f"{prefix}_speed_alive": max(
            (dex[n].base_spe for n, w in alive_w.items() if w > 0.5 and n in dex), default=0
        ),
    }
    for st in STATUSES:
        f[f"{prefix}_n_{st}"] = sum(w for n, w in alive_w.items() if status[n] == st)
    f[f"{prefix}_disabled"] = f[f"{prefix}_n_slp"] + f[f"{prefix}_n_frz"]
    f[f"{prefix}_sleep_talkers"] = sum(
        w
        for n, w in alive_w.items()
        if status[n] == "slp" and "sleeptalk" in side.revealed.get(n, {})
    )
    f[f"{prefix}_resters"] = sum(
        w for n, w in alive_w.items() if "rest" in side.revealed.get(n, {})
    )
    return f


def state_features(b: Battle, t1: Track, t2: Track) -> dict[str, Any]:
    """The ``f_`` (state) and ``t_`` (trajectory) columns of a battle."""
    f: dict[str, Any] = {}
    f.update({"f_" + k: v for k, v in _side_summary("p1", t1, b, "p1").items()})
    f.update({"f_" + k: v for k, v in _side_summary("p2", t2, b, "p2").items()})
    s1 = b.sides["p1"]
    f["f_p1_unseen"] = sum(1 for p in b.team if p["name"] not in s1.first_index)
    f["f_p2_revealed"] = len(t2["hp"])
    f["f_p2_unrevealed"] = 6 - len(t2["hp"])
    f["f_p2_fainted_rec"] = sum(1 for n in t2["hp"] if t2["status"][n] == "fnt")
    f["f_p2_fainted_hard"] = sum(1 for n in t2["hp"] if t2["pf"][n] >= 0.5)
    f["f_p2_hp_sum_opt"] = (
        sum(v for n, v in t2["hp"].items() if t2["status"][n] != "fnt") + 6 - len(t2["hp"])
    )
    f["f_p2_hp_sum_hard"] = (
        sum(v for n, v in t2["hp"].items() if t2["pf"][n] < 0.5) + 6 - len(t2["hp"])
    )
    f["f_p2_pf_uncertain"] = sum(min(p, 1 - p) for p in t2["pf"].values())
    for k in (
        "fainted",
        "hp_sum",
        "hp_low",
        "strength",
        "disabled",
        "n_par",
        "n_tox",
        "n_psn",
        "n_frz",
        "n_slp",
    ):
        f[f"f_d_{k}"] = f[f"f_p1_{k}"] - f[f"f_p2_{k}"]
    f["f_d_alive"] = (6 - f["f_p1_fainted"]) - (6 - f["f_p2_fainted"])
    f["f_d_fainted_hard"] = f["f_p2_fainted_hard"] - f["f_p1_fainted"]
    f["f_d_hp_hard"] = f["f_p1_hp_sum"] - f["f_p2_hp_sum_hard"]
    f["f_p1_forced_entries"] = f["f_p1_fainted"]
    f.update(_trajectory(b, t1, t2))
    return f


def _trajectory(b: Battle, t1: Track, t2: Track) -> dict[str, Any]:
    f: dict[str, Any] = {}
    fd = t2["faint_cum"] - t1["faint_cum"]
    hd = t1["hp_tot"] - t2["hp_tot"]
    for back in SNAPSHOTS:
        k = max(0, b.n - back)
        f[f"t_fd_{30 - back + 1}"] = fd[k]
        f[f"t_hd_{30 - back + 1}"] = hd[k]
    f["t_fd_mean"], f["t_hd_mean"] = fd.mean(), hd.mean()
    f["t_hd_min"], f["t_hd_max"] = hd.min(), hd.max()
    f["t_fd_min"], f["t_fd_max"] = fd.min(), fd.max()
    f["t_hd_delta5"], f["t_hd_delta10"] = hd[-1] - hd[-6], hd[-1] - hd[-11]
    f["t_fd_delta5"], f["t_fd_delta10"] = fd[-1] - fd[-6], fd[-1] - fd[-11]
    x = np.arange(10)
    f["t_hd_slope10"] = np.polyfit(x, hd[-10:], 1)[0]
    f["t_p1_first_faint"] = (
        int(np.argmax(t1["faint_cum"] >= 1)) if t1["faint_cum"][-1] >= 1 else b.n
    )
    f["t_p2_first_faint"] = (
        int(np.argmax(t2["faint_cum"] >= 0.5)) if t2["faint_cum"][-1] >= 0.5 else b.n
    )
    f["t_p2_reveal_10"] = t2["revealed"][max(0, b.n - 21)]
    f["t_p2_reveal_20"] = t2["revealed"][max(0, b.n - 11)]
    return f
