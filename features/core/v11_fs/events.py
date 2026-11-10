"""v11's action statistics of each player (move classes, attack effectiveness, switches, damage
dealt, healing, hazards, screens, weather) and their difference. A copy of the original's
``features/events.py``."""

from __future__ import annotations

from typing import Any

from domain import expected_damage as ed
from features.core.v11_fs import moves as mx
from features.core.v11_fs.battle import OPP, STATUSES, Battle
from features.core.v11_fs.moves import CHART

CLASSES: dict[str, Any] = {
    "heal": mx.HEAL,
    "setup": mx.SETUP,
    "sleep": mx.SLEEP,
    "para": mx.PARA,
    "phaze": mx.PHAZE,
    "selfko": mx.SELF_KO,
    "screens": mx.SCREENS,
    "weather": mx.WEATHER,
    "spikes": {"spikes"},
    "spin": {"rapidspin"},
    "toxic": {"toxic"},
    "thief": {"thief"},
    "sleeptalk": {"sleeptalk"},
    "sub": {"substitute"},
    "protect": {"protect", "detect", "endure"},
    "batonpass": {"batonpass"},
    "trap": {"meanlook", "spiderweb", "perishsong"},
    "encore": {"encore"},
    "seed": {"leechseed"},
    "healbell": {"healbell"},
    "dbond": {"destinybond"},
}
SCREEN_TURNS = 5


def _moves(b: Battle, s: str) -> dict[str, Any]:
    """Move classes (``cls_*``) and attack statistics of the fresh move events of side s."""
    own, opp = b.sides[s], b.sides[OPP[s]]
    f: dict[str, Any] = {k: 0.0 for k in CLASSES}
    n_events = n_damaging = n_status = 0
    bp_sum = eff_sum = 0.0
    se = resisted = immune = 0
    for i, m in enumerate(own.events):
        if m is None:
            continue
        n_events += 1
        for k, names in CLASSES.items():
            if m["name"] in names:
                f[k] += 1
        if m["category"] == "status":
            n_status += 1
            continue
        n_damaging += 1
        bp_sum += m["base_power"]
        dd = b.species(opp.names[i])
        if dd is not None:
            e = ed.effectiveness(CHART, m["type"], dd.types)
            eff_sum += e
            se += e > 1
            resisted += 0 < e < 1
            immune += e == 0
    f = {f"cls_{k}": v for k, v in f.items()}
    f.update(
        {
            "events": n_events,
            "damaging": n_damaging,
            "status_moves": n_status,
            "bp_mean": bp_sum / n_damaging if n_damaging else 0.0,
            "eff_mean": eff_sum / n_damaging if n_damaging else 1.0,
            "se": se,
            "resisted": resisted,
            "immune": immune,
        }
    )
    return f


def _side(b: Battle, s: str) -> dict[str, Any]:
    own, opp = b.sides[s], b.sides[OPP[s]]
    f = _moves(b, s)
    switches = forced = dealt = healed = inflicted = 0
    for i in range(1, own.n):
        if own.names[i] != own.names[i - 1]:
            if own.status[i - 1] == "fnt":
                forced += 1
            else:
                switches += 1
        elif own.hp[i] > own.hp[i - 1]:
            healed += own.hp[i] - own.hp[i - 1]
        if opp.names[i] == opp.names[i - 1]:
            if opp.hp[i] < opp.hp[i - 1]:
                dealt += opp.hp[i - 1] - opp.hp[i]
            if opp.status[i - 1] == "nostatus" and opp.status[i] in STATUSES:
                inflicted += 1
    f.update(
        {
            "switches": switches,
            "forced": forced,
            "dealt": dealt,
            "healed": healed,
            "inflicted": inflicted,
            "distinct_moves": len({m for mv in own.revealed.values() for m in mv}),
            "seen": len(own.revealed),
            "turns_disabled": sum(st in ("slp", "frz") for st in own.status),
            "turns_par": sum(st == "par" for st in own.status),
            "spikes_on_opp": float(b.spikes[OPP[s]][-1]),
            "spikes_on_opp_turns": sum(b.spikes[OPP[s]]),
        }
    )
    last_screen = max(
        (i for i, m in enumerate(own.events) if m is not None and m["name"] in mx.SCREENS),
        default=-99,
    )
    f["screen_up"] = float(own.n - 1 - last_screen < SCREEN_TURNS)
    return f


def event_features(b: Battle) -> dict[str, Any]:
    """The ``e_`` columns of a battle."""
    p1, p2 = _side(b, "p1"), _side(b, "p2")
    f: dict[str, Any] = {}
    for k in p1:
        f[f"e_p1_{k}"] = p1[k]
        f[f"e_p2_{k}"] = p2[k]
        f[f"e_d_{k}"] = p1[k] - p2[k]
    weather = [
        (i, m["name"])
        for s in ("p1", "p2")
        for i, m in enumerate(b.sides[s].events)
        if m is not None and m["name"] in mx.WEATHER
    ]
    last = max(weather, default=(-99, None))
    for w in mx.WEATHER:
        f[f"e_weather_{w}"] = float(last[1] == w and b.n - 1 - last[0] < SCREEN_TURNS)
    return f
