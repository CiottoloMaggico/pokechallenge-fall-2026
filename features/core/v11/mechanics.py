"""v11's Gen 2 knowledge: type chart, Pokédex, stat formula, boost stages, damage estimate and
move classes.

A copy of the original's ``data.py`` (type chart, ``Pokedex``) and ``mechanics.py``. The chart
keeps the original's semantics (product over the defender's types, ``notype`` skipped, a missing
pair counts 1.0), unlike ``domain.typechart.TypeChart``; the Pokédex keeps the first record of
each species and ``for_battle`` adds the battle's own species, unlike ``domain.pokedex``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from functools import cache
from typing import Any

import pandas as pd

from features.common import TYPE_CHART_PATH

Record = Mapping[str, Any]
Entry = dict[str, Any]
Move = Mapping[str, Any]

STATS = ("hp", "atk", "def", "spa", "spd", "spe")
ALL_TYPES = (
    "normal",
    "fire",
    "water",
    "electric",
    "grass",
    "ice",
    "fighting",
    "poison",
    "ground",
    "flying",
    "psychic",
    "bug",
    "rock",
    "ghost",
    "dragon",
    "dark",
    "steel",
)
BOOSTS = ("atk", "def", "spa", "spd", "spe")

HEAL = {"rest", "recover", "softboiled", "milkdrink", "moonlight", "morningsun", "synthesis"}
SETUP = {
    "curse",
    "bellydrum",
    "swordsdance",
    "growth",
    "agility",
    "meditate",
    "sharpen",
    "amnesia",
    "acidarmor",
    "barrier",
    "defensecurl",
    "harden",
    "withdraw",
    "focusenergy",
    "doubleteam",
    "minimize",
}
SLEEP = {"lovelykiss", "spore", "sleeppowder", "hypnosis", "sing"}
PARA = {"thunderwave", "stunspore", "glare"}
PHAZE = {"roar", "whirlwind"}
SELF_KO = {"explosion", "selfdestruct"}
SCREENS = {"reflect", "lightscreen"}
WEATHER = ("raindance", "sunnyday", "sandstorm")
FIXED_DAMAGE = {"seismictoss", "nightshade"}
PARTIAL_TRAP = {"clamp", "wrap", "bind", "firespin", "whirlpool"}


@cache
def type_chart() -> dict[tuple[str, str], float]:
    """The supplied Gen 2 chart as ``{(attacking type, defending type): multiplier}``."""
    df = pd.read_csv(TYPE_CHART_PATH)
    return {(a, d): m for a, d, m in df.itertuples(index=False)}


def effectiveness(move_type: str, def_types: Iterable[str]) -> float:
    """Product of the multipliers over the defender's types (``notype`` skipped)."""
    chart = type_chart()
    mult = 1.0
    for t in def_types:
        if t != "notype":
            mult *= chart.get((move_type, t), 1.0)
    return mult


class Pokedex:
    """Base stats, types and base stat total per species, from the first record that shows it."""

    def __init__(self, battles: Iterable[Record]) -> None:
        self.entries: dict[str, Entry] = {}
        for b in battles:
            for p in b["p1_team_details"] + [b["p2_lead_details"]]:
                self.entries.setdefault(p["name"], self._entry(p))

    @staticmethod
    def _entry(p: Record) -> Entry:
        e: Entry = {s: p[f"base_{s}"] for s in STATS}
        e["types"] = tuple(t for t in p["types"] if t != "notype")
        e["bst"] = sum(e[s] for s in STATS)
        return e

    def for_battle(self, battle: Record) -> dict[str, Entry]:
        """The entries plus the battle's own species that the Pokédex lacks."""
        local = dict(self.entries)
        for p in battle["p1_team_details"] + [battle["p2_lead_details"]]:
            if p["name"] not in local:
                local[p["name"]] = self._entry(p)
        return local


def stat_value(base: int, level: int = 100, hp: bool = False) -> int:
    """Gen 2 stat with maximal DVs and stat experience."""
    core = ((base + 15) * 2 + 63) * level // 100
    return core + level + 10 if hp else core + 5


def stage_mult(stage: int) -> float:
    """Multiplier of a boost stage."""
    return (2 + stage) / 2 if stage >= 0 else 2 / (2 - stage)


def damage_fraction(
    move: Move | None,
    att: Entry | None,
    dfn: Entry | None,
    att_level: int = 100,
    def_level: int = 100,
    att_boosts: Mapping[str, int] | None = None,
    def_boosts: Mapping[str, int] | None = None,
    att_item: str | None = None,
    att_status: str | None = None,
    att_name: str | None = None,
) -> float:
    """Expected share of the defender's maximal HP taken by ``move`` (accuracy included)."""
    if move is None or move["category"] == "status" or att is None or dfn is None:
        return 0.0
    def_hp = stat_value(dfn["hp"], def_level, hp=True)
    eff = effectiveness(move["type"], dfn["types"])
    if move["name"] in FIXED_DAMAGE:
        return 0.0 if eff == 0 else att_level / def_hp
    bp = move["base_power"]
    if bp <= 1:
        return 0.0
    physical = move["category"] == "physical"
    a_key, d_key = ("atk", "def") if physical else ("spa", "spd")
    a = stat_value(att[a_key], att_level) * stage_mult((att_boosts or {}).get(a_key, 0))
    d = stat_value(dfn[d_key], def_level) * stage_mult((def_boosts or {}).get(d_key, 0))
    if physical and att_status == "brn":
        a *= 0.5
    if att_item == "thickclub" and att_name in ("marowak", "cubone") and physical:
        a *= 2
    if att_item == "lightball" and att_name == "pikachu" and not physical:
        a *= 2
    if move["name"] in SELF_KO:
        d *= 0.5
    base = (2 * att_level / 5 + 2) * bp * a / d / 50 + 2
    stab = 1.5 if move["type"] in att["types"] else 1.0
    return base * stab * eff * 0.925 * move["accuracy"] / def_hp


def best_damage(
    moves: Iterable[Move | None], att: Entry | None, dfn: Entry | None, **kw: Any
) -> float:
    """The largest ``damage_fraction`` over ``moves`` (0 without moves)."""
    best = 0.0
    for m in moves:
        best = max(best, damage_fraction(m, att, dfn, **kw))
    return best


def stab_effectiveness(att_types: Sequence[str], def_types: Sequence[str]) -> float:
    """Best multiplier of the attacker's own types (1.0 without types)."""
    return max((effectiveness(t, def_types) for t in att_types), default=1.0)


def effective_speed(
    entry: Entry | None, boosts: Mapping[str, int], status: str | None, level: int = 100
) -> float:
    """Speed after boosts and paralysis (NaN for an unknown species)."""
    if entry is None:
        return float("nan")
    v = stat_value(entry["spe"], level) * stage_mult(boosts.get("spe", 0))
    return v * 0.25 if status == "par" else v
