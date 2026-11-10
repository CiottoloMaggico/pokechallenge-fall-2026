"""v11's move classes, type and stat orders, and the type chart its estimates use.

The classes are the original's (``mechanics.py``): what a feature counts as healing, set-up,
sleep, paralysis, phazing, self-KO, screens, weather, fixed damage or partial trapping. The
damage, speed and effectiveness estimates themselves are :mod:`domain.expected_damage`, over
``CHART``.
"""

from __future__ import annotations

from features.common import type_chart

CHART = type_chart()

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
