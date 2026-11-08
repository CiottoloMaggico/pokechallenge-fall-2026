"""Repairs by name, as ``--repair`` selects them: ``none``, ``rule``, ``model=<artifact folder>``.

An addon implements ``domain.repair.TimelineRepair``; the domain never imports one, the
application picks it here and hands it to the replay.
"""

from __future__ import annotations

from pathlib import Path

from addons.knockouts_model import ModelKnockoutRepair
from addons.knockouts_rule import RuleKnockoutRepair
from domain.damage import DamageCalculator
from domain.pokedex import Pokedex
from domain.repair import NO_REPAIR, TimelineRepair
from domain.typechart import TypeChart


def check_spec(spec: str) -> str:
    """``spec`` unchanged if it names a repair; ValueError otherwise (for the CLI)."""
    name, separator, argument = spec.partition("=")
    if (name in ("none", "rule") and not separator) or (name == "model" and argument):
        return spec
    raise ValueError(f"unknown repair {spec!r}: none, rule or model=<artifact folder>")


def load_repair(spec: str, pokedex: Pokedex, chart: TypeChart) -> TimelineRepair:
    """The repair named by ``spec`` (see :func:`check_spec`)."""
    name, _, argument = check_spec(spec).partition("=")
    damage = DamageCalculator(chart)
    if name == "none":
        return NO_REPAIR
    if name == "rule":
        return RuleKnockoutRepair(pokedex, damage)
    return ModelKnockoutRepair.from_artifact(Path(argument), pokedex, damage)
