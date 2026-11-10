"""Repairs by name, as ``--repair`` selects them: ``none``, ``rule``, ``model=<artifact folder>``,
``v11=<artifact folder>``.

An addon implements ``domain.repair.TimelineRepair``; the domain never imports one, the
application picks it here and hands it to the replay.
"""

from __future__ import annotations

from pathlib import Path

from addons.knockouts_model import ModelKnockoutRepair
from addons.knockouts_rule import RuleKnockoutRepair
from addons.knockouts_v11 import V11KnockoutRepair
from domain.damage import DamageCalculator
from domain.pokedex import Pokedex
from domain.repair import NO_REPAIR, TimelineRepair
from domain.typechart import TypeChart

ARTIFACT_REPAIRS = ("model", "v11")


def check_spec(spec: str) -> str:
    """``spec`` unchanged if it names a repair; ValueError otherwise (for the CLI)."""
    name, separator, argument = spec.partition("=")
    if (name in ("none", "rule") and not separator) or (name in ARTIFACT_REPAIRS and argument):
        return spec
    raise ValueError(
        f"unknown repair {spec!r}: none, rule, model=<artifact folder> or v11=<artifact folder>"
    )


def load_repair(spec: str, pokedex: Pokedex, chart: TypeChart) -> TimelineRepair:
    """The repair named by ``spec`` (see :func:`check_spec`)."""
    name, _, argument = check_spec(spec).partition("=")
    damage = DamageCalculator(chart)
    if name == "none":
        return NO_REPAIR
    if name == "rule":
        return RuleKnockoutRepair(pokedex, damage)
    if name == "v11":
        return V11KnockoutRepair.from_artifact(Path(argument), pokedex, chart)
    return ModelKnockoutRepair.from_artifact(Path(argument), pokedex, damage)
