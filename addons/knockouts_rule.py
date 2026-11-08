"""The hidden P2 knockouts found by a hand-written rule, without any training.

A P2 switch whose leaver never comes back is a knockout when the expected damage of the
opposing move covers the leaver's HP (``x_hp_minus_dmg <= margin``) and the replacement entered
unhurt (``y_drop <= 0``): the rule of ``compact_v2``, before the learned detector. Probabilities
are 0 / 1; a column that cannot be computed (NaN) never marks a knockout.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from domain.damage import DamageCalculator
from domain.pokedex import Missing, Pokedex
from domain.records import fingerprint
from domain.transitions import TransitionExtractor

KEEP_BATTLE = Missing()


class RuleKnockoutRepair:
    """Deterministic ``TimelineRepair`` of P2's candidate switches."""

    def __init__(self, pokedex: Pokedex, damage: DamageCalculator, margin: float = 0.0) -> None:
        self._pokedex = pokedex
        self._damage = damage
        self._margin = margin
        self._cache: dict[str, dict[int, float]] = {}

    def probabilities(self, record: Mapping[str, Any]) -> dict[int, float]:
        """Turn index -> 1.0 when the rule marks the switch as a knockout, else 0.0."""
        key = fingerprint(record)
        cached = self._cache.get(key)
        if cached is None:
            cached = self._score(record)
            self._cache[key] = cached
        return dict(cached)

    def repairs(self, record: Mapping[str, Any]) -> dict[int, str]:
        """Turn index -> name of every switch the rule marks."""
        timeline = record["battle_timeline"]
        return {
            i: timeline[i - 1]["p2_pokemon_state"]["name"]
            for i, p in self.probabilities(record).items()
            if p == 1.0
        }

    def describe(self) -> dict[str, Any]:
        """Identify this repair in a run."""
        return {
            "name": "knockouts_rule",
            "version": "1",
            "params": {"margin": self._margin, "pokedex_sha256": self._pokedex.source_sha256},
        }

    def _score(self, record: Mapping[str, Any]) -> dict[int, float]:
        species = self._pokedex.lookup(record, KEEP_BATTLE)
        return {
            t.index: float(
                t.features["x_hp_minus_dmg"] <= self._margin and t.features["y_drop"] <= 0
            )
            for t in TransitionExtractor(species, self._damage).candidates(record)
        }
