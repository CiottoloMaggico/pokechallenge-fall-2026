"""Repairs of the data artifacts of a timeline (today: hidden P2 faints).

A repair may be deterministic code or a model loaded from disk: the replay
only knows this protocol, and the implementation is passed in. A repair never
sees the outcome: :class:`domain.battle.BattleReplay` removes ``player_won``
from a record before handing it over.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol


class TimelineRepair(Protocol):
    """Finds the P2 faints a record omits; a deterministic repair gives 0 / 1."""

    def probabilities(self, record: Mapping[str, Any]) -> Mapping[int, float]:
        """Turn index -> probability that the P2 Pokémon that left at that row fainted."""
        ...

    def repairs(self, record: Mapping[str, Any]) -> Mapping[int, str]:
        """Turn index -> name of the Pokémon to mark as fainted."""
        ...

    def describe(self) -> dict[str, Any]:
        """JSON with ``name``, ``version`` and ``params``, to be saved with a run."""
        ...


class NoRepair:
    """The timeline as recorded, for tests and baselines."""

    def probabilities(self, record: Mapping[str, Any]) -> Mapping[int, float]:
        """Nothing is scored."""
        return {}

    def repairs(self, record: Mapping[str, Any]) -> Mapping[int, str]:
        """Nothing is repaired."""
        return {}

    def describe(self) -> dict[str, Any]:
        """Identify this repair in a run."""
        return {"name": "none", "version": "1", "params": {}}


NO_REPAIR = NoRepair()
