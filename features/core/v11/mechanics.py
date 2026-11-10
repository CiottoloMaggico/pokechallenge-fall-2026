"""v11's Pokédex: base stats and types per species, from the first record that shows it.

The species are ``domain.pokedex.PokemonGen2`` (placeholder types dropped, integer stats); the
damage, speed and effectiveness estimates are :mod:`domain.expected_damage` and the move classes
are in ``features.core.v11.moves``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from domain.pokedex import PokemonGen2

Record = Mapping[str, Any]


class Pokedex:
    """Species from the first record that shows each; ``for_battle`` adds the battle's own."""

    def __init__(self, battles: Iterable[Record]) -> None:
        self.entries: dict[str, PokemonGen2] = {}
        for b in battles:
            for p in b["p1_team_details"] + [b["p2_lead_details"]]:
                self.entries.setdefault(p["name"], PokemonGen2.from_dict(p))

    def for_battle(self, battle: Record) -> dict[str, PokemonGen2]:
        """The entries plus the battle's own species that the Pokédex lacks."""
        local = dict(self.entries)
        for p in battle["p1_team_details"] + [battle["p2_lead_details"]]:
            if p["name"] not in local:
                local[p["name"]] = PokemonGen2.from_dict(p)
        return local
