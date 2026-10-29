"""Gen 2 species data and stat formulas.

A :class:`PokemonGen2` is a *species*: name, types and base stats. It
holds nothing about an individual Pokémon; level, DVs and stat
experience belong to the individual (see ``battle.PokemonState``) and
are passed to the stat methods.

The :class:`Pokedex` is loaded from a JSON file listing every species::

    [
      {"name": "snorlax", "types": ["normal"],
       "base_hp": 160, "base_atk": 110, "base_def": 65,
       "base_spa": 65, "base_spd": 110, "base_spe": 30},
      ...
    ]

Example::

    pokedex = Pokedex.from_json("assets/species.json")
    snorlax = pokedex.get("snorlax")
    snorlax.max_hp(level=100)        # 523
    snorlax.stat("atk", level=100)   # 318
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

STATS = ("atk", "def", "spa", "spd", "spe")
TYPES = frozenset(
    {
        "normal", "fire", "water", "electric", "grass", "ice", "fighting",
        "poison", "ground", "flying", "psychic", "bug", "rock", "ghost",
        "dragon", "dark", "steel",
    }
)  # fmt: skip
PLACEHOLDER_TYPES = frozenset({"notype"})  # filler for single-type species

MAX_DV = 15
MAX_STAT_EXP = 65535
DEFAULT_LEVEL = 100


class PokedexError(ValueError):
    """The species file is malformed or a species is unknown."""


def stat_points(stat_exp: int = MAX_STAT_EXP) -> int:
    """Gen 2 stat-experience bonus: min(255, isqrt(exp - 1) + 1) // 4.

    0 stat experience gives 0; the maximum (65535) gives 63.
    """
    return min(255, math.isqrt(max(0, stat_exp - 1)) + 1) // 4


def calc_hp(
    base: int, level: int, dv: int = MAX_DV, stat_exp: int = MAX_STAT_EXP
) -> int:
    """Gen 2 HP formula."""
    core = ((base + dv) * 2 + stat_points(stat_exp)) * level // 100
    return core + level + 10


def calc_stat(
    base: int, level: int, dv: int = MAX_DV, stat_exp: int = MAX_STAT_EXP
) -> int:
    """Gen 2 formula for Attack, Defense, Sp. Atk, Sp. Def and Speed."""
    return ((base + dv) * 2 + stat_points(stat_exp)) * level // 100 + 5


@dataclass(frozen=True)
class PokemonGen2:
    """A Gen 2 species: name, real types and base stats."""

    name: str
    types: tuple[str, ...]
    base_hp: int
    base_atk: int
    base_def: int
    base_spa: int
    base_spd: int
    base_spe: int

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PokemonGen2:
        """Build a species from a record; placeholder types are dropped."""
        try:
            types = tuple(t for t in data["types"] if t not in PLACEHOLDER_TYPES)
            species = cls(
                name=str(data["name"]),
                types=types,
                **{f"base_{s}": int(data[f"base_{s}"]) for s in ("hp", *STATS)},
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise PokedexError(f"invalid species entry {dict(data)!r}") from exc
        species.validate()
        return species

    def validate(self) -> None:
        if not self.name:
            raise PokedexError("species without a name")
        if not 1 <= len(self.types) <= 2 or not set(self.types) <= TYPES:
            raise PokedexError(f"{self.name}: invalid types {self.types}")
        if any(not 1 <= self.base(s) <= 255 for s in ("hp", *STATS)):
            raise PokedexError(f"{self.name}: base stats must be in 1..255")

    def base(self, stat: str) -> int:
        """Base value of ``"hp"``, ``"atk"``, ``"def"``, ... ."""
        if stat not in ("hp", *STATS):
            raise KeyError(f"unknown stat {stat!r}")
        return getattr(self, f"base_{stat}")

    def max_hp(self, level: int, dv: int = MAX_DV, stat_exp: int = MAX_STAT_EXP) -> int:
        return calc_hp(self.base_hp, level, dv, stat_exp)

    def stat(
        self,
        stat: str,
        level: int,
        dv: int = MAX_DV,
        stat_exp: int = MAX_STAT_EXP,
    ) -> int:
        """Real value of a non-HP stat at ``level``."""
        if stat == "hp":
            return self.max_hp(level, dv, stat_exp)
        return calc_stat(self.base(stat), level, dv, stat_exp)

    @property
    def base_stat_total(self) -> int:
        return sum(self.base(s) for s in ("hp", *STATS))


class Pokedex:
    """Read-only collection of Gen 2 species, indexed by name."""

    def __init__(self, species: Iterable[PokemonGen2]) -> None:
        """Index ``species`` by name.

        Repeated identical entries are kept once; two different entries
        with the same name raise PokedexError.
        """
        self._species: dict[str, PokemonGen2] = {}
        for entry in species:
            known = self._species.setdefault(entry.name, entry)
            if known != entry:
                raise PokedexError(
                    f"species {entry.name!r} listed twice with different data: "
                    f"{known} and {entry}"
                )

    @classmethod
    def from_json(cls, path: str | Path) -> Pokedex:
        """Load a JSON list of species entries."""
        with Path(path).open(encoding="utf-8") as fh:
            raw = json.load(fh)
        if not isinstance(raw, list):
            raise PokedexError(f"{path}: expected a JSON list of species")
        return cls(PokemonGen2.from_dict(entry) for entry in raw)

    def get(self, name: str) -> PokemonGen2:
        """The species called ``name``; raises PokedexError if unknown."""
        try:
            return self._species[name]
        except KeyError:
            raise PokedexError(f"unknown species {name!r}") from None

    def find(self, name: str) -> PokemonGen2 | None:
        """The species called ``name``, or None if unknown."""
        return self._species.get(name)

    def validate(self, records: Iterable[Mapping[str, Any]]) -> list[str]:
        """Compare the Pokédex with the species data inside battle records.

        Returns a list of human-readable problems (unknown species,
        different types or base stats); an empty list means consistent.
        """
        problems: set[str] = set()
        for record in records:
            for raw in [*record["p1_team_details"], record["p2_lead_details"]]:
                problems.update(self._compare(raw))
            for turn in record["battle_timeline"]:
                for side in ("p1", "p2"):
                    name = turn[f"{side}_pokemon_state"]["name"]
                    if name not in self._species:
                        problems.add(f"{name}: not in the Pokédex")
        return sorted(problems)

    def _compare(self, raw: Mapping[str, Any]) -> list[str]:
        known = self.find(raw["name"])
        if known is None:
            return [f"{raw['name']}: not in the Pokédex"]
        seen = PokemonGen2.from_dict(raw)
        if seen == known:
            return []
        return [f"{raw['name']}: record {seen} differs from Pokédex {known}"]

    @property
    def names(self) -> list[str]:
        return list(self._species)

    def __contains__(self, name: object) -> bool:
        return name in self._species

    def __len__(self) -> int:
        return len(self._species)

    def __iter__(self) -> Iterator[PokemonGen2]:
        return iter(self._species.values())
