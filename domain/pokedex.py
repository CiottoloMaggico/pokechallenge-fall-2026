"""Gen 2 species data and stat formulas.

A :class:`PokemonGen2` is a *species*: name, types and base stats. It
holds nothing about an individual Pokémon; level, DVs and stat
experience belong to the individual (see ``battle.PokemonState``) and
are passed to the stat methods.

The :class:`Pokedex` comes only from battle records in the competition
format: their ``p1_team_details`` and ``p2_lead_details`` are the only
places with base stats. Only the training file builds it; a validation or
test battle only adds the stats it carries itself (:meth:`Pokedex.for_battle`).
P2 Pokémon revealed after the lead may have no stats in any record: nothing
is invented for them, an :class:`UnknownSpeciesPolicy` decides what happens.

Example::

    pokedex = PokedexBuilder().add_records(read_records("input/train.jsonl")).build()
    snorlax = pokedex.get("snorlax")
    snorlax.max_hp(level=100)        # 523
    snorlax.stat("atk", level=100)   # 318
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from domain.errors import PokedexError, UnknownSpeciesError

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

MISSING_STAT = float("nan")


def stat_points(stat_exp: int = MAX_STAT_EXP) -> int:
    """Gen 2 stat-experience bonus: min(255, isqrt(exp - 1) + 1) // 4.

    0 stat experience gives 0; the maximum (65535) gives 63.
    """
    return min(255, math.isqrt(max(0, stat_exp - 1)) + 1) // 4


def calc_hp(base: float, level: int, dv: int = MAX_DV, stat_exp: int = MAX_STAT_EXP) -> float:
    """Gen 2 HP formula (an integer, or NaN for a base without stats)."""
    core = ((base + dv) * 2 + stat_points(stat_exp)) * level // 100
    return core + level + 10


def calc_stat(base: float, level: int, dv: int = MAX_DV, stat_exp: int = MAX_STAT_EXP) -> float:
    """Gen 2 formula for Attack, Defense, Sp. Atk, Sp. Def and Speed
    (an integer, or NaN for a base without stats)."""
    return ((base + dv) * 2 + stat_points(stat_exp)) * level // 100 + 5


@dataclass(frozen=True)
class PokemonGen2:
    """A Gen 2 species: name, real types and base stats.

    Base stats are integers, except for :meth:`without_stats`: there they
    are NaN, so that whatever is computed from them is NaN too.
    """

    name: str
    types: tuple[str, ...]
    base_hp: float
    base_atk: float
    base_def: float
    base_spa: float
    base_spd: float
    base_spe: float

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PokemonGen2:
        """Build a species from a record; placeholder types are dropped."""
        try:
            types = tuple(t for t in data["types"] if t not in PLACEHOLDER_TYPES)
            species = cls(
                str(data["name"]),
                types,
                *(int(data[f"base_{stat}"]) for stat in ("hp", *STATS)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise PokedexError(f"invalid species entry {dict(data)!r}") from exc
        species.validate()
        return species

    @classmethod
    def without_stats(cls, name: str) -> PokemonGen2:
        """A species revealed by a battle but with no stats in any record."""
        return cls(name, (), *[MISSING_STAT] * (1 + len(STATS)))

    @property
    def has_stats(self) -> bool:
        """False for :meth:`without_stats`: its types are unknown as well."""
        return not math.isnan(self.base_hp)

    def validate(self) -> None:
        """Raise PokedexError unless the name, types and base stats are valid."""
        if not self.name:
            raise PokedexError("species without a name")
        if not 1 <= len(self.types) <= 2 or not set(self.types) <= TYPES:
            raise PokedexError(f"{self.name}: invalid types {self.types}")
        if any(not 1 <= self.base(s) <= 255 for s in ("hp", *STATS)):
            raise PokedexError(f"{self.name}: base stats must be in 1..255")

    def base(self, stat: str) -> float:
        """Base value of ``"hp"``, ``"atk"``, ``"def"``, ... ."""
        if stat not in ("hp", *STATS):
            raise KeyError(f"unknown stat {stat!r}")
        return getattr(self, f"base_{stat}")

    def max_hp(self, level: int, dv: int = MAX_DV, stat_exp: int = MAX_STAT_EXP) -> float:
        """Maximum HP at ``level``."""
        return calc_hp(self.base_hp, level, dv, stat_exp)

    def stat(
        self,
        stat: str,
        level: int,
        dv: int = MAX_DV,
        stat_exp: int = MAX_STAT_EXP,
    ) -> float:
        """Real value of a non-HP stat at ``level``."""
        if stat == "hp":
            return self.max_hp(level, dv, stat_exp)
        return calc_stat(self.base(stat), level, dv, stat_exp)

    @property
    def base_stat_total(self) -> float:
        """Sum of the six base stats."""
        return sum(self.base(s) for s in ("hp", *STATS))


SpeciesLookup = Callable[[str], PokemonGen2 | None]
"""A species by name; None when no record has its stats (``Missing`` policy)."""


class UnknownSpeciesPolicy(Protocol):
    """What to do with a species that has no stats in any available record."""

    def resolve(
        self, name: str, record: Mapping[str, Any], pokedex: Pokedex
    ) -> PokemonGen2 | None: ...


class Strict:
    """Refuse the battle, for checks of the data."""

    def resolve(self, name: str, record: Mapping[str, Any], pokedex: Pokedex) -> PokemonGen2 | None:
        """Raise UnknownSpeciesError."""
        raise UnknownSpeciesError(f"battle {record['battle_id']}: no stats for species {name!r}")


class Missing:
    """Keep the battle: features that need the stats give NaN, damage is not computed."""

    def resolve(self, name: str, record: Mapping[str, Any], pokedex: Pokedex) -> PokemonGen2 | None:
        """Return None."""
        return None


class Pokedex:
    """Read-only collection of Gen 2 species, indexed by name."""

    def __init__(self, species: Iterable[PokemonGen2], source_sha256: str | None = None) -> None:
        """Index ``species`` by name.

        Repeated identical entries are kept once; two different entries
        with the same name raise PokedexError. ``source_sha256`` is the
        hash of the file the species were read from, when there is one.
        """
        self.source_sha256 = source_sha256
        self._species: dict[str, PokemonGen2] = {}
        for entry in species:
            _add(self._species, entry)

    @classmethod
    def from_json(cls, path: str | Path) -> Pokedex:
        """Load a file written by :meth:`to_json`."""
        try:
            with Path(path).open(encoding="utf-8") as fh:
                raw = json.load(fh)
        except FileNotFoundError:
            raise PokedexError(
                f"{path} not found: build it with "
                f"`python -m domain build-pokedex input/train.jsonl --out {path}`"
            ) from None
        return cls((PokemonGen2.from_dict(entry) for entry in raw["species"]), raw["source_sha256"])

    def to_json(self, path: str | Path) -> None:
        """Write the species, sorted by name, and the hash of their source file."""
        species = [
            {
                "name": s.name,
                "types": list(s.types),
                **{f"base_{stat}": s.base(stat) for stat in ("hp", *STATS)},
            }
            for s in sorted(self, key=lambda s: s.name)
        ]
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        content = {"source_sha256": self.source_sha256, "species": species}
        target.write_text(json.dumps(content, indent=1) + "\n", encoding="utf-8")

    def get(self, name: str) -> PokemonGen2:
        """The species called ``name``; raises UnknownSpeciesError if unknown."""
        try:
            return self._species[name]
        except KeyError:
            raise UnknownSpeciesError(f"unknown species {name!r}") from None

    def find(self, name: str) -> PokemonGen2 | None:
        """The species called ``name``, or None if unknown."""
        return self._species.get(name)

    def for_battle(self, record: Mapping[str, Any]) -> Pokedex:
        """This Pokédex plus the species whose stats are in ``record``.

        It reads the battle being replayed instead of fitting on it, so it
        is allowed on validation and test battles. Stats contradicting
        this Pokédex raise PokedexError.
        """
        own = _species_in(record)
        if all(self._species.get(s.name) == s for s in own):
            return self
        return Pokedex([*self, *own], self.source_sha256)

    def lookup(self, record: Mapping[str, Any], unknown: UnknownSpeciesPolicy) -> SpeciesLookup:
        """The species available to the replay of ``record``."""
        battle = self.for_battle(record)

        def resolve(name: str) -> PokemonGen2 | None:
            species = battle.find(name)
            return species if species is not None else unknown.resolve(name, record, battle)

        return resolve

    def __len__(self) -> int:
        return len(self._species)

    def __iter__(self) -> Iterator[PokemonGen2]:
        return iter(self._species.values())


class PokedexBuilder:
    """Collects the species of battle records in the competition format."""

    def __init__(self) -> None:
        self._species: dict[str, PokemonGen2] = {}

    def add_records(self, records: Iterable[Mapping[str, Any]]) -> PokedexBuilder:
        """Add every species with stats in ``records``; a contradiction raises PokedexError."""
        for record in records:
            for species in _species_in(record):
                _add(self._species, species)
        return self

    def build(self, source_sha256: str | None = None) -> Pokedex:
        """The Pokédex of the records added so far."""
        return Pokedex(self._species.values(), source_sha256)


def species_without_stats(records: Iterable[Mapping[str, Any]], pokedex: Pokedex) -> dict[str, int]:
    """Species seen in ``records`` with stats nowhere available -> battles with them."""
    missing: Counter[str] = Counter()
    for record in records:
        battle = pokedex.for_battle(record)
        names = {
            turn[f"{side}_pokemon_state"]["name"]
            for turn in record["battle_timeline"]
            for side in ("p1", "p2")
        }
        missing.update(name for name in names if battle.find(name) is None)
    return dict(sorted(missing.items()))


def _species_in(record: Mapping[str, Any]) -> list[PokemonGen2]:
    return [
        PokemonGen2.from_dict(details)
        for details in (*record["p1_team_details"], record["p2_lead_details"])
    ]


def _add(species: dict[str, PokemonGen2], entry: PokemonGen2) -> None:
    known = species.setdefault(entry.name, entry)
    if known != entry:
        raise PokedexError(
            f"species {entry.name!r} listed twice with different data: {known} and {entry}"
        )
