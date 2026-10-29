"""Data layer: JSONL records -> replayed battles ready for the features.

Pipeline::

    pokedex = Pokedex.from_json("assets/species.json")
    records = load_records("input/train.jsonl")
    train = Dataset.from_records(records, pokedex)

    gym.evaluate(train.contexts, train.labels, ids=train.ids)

Each battle is replayed into a :class:`battle.BattleContext`, which keeps
the final state, the raw record and one snapshot per turn.

Unknown species (not in the Pokédex) never stop a prediction: with the
default ``UnknownSpecies.RANDOM`` each one is replaced, everywhere in its
battle, by a random Pokédex species not already in that battle. The
choice depends only on the battle id and a seed (reproducible, no
labels involved), and every replacement is listed in
:attr:`Dataset.substitutions`.

Leakage rules enforced here:

* labels are kept only in :class:`Dataset`, never inside the contexts
  that features receive (the stored record has no ``player_won``);
* :meth:`Dataset.duplicate_groups` gives one group id per distinct
  battle, so identical battles never end up on both sides of a split.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import random
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from domain.battle import LABEL_KEY, BattleContext, Repair
from domain.pokedex import PLACEHOLDER_TYPES, STATS, Pokedex, PokemonGen2

Record = dict[str, Any]

REQUIRED_KEYS = ("battle_id", "p1_team_details", "p2_lead_details", "battle_timeline")
SIDES = ("p1", "p2")

logger = logging.getLogger(__name__)


class DataError(ValueError):
    """A record is malformed or cannot be replayed."""


def load_records(path: str | Path) -> list[Record]:
    """Read a JSONL file and check that every record has the needed keys."""
    records = []
    with Path(path).open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DataError(f"{path}:{line_no}: invalid JSON") from exc
            missing = [k for k in REQUIRED_KEYS if k not in record]
            if missing:
                raise DataError(f"{path}:{line_no}: missing keys {missing}")
            records.append(record)
    return records


class UnknownSpecies(Enum):
    """What to do with a species missing from the Pokédex."""

    RAISE = "raise"
    RANDOM = "random"  # replace it with a random Pokédex species


@dataclass(frozen=True)
class Substitution:
    """An unknown species replaced in one battle."""

    battle_id: Any
    original: str
    replacement: str


def species_names(record: Record) -> list[str]:
    """Every species name in a record, in order of first appearance."""
    names = [d["name"] for d in record["p1_team_details"]]
    names.append(record["p2_lead_details"]["name"])
    for turn in record["battle_timeline"]:
        names.extend(turn[f"{side}_pokemon_state"]["name"] for side in SIDES)
    return list(dict.fromkeys(names))


def substitute_unknown(
    record: Record, pokedex: Pokedex, seed: int = 0
) -> tuple[Record, list[Substitution]]:
    """Replace species missing from ``pokedex`` with random known ones.

    Each unknown name is replaced consistently in the whole battle (team
    details, lead, timeline) by a species not already in the battle; the
    details get the replacement's types and base stats, keeping level
    and item. The random generator is seeded with ``seed`` and the
    battle id, so the result does not depend on the dataset order. The
    input record is not modified.
    """
    names = species_names(record)
    unknown = [n for n in names if n not in pokedex]
    if not unknown:
        return record, []
    rng = random.Random(f"{seed}:{record['battle_id']}")
    candidates = sorted(set(pokedex.names) - set(names))
    if len(candidates) < len(unknown):
        raise DataError(f"battle {record['battle_id']}: not enough species")
    mapping = dict(zip(unknown, rng.sample(candidates, len(unknown)), strict=True))
    fixed = copy.deepcopy(record)
    for details in [*fixed["p1_team_details"], fixed["p2_lead_details"]]:
        if details["name"] in mapping:
            details.update(_details(pokedex.get(mapping[details["name"]])))
    for turn in fixed["battle_timeline"]:
        for side in SIDES:
            state = turn[f"{side}_pokemon_state"]
            state["name"] = mapping.get(state["name"], state["name"])
    return fixed, [
        Substitution(record["battle_id"], old, new) for old, new in mapping.items()
    ]


def _details(species: PokemonGen2) -> dict[str, Any]:
    """Species fields of a team-details entry (types padded as in records)."""
    types = list(species.types)
    types += sorted(PLACEHOLDER_TYPES)[: 2 - len(types)]
    return {
        "name": species.name,
        "types": types,
        **{f"base_{s}": species.base(s) for s in ("hp", *STATS)},
    }


def species_from_records(records: Iterable[Record]) -> list[PokemonGen2]:
    """Every species whose data appears in the records, sorted by name.

    Species data is read from ``p1_team_details`` and ``p2_lead_details``
    (P2 Pokémon revealed later have no data). The same species must have
    the same types and base stats everywhere, or a DataError is raised.
    """
    found: dict[str, PokemonGen2] = {}
    for record in records:
        for raw in [*record["p1_team_details"], record["p2_lead_details"]]:
            species = PokemonGen2.from_dict(raw)
            known = found.setdefault(species.name, species)
            if known != species:
                raise DataError(
                    f"battle {record['battle_id']}: {species} differs from {known}"
                )
    return [found[name] for name in sorted(found)]


def replay(
    record: Record, pokedex: Pokedex, repair: Repair | None = None
) -> BattleContext:
    """Replay one record; any failure becomes a DataError with its battle id."""
    try:
        return BattleContext.from_record(record, pokedex.get, repair)
    except (ValueError, RuntimeError, KeyError) as exc:  # includes PokedexError
        raise DataError(f"battle {record.get('battle_id')}: {exc}") from exc


@dataclass(frozen=True)
class Dataset:
    """Replayed battles, their ids and, for training data, their labels.

    ``label_values`` is None for unlabelled data (the competition test
    set). ``substitutions`` lists the unknown species that were replaced.
    """

    contexts: tuple[BattleContext, ...]
    label_values: tuple[int, ...] | None = None
    substitutions: tuple[Substitution, ...] = field(default=())

    def __post_init__(self) -> None:
        if self.label_values is not None:
            if len(self.label_values) != len(self.contexts):
                raise DataError("one label per battle is required")
            if not set(self.label_values) <= {0, 1}:
                raise DataError("labels must be 0 or 1")

    @classmethod
    def from_records(
        cls,
        records: Iterable[Record],
        pokedex: Pokedex,
        unknown_species: UnknownSpecies = UnknownSpecies.RANDOM,
        seed: int = 0,
        repair: Repair | None = None,
    ) -> Dataset:
        """Replay every record (see the module docstring for unknowns).

        ``repair`` finds the faints each record omits (e.g.
        ``knockout_detector.KnockoutDetector``); see ``BattleContext.from_record``.
        """
        records = list(records)
        labelled = [LABEL_KEY in r for r in records]
        if any(labelled) and not all(labelled):
            raise DataError("some records have a label and others do not")
        labels = (
            tuple(int(r[LABEL_KEY]) for r in records)
            if records and all(labelled)
            else None
        )
        substitutions: list[Substitution] = []
        if unknown_species is UnknownSpecies.RANDOM:
            fixed = []
            for record in records:
                record, done = substitute_unknown(record, pokedex, seed)
                fixed.append(record)
                substitutions.extend(done)
            records = fixed
        for sub in substitutions:
            logger.warning(
                "battle %s: unknown species %r replaced with %r",
                sub.battle_id,
                sub.original,
                sub.replacement,
            )
        contexts = tuple(replay(r, pokedex, repair) for r in records)
        return cls(contexts, labels, tuple(substitutions))

    @classmethod
    def from_jsonl(
        cls,
        path: str | Path,
        pokedex: Pokedex,
        unknown_species: UnknownSpecies = UnknownSpecies.RANDOM,
        seed: int = 0,
        repair: Repair | None = None,
    ) -> Dataset:
        records = load_records(path)
        return cls.from_records(records, pokedex, unknown_species, seed, repair)

    @property
    def ids(self) -> list[Any]:
        return [c.battle_id for c in self.contexts]

    @property
    def is_labelled(self) -> bool:
        return self.label_values is not None

    @property
    def labels(self) -> np.ndarray:
        """Labels as an int array; raises for unlabelled data."""
        if self.label_values is None:
            raise DataError("this dataset has no labels (test set?)")
        return np.asarray(self.label_values, dtype=int)

    def duplicate_groups(self) -> np.ndarray:
        """One integer group per distinct battle content (ids ignored).

        Pass it to ``Gym.evaluate(groups=...)``: battles with identical
        teams and timelines then always fall in the same fold.
        """
        keys: dict[str, int] = {}
        groups = []
        for context in self.contexts:
            content = {k: v for k, v in context.record.items() if k != "battle_id"}
            digest = hashlib.sha1(
                json.dumps(content, sort_keys=True).encode()
            ).hexdigest()
            groups.append(keys.setdefault(digest, len(keys)))
        return np.asarray(groups, dtype=int)

    def subset(self, indices: Sequence[int]) -> Dataset:
        labels = (
            None
            if self.label_values is None
            else tuple(self.label_values[i] for i in indices)
        )
        contexts = tuple(self.contexts[i] for i in indices)
        ids = {c.battle_id for c in contexts}
        subs = tuple(s for s in self.substitutions if s.battle_id in ids)
        return Dataset(contexts, labels, subs)

    def __len__(self) -> int:
        return len(self.contexts)

    def __iter__(self) -> Iterator[BattleContext]:
        return iter(self.contexts)

    def __getitem__(self, index: int) -> BattleContext:
        return self.contexts[index]
