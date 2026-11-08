"""Data layer: JSONL records -> replayed battles ready for the features.

Pipeline::

    pokedex = Pokedex.from_json("results/cache/species.json")
    records = load_records("input/train.jsonl")
    train = Dataset.from_records(records, pokedex)

    gym.evaluate(train.labelled())

Each battle is replayed into a :class:`battle.BattleContext`, which keeps
the final state, the raw record and one state per turn.

Species with no stats in any record never stop a prediction: with the
default ``Missing`` policy the battle is replayed without them, features
that need their stats give NaN and no data is invented (``Strict`` raises
``UnknownSpeciesError`` instead).

Leakage rules enforced here:

* labels are kept only in :class:`Dataset`, never inside the contexts
  that features receive (the stored record has no ``player_won``);
* :meth:`Dataset.duplicate_groups` gives one group id per distinct
  battle, so identical battles never end up on both sides of a split.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from domain.battle import BattleContext, BattleReplay
from domain.errors import ReplayError
from domain.pokedex import Missing, Pokedex, UnknownSpeciesPolicy
from domain.records import LABEL_KEY, fingerprint, read_records
from domain.repair import NO_REPAIR, TimelineRepair
from interfaces import LabelledBattles

Record = dict[str, Any]

KEEP_BATTLE = Missing()


class DataError(ValueError):
    """A record is malformed or cannot be replayed."""


def load_records(path: str | Path) -> list[Record]:
    """Read a JSONL file; every record is validated (InvalidRecordError)."""
    return cast(list[Record], list(read_records(path)))


def _replay(replayer: BattleReplay, record: Record) -> BattleContext:
    """Replay one record; a failure becomes a DataError with its battle id,
    except UnknownSpeciesError, which is what ``Strict`` asks for."""
    try:
        return replayer.replay(record)
    except (ValueError, RuntimeError, KeyError, ReplayError) as exc:  # includes PokedexError
        raise DataError(f"battle {record.get('battle_id')}: {exc}") from exc


@dataclass(frozen=True)
class Dataset:
    """Replayed battles, their ids and, for training data, their labels.

    ``label_values`` is None for unlabelled data (the competition test
    set).
    """

    contexts: tuple[BattleContext, ...]
    label_values: tuple[int, ...] | None = None

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
        unknown: UnknownSpeciesPolicy = KEEP_BATTLE,
        repair: TimelineRepair = NO_REPAIR,
    ) -> Dataset:
        """Replay every record (see the module docstring for unknowns).

        ``repair`` finds the faints each record omits (e.g.
        ``knockout_detector.DetectorRepair``); see ``BattleReplay``.
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
        replayer = BattleReplay(pokedex, repair, unknown)
        contexts = tuple(_replay(replayer, r) for r in records)
        return cls(contexts, labels)

    @classmethod
    def from_jsonl(
        cls,
        path: str | Path,
        pokedex: Pokedex,
        unknown: UnknownSpeciesPolicy = KEEP_BATTLE,
        repair: TimelineRepair = NO_REPAIR,
    ) -> Dataset:
        records = load_records(path)
        return cls.from_records(records, pokedex, unknown, repair)

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

        Battles with identical teams and timelines then always fall in the
        same fold (see :meth:`labelled`).
        """
        keys: dict[str, int] = {}
        groups = []
        for context in self.contexts:
            digest = fingerprint(context.record)
            groups.append(keys.setdefault(digest, len(keys)))
        return np.asarray(groups, dtype=int)

    def labelled(self) -> LabelledBattles:
        """The battles as the gym takes them; raises for unlabelled data."""
        return LabelledBattles(
            self.contexts, self.labels, np.asarray(self.ids), self.duplicate_groups()
        )

    def subset(self, indices: Sequence[int]) -> Dataset:
        labels = (
            None
            if self.label_values is None
            else tuple(self.label_values[i] for i in indices)
        )
        return Dataset(tuple(self.contexts[i] for i in indices), labels)

    def __len__(self) -> int:
        return len(self.contexts)

    def __iter__(self) -> Iterator[BattleContext]:
        return iter(self.contexts)

    def __getitem__(self, index: int) -> BattleContext:
        return self.contexts[index]
