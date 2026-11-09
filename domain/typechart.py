"""Gen 2 type effectiveness, loaded from a CSV file.

The file lists only the matchups that are not neutral::

    attacking_type,defending_type,multiplier
    fire,grass,2.0
    normal,ghost,0.0
    ...

Every pair missing from the file is neutral (1.0).

Example::

    chart = TypeChart.from_csv("assets/type_chart_gen2.csv")
    chart.multiplier("fire", "grass")                 # 2.0
    chart.effectiveness("ground", ("fire", "flying")) # 0.0
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping
from pathlib import Path

from domain.errors import TypeChartError
from domain.pokedex import TYPES

NEUTRAL = 1.0
ALLOWED_MULTIPLIERS = frozenset({0.0, 0.5, 1.0, 2.0})
CSV_COLUMNS = ("attacking_type", "defending_type", "multiplier")


class TypeChart:
    """Read-only table ``(attacking type, defending type) -> multiplier``."""

    def __init__(self, multipliers: Mapping[tuple[str, str], float]) -> None:
        for (attacking, defending), value in multipliers.items():
            unknown = {attacking, defending} - TYPES
            if unknown:
                raise TypeChartError(f"unknown types {sorted(unknown)}")
            if value not in ALLOWED_MULTIPLIERS:
                raise TypeChartError(
                    f"{attacking} -> {defending}: multiplier {value} not in "
                    f"{sorted(ALLOWED_MULTIPLIERS)}"
                )
        self._table = dict(multipliers)

    @classmethod
    def from_csv(cls, path: str | Path) -> TypeChart:
        """Load the non-neutral matchups from a CSV file."""
        table: dict[tuple[str, str], float] = {}
        with Path(path).open(encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            if tuple(reader.fieldnames or ()) != CSV_COLUMNS:
                raise TypeChartError(f"{path}: expected columns {CSV_COLUMNS}")
            for line_no, row in enumerate(reader, start=2):
                key = (row["attacking_type"].strip(), row["defending_type"].strip())
                if key in table:
                    raise TypeChartError(f"{path}:{line_no}: duplicate {key}")
                try:
                    table[key] = float(row["multiplier"])
                except ValueError as exc:
                    raise TypeChartError(f"{path}:{line_no}: bad multiplier") from exc
        return cls(table)

    def multiplier(self, attacking: str, defending: str) -> float:
        """Multiplier of one attacking type against one defending type."""
        self._check(attacking)
        self._check(defending)
        return self._table.get((attacking, defending), NEUTRAL)

    def effectiveness(self, move_type: str | None, defender: Iterable[str]) -> float:
        """Product of the multipliers against every type of the defender.

        ``move_type=None`` stands for a typeless attack (always 1.0).
        """
        if move_type is None:
            return NEUTRAL
        result = NEUTRAL
        for defending_type in defender:
            result *= self.multiplier(move_type, defending_type)
        return result

    def __len__(self) -> int:
        return len(self._table)

    @staticmethod
    def _check(type_name: str) -> None:
        if type_name not in TYPES:
            raise KeyError(f"unknown type {type_name!r}")
