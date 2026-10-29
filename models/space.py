"""What a search space needs from a trial (the subset of ``optuna.Trial`` used).

Each model module exposes ``search_space(trial) -> dict`` returning
keyword arguments for its ``build``; the tuner passes an Optuna trial,
but any object with these methods works (e.g. in tests).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol


class Trial(Protocol):
    def suggest_float(
        self, name: str, low: float, high: float, *, log: bool = False
    ) -> float: ...

    def suggest_int(
        self, name: str, low: int, high: int, *, log: bool = False
    ) -> int: ...

    def suggest_categorical(self, name: str, choices: Sequence[Any]) -> Any: ...
