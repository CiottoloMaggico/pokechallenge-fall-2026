"""Errors of the domain: raised only when the program cannot continue.

Data artifacts are not errors: they are handled by policies.
"""

from __future__ import annotations


class DomainError(Exception):
    """Base class of every error raised by the domain."""


class InvalidRecordError(DomainError):
    """A battle record is malformed.

    ``battle_id`` is None when the record has no usable id; ``path`` is the
    ``file:line`` it was read from, None when it was not read from a file.
    """

    def __init__(self, message: str, battle_id: object = None, path: str | None = None) -> None:
        super().__init__(message)
        self.battle_id = battle_id
        self.path = path


class UnknownSpeciesError(DomainError):
    """A species is not in the Pokédex."""


class ReplayError(DomainError):
    """A timeline is impossible (for example, two active Pokémon)."""


class TypeChartError(DomainError, ValueError):
    """The type chart file is malformed."""
