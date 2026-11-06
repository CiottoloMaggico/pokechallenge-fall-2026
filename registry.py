"""Names -> feature sets and models, filled by decorators where they are declared.

A registry is filled when the module that declares its entries is imported
(``features.sets``, ``model_catalog``).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from domain.battle import BattleContext
    from feature_set import FeatureSet
    from model import Model
    from models.space import Trial

    FeatureSetBuilder = Callable[[int | None], FeatureSet[BattleContext]]
    ModelBuilder = Callable[..., Model]
    SearchSpace = Callable[[Trial], dict[str, Any]]

TURN_SEPARATOR = "@t"


class Registry[T](dict[str, T]):
    """Entries of one kind by name; an unknown name lists the valid ones, a repeated one fails."""

    def __init__(self, kind: str) -> None:
        super().__init__()
        self.kind = kind

    def add(self, name: str, item: T) -> None:
        """Register ``item``; a name is declared once."""
        if name in self:
            raise ValueError(f"{self.kind} {name!r} is already registered")
        self[name] = item

    def __missing__(self, name: str) -> T:
        raise KeyError(f"unknown {self.kind} {name!r}; choose from {sorted(self)}") from None


@dataclass(frozen=True)
class ModelSpec:
    """How to build a model, the hyperparameters to tune and the optional libraries it needs."""

    name: str
    build: ModelBuilder
    search_space: SearchSpace | None = None
    requires: tuple[str, ...] = ()


FEATURE_SETS: Registry[FeatureSetBuilder] = Registry("feature set")
MODELS: Registry[ModelSpec] = Registry("model")
ENSEMBLES: Registry[tuple[str, ...]] = Registry("ensemble")


def feature_set[B: Callable[..., Any]](name: str) -> Callable[[B], B]:
    """Register a builder ``(turn) -> FeatureSet`` whose set is called ``set_name(name, turn)``."""

    def register(builder: B) -> B:
        FEATURE_SETS.add(name, builder)
        return builder

    return register


def model[B: Callable[..., Any]](
    name: str, space: SearchSpace | None = None, requires: tuple[str, ...] = ()
) -> Callable[[B], B]:
    """Register a model builder; keyword arguments of the builder override its defaults."""

    def register(builder: B) -> B:
        MODELS.add(name, ModelSpec(name, builder, space, requires))
        return builder

    return register


def ensemble[B: Callable[..., Any]](name: str, members: Sequence[str]) -> Callable[[B], B]:
    """Register an average of models already registered; it needs the libraries of its members."""

    def register(builder: B) -> B:
        requires = dict.fromkeys(library for m in members for library in MODELS[m].requires)
        MODELS.add(name, ModelSpec(name, builder, requires=tuple(requires)))
        ENSEMBLES.add(name, tuple(members))
        return builder

    return register


def set_name(base: str, turn: int | None) -> str:
    """``base`` or ``base@t<turn>``."""
    return base if turn is None else f"{base}{TURN_SEPARATOR}{turn}"


def parse_name(name: str) -> tuple[str, int | None]:
    """Split ``"core@t10"`` into ``("core", 10)``; no suffix gives None."""
    base, separator, turn = name.partition(TURN_SEPARATOR)
    if not separator:
        return name, None
    if not turn.isdigit() or int(turn) < 1:
        raise ValueError(f"invalid turn in feature set name {name!r}")
    return base, int(turn)


def build_feature_set(name: str, turn: int | None = None) -> FeatureSet[BattleContext]:
    """Build a feature set from its name (``"core"``, ``"core@t10"``).

    ``turn`` can be given separately instead of the ``@t`` suffix.
    """
    base, parsed = parse_name(name)
    if parsed is not None and turn is not None and parsed != turn:
        raise ValueError(f"{name!r} conflicts with turn={turn}")
    sampled = parsed if parsed is not None else turn
    features = FEATURE_SETS[base](sampled)
    expected = set_name(base, sampled)
    if features.name != expected:
        raise ValueError(f"builder {base!r} returned feature set {features.name!r}")
    return features
