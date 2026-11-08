"""The one feature of the detector's sets: columns of ``domain.transitions`` picked by name."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from domain.transitions import ALL_COLUMNS, Transition
from feature_set import FeatureSet
from interfaces import Feature


class TransitionColumns(Feature[Transition]):
    """The listed columns of a switch, in the listed order; it never reads ``fainted``."""

    def __init__(self, columns: Sequence[str]) -> None:
        super().__init__("transition_columns")
        unknown = [c for c in columns if c not in ALL_COLUMNS]
        if unknown:
            raise ValueError(f"not columns of domain.transitions: {unknown}")
        self.columns = tuple(columns)

    @property
    def output_names(self) -> tuple[str, ...]:
        return self.columns

    def compute(self, battle: Transition) -> Mapping[str, float]:
        return {name: battle.features[name] for name in self.columns}


def transition_set(name: str, columns: Sequence[str]) -> FeatureSet[Transition]:
    """A detector feature set called ``name`` that keeps ``columns``."""
    features = FeatureSet[Transition](name)
    features.register(TransitionColumns(columns))
    return features
