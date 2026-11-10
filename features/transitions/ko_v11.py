"""``ko_v11``: the columns of v11's hidden-faint detector, one row per switch (``domain.switches``).

Every column of ``domain.switches.COLUMNS``, in the original's order; booleans and integers
become floats, as the original's ``astype(float)``. The set is not in ``TRANSITION_SETS``:
``detector train`` and ``detector evaluate`` feed ``domain.transitions`` rows, while this set
reads ``Switch`` rows (``detector train-v11``, ``addons.knockouts_v11``).
"""

from __future__ import annotations

from collections.abc import Mapping

from domain.switches import COLUMNS, Switch
from feature_set import FeatureSet
from interfaces import Feature

NAME = "ko_v11"


class SwitchColumns(Feature[Switch]):
    """The described columns of a switch; it never reads ``recorded`` (the detector's label)."""

    def __init__(self) -> None:
        super().__init__("switch_columns")

    @property
    def output_names(self) -> tuple[str, ...]:
        return COLUMNS

    def compute(self, battle: Switch) -> Mapping[str, float]:
        return {name: float(battle.features[name]) for name in COLUMNS}


def build() -> FeatureSet[Switch]:
    """The detector set of v11."""
    features = FeatureSet[Switch](NAME)
    features.register(SwitchColumns())
    return features
