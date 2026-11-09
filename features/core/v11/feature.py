"""``V11Features``: the whole v11 feature pipeline as one feature of the repository.

v11 fits on the training battles alone and cross-fits its own training rows (detector and
forecast), so the feature declares ``uses_labels`` (``FeatureSet.fit`` then passes it only the
training battles, never unlabeled ones) without reading the labels, and ``inner_oof = False``.
It reads ``context.record``, the raw record: the repository's knockout repair does not change its
values, because v11 runs its own hidden-faint detector.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from domain.battle import BattleContext
from domain.records import fingerprint
from features.core.v11.build import V11State, fit_training, new_matrix
from interfaces import Feature

# The groups of blend_v11's four pipelines: every column but ``cut``, in the matrix order.
OUTPUT_PREFIXES = (
    "s_",
    "f_",
    "t_",
    "a_",
    "e_",
    "m_",
    "w_",
    "x_",
    "bs_",
    "bh_",
    "bm_",
    "d_",
    "v_",
    "z_",
    "c_",
    "q_",
    "k_",
)

Key = tuple[object, str]


def battle_key(context: BattleContext) -> Key:
    """(battle id, content hash): training duplicates have distinct ids, and no test battle of
    the competition files shares both with a training one."""
    return context.record["battle_id"], fingerprint(context.record)


class V11Features(Feature[BattleContext]):
    """v11's columns: cross-fitted values for the training battles, full-model values for the
    others (computed in batches by ``prepare`` and cached)."""

    requires_fit = True
    uses_labels = True
    inner_oof = False

    def __init__(self) -> None:
        super().__init__("v11")
        self._columns: tuple[str, ...] | None = None
        self._state: V11State | None = None
        self._train_rows: dict[Key, np.ndarray] = {}
        self._cache: dict[Key, np.ndarray] = {}

    @property
    def output_names(self) -> tuple[str, ...] | None:
        """The output columns, known after ``fit``."""
        return self._columns

    def _fit(self, battles: Sequence[BattleContext], labels: Sequence[int] | None) -> None:
        X, self._state = fit_training([c.record for c in battles])
        columns = [c for c in X.columns if c.startswith(OUTPUT_PREFIXES)]
        values = X[columns].to_numpy()
        self._train_rows = {battle_key(c): values[k] for k, c in enumerate(battles)}
        self._columns = tuple(columns)
        self._cache = {}

    def prepare(self, battles: Sequence[BattleContext]) -> None:
        """Compute in one batch the rows of the battles that are neither training nor cached."""
        new: dict[Key, BattleContext] = {}
        for context in battles:
            key = battle_key(context)
            if key not in self._train_rows and key not in self._cache:
                new.setdefault(key, context)
        if not new:
            return
        assert self._state is not None and self._columns is not None
        X = new_matrix(self._state, [c.record for c in new.values()])
        values = X[list(self._columns)].to_numpy()
        self._cache.update(zip(new, values, strict=True))

    def compute(self, battle: BattleContext) -> Mapping[str, float]:
        """The training row of a training battle, else the (cached) full-model row."""
        assert self._columns is not None
        key = battle_key(battle)
        row = self._train_rows.get(key)
        if row is None:
            if key not in self._cache:
                self.prepare([battle])
            row = self._cache[key]
        return dict(zip(self._columns, row.tolist(), strict=True))
