"""The label-aware steps of v11's members as features of ``v11_fs``, and the spline expansion.

The faithful v11 members learn two things from the labels inside the model: the logit of the bag
columns (``stack0``) and the team effect (``team``). Here they are ``Feature``s with
``uses_labels = True``: ``FeatureSet.fit_transform`` gives their training rows out of fold
(grouped inner folds), any other battle gets the state fitted on all the training battles.

* ``V11BagLogit`` (``stack_logit_0``): the logistic regression (C=0.01) of the bag columns.
* ``V11TeamEffect`` (``tm*``): ``team_effect.TeamScores`` (the members' parameters) against one
  base logit, the logistic regression (C=0.003) of every non-key column of ``v11`` (bags
  included, as the boosters' ``stack_full``), cross-fitted on the feature's training battles
  (``StratifiedKFold(5, shuffle, 42)``, as the original), so a training battle's residual never
  uses its own label.
* ``V11Splines`` (``spl_<key>_<k>``, label-free): the cubic splines of the spline member's keys
  (median-imputed), fitted on the training battles.

Their inputs are v11's label-free columns, computed by fitted copies of the group features
(``LabelFreeInputs``). Fitting them costs minutes (the forecast), so they are fitted once: a
copy (the inner folds of ``fit_transform``) refitted on a subset of the battles they were fitted
on reuses them, as the set's own label-free columns; other battles refit them.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from domain.battle import BattleContext
from features.core.v11_fs import groups as g
from features.core.v11_fs.battle import battle_of
from features.core.v11_fs.forecast import Key, V11Forecast, battle_key
from features.core.v11_fs.team import team_keys
from features.core.v11_fs.team_effect import TeamScores
from interfaces import Feature

SEED = 42
BAG_C = 0.01
FULL_C = 0.003
BASE = "stack_logit_full"
# the team effect of the frozen members (TEAM of models/v11.py)
TEAM: dict[str, Any] = {"a": 1.0, "p2": True, "link": True, "pair": True, "soft": True, "raw": True}
TEAM_COLUMNS = ("tm", "tm_n1", "tm_n2", "tm_pair", "tm_npair", "tm_soft", "tm_nsoft")
TEAM_COLUMNS += ("tm_raw1", "tm_raw2")
# the spline keys of the frozen spline member (LR_SPLINE["keys"] of models/v11.py)
SPLINE_KEYS = (
    "f_d_fainted", "f_d_hp_sum", "f_p1_fainted", "f_p2_fainted", "f_p1_hp_sum", "f_p2_hp_sum",
    "f_d_strength", "t_hd_mean", "t_fd_mean", "a_ttk_adv", "m_win_mean", "f_d_alive", "a_a1_hp",
    "a_a2_hp", "f_d_hp_hard", "f_d_fainted_hard", "f_p2_unrevealed", "m_p1_cover", "m_p2_cover",
    "a_dmg12", "a_dmg21", "a_speed_logratio", "t_hd_delta10", "f_d_disabled", "e_d_dealt",
)  # fmt: skip
N_KNOTS, DEGREE = 4, 3


def logit_pipeline(C: float) -> Pipeline:
    """Median imputation, standardisation, L2 logistic regression: the regression of the
    frozen ``StackedLogit`` (``models.core.v11_stacking``), built here (no import of models)."""
    return Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("lr", LogisticRegression(C=C, max_iter=5000)),
        ]
    )


def non_key_groups() -> list[Feature[BattleContext]]:
    """v11's features but the team keys, in v11's order (its non-key columns)."""
    return [
        g.V11Static(), g.V11State(), g.V11Active(), g.V11Events(), g.V11Matchup(),
        g.V11WinConditions(), g.V11Sleep(), g.V11Bags(), g.V11Duel(), g.V11Behaviour(),
        g.V11Simulation(), g.V11Condition(), V11Forecast(),
    ]  # fmt: skip


class LabelFreeInputs:
    """Fitted copies of label-free features and the keys of the battles they were fitted on."""

    def __init__(self, features: list[Feature[BattleContext]]) -> None:
        self.features = features
        self.keys: frozenset[Key] = frozenset()

    def fit(self, battles: Sequence[BattleContext]) -> None:
        """Fit them on ``battles``, unless they were fitted on a superset of them."""
        wanted = {battle_key(c) for c in battles}
        if self.keys and wanted <= self.keys:
            return
        for feature in self.features:
            feature.reset()
            if feature.requires_fit:
                feature.fit(battles)
        self.keys = frozenset(wanted)

    def frame(self, battles: Sequence[BattleContext]) -> pd.DataFrame:
        """Their columns on ``battles`` (rows in order, ``RangeIndex``)."""
        for feature in self.features:
            feature.prepare(battles)
        rows = [{k: v for f in self.features for k, v in f(c).items()} for c in battles]
        return pd.DataFrame(rows)


class CachedRows(Feature[BattleContext]):
    """A fitted feature scoring battles in batches (``prepare``), rows kept by battle key."""

    requires_fit = True

    def __init__(self) -> None:
        super().__init__()
        self._rows: dict[Key, list[float]] = {}

    def _fit(self, battles: Sequence[BattleContext], labels: Sequence[int] | None) -> None:
        self._rows = {}
        self._learn(battles, None if labels is None else np.asarray(labels))

    @abstractmethod
    def _learn(self, battles: Sequence[BattleContext], y: np.ndarray | None) -> None:
        """Fit the state on the training battles (labels ``y`` if label-aware)."""

    @abstractmethod
    def _score(self, battles: Sequence[BattleContext]) -> np.ndarray:
        """One row of ``output_names`` per battle."""

    def prepare(self, battles: Sequence[BattleContext]) -> None:
        new: dict[Key, BattleContext] = {}
        for context in battles:
            key = battle_key(context)
            if key not in self._rows:
                new.setdefault(key, context)
        if new:
            self._rows.update(zip(new, self._score(list(new.values())).tolist(), strict=True))

    def compute(self, battle: BattleContext) -> Mapping[str, float]:
        key = battle_key(battle)
        if key not in self._rows:
            self.prepare([battle])
        assert self.output_names is not None
        return dict(zip(self.output_names, self._rows[key], strict=True))


class V11BagLogit(CachedRows):
    """``stack_logit_0``: the logit of the regression of v11's bag columns."""

    uses_labels = True

    def __init__(self) -> None:
        super().__init__()
        self.inputs = LabelFreeInputs([g.V11Bags()])
        self.model: Pipeline | None = None

    @property
    def output_names(self) -> tuple[str, ...]:
        return ("stack_logit_0",)

    def _learn(self, battles: Sequence[BattleContext], y: np.ndarray | None) -> None:
        self.inputs.fit(battles)
        self.model = logit_pipeline(BAG_C).fit(self.inputs.frame(battles), y)

    def _score(self, battles: Sequence[BattleContext]) -> np.ndarray:
        assert self.model is not None
        return self.model.decision_function(self.inputs.frame(battles))[:, None]


def key_frame(battles: Sequence[BattleContext], base: np.ndarray) -> pd.DataFrame:
    """The ``k_`` columns of the battles and the base logit, as ``TeamScores`` reads them."""
    return pd.DataFrame([team_keys(battle_of(c)) for c in battles]).assign(**{BASE: base})


class V11TeamEffect(CachedRows):
    """``tm*``: the team effect against a base logit of v11's non-key columns."""

    uses_labels = True

    def __init__(self) -> None:
        super().__init__()
        self.inputs = LabelFreeInputs(non_key_groups())
        self.scores = TeamScores(BASE, **TEAM)
        self.base: Pipeline | None = None
        self.state: tuple[Any, ...] | None = None

    @property
    def output_names(self) -> tuple[str, ...]:
        return TEAM_COLUMNS

    def _learn(self, battles: Sequence[BattleContext], y: np.ndarray | None) -> None:
        assert y is not None
        self.inputs.fit(battles)
        x = self.inputs.frame(battles)
        base = np.zeros(len(x))
        for tr, va in StratifiedKFold(5, shuffle=True, random_state=SEED).split(x, y):
            m = logit_pipeline(FULL_C).fit(x.iloc[tr], y[tr])
            base[va] = m.decision_function(x.iloc[va])
        self.base = logit_pipeline(FULL_C).fit(x, y)
        self.state = self.scores.fit(key_frame(battles, base), y)

    def _score(self, battles: Sequence[BattleContext]) -> np.ndarray:
        assert self.base is not None and self.state is not None
        base = self.base.decision_function(self.inputs.frame(battles))
        out = self.scores.encode(key_frame(battles, base), self.state)
        return np.column_stack([out[name] for name in TEAM_COLUMNS])


class V11Splines(CachedRows):
    """``spl_<key>_<k>``: cubic splines (4 knots) of the spline member's keys, median-imputed."""

    def __init__(self) -> None:
        super().__init__()
        self.groups: list[Feature[BattleContext]] = [
            g.V11State(), g.V11Active(), g.V11Events(), g.V11Matchup()
        ]  # fmt: skip
        self.splines: Pipeline | None = None

    @property
    def output_names(self) -> tuple[str, ...]:
        n = N_KNOTS + DEGREE - 1
        return tuple(f"spl_{key}_{k}" for key in SPLINE_KEYS for k in range(n))

    def _keys(self, battles: Sequence[BattleContext]) -> pd.DataFrame:
        rows = [{k: v for f in self.groups for k, v in f(c).items()} for c in battles]
        return pd.DataFrame(rows).loc[:, list(SPLINE_KEYS)]

    def _learn(self, battles: Sequence[BattleContext], y: np.ndarray | None) -> None:
        self.splines = Pipeline(
            [
                ("imp", SimpleImputer(strategy="median")),
                ("s", SplineTransformer(n_knots=N_KNOTS, degree=DEGREE)),
            ]
        ).fit(self._keys(battles))

    def _score(self, battles: Sequence[BattleContext]) -> np.ndarray:
        assert self.splines is not None
        return np.asarray(self.splines.transform(self._keys(battles)))
