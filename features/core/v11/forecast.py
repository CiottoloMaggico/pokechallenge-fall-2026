"""v11's learned forecast of the next turns (``q_`` columns), as one fitted ``Feature``.

LightGBM regressors map the state after ``c`` entries (the non-bag groups recomputed on the
timeline truncated at ``c``, then ``cut``) to the change of the faint difference, of the HP
difference and of each player's faints over the next ``h`` entries; applied to the last entry
of a battle, they forecast how it goes on. Label-free.

``fit`` follows the original's ``truncated_features`` and ``forecast_features``: on the training
battles, in their order, the inputs of the full battles and of ``battle.at_turn(c)`` for each
cut (replayed with the same repair: with the v11 repair, the detector of that cut), with its
own typical moves, team model and switch rates fitted on the full battles; the targets from the
tracks of the full battles; the out-of-fold ``q_*`` of the fitted battles (``KFold`` over them,
not grouped, as the original) are kept by battle id and ``fingerprint``, and the regressors are
fitted on all truncated rows. ``prepare`` predicts the other battles in one batch per regressor;
``compute`` gives a fitted battle its out-of-fold row, any other battle its predicted row.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold

from domain.battle import BattleContext
from domain.records import fingerprint
from features.core.v11.active import active_features
from features.core.v11.battle import battle_of
from features.core.v11.condition import condition_features
from features.core.v11.events import event_features
from features.core.v11.matchup import matchup_features, wincon_features
from features.core.v11.simulation import MoveTable, duel_features
from features.core.v11.simulation2 import simulation_features
from features.core.v11.sleep import sleep_features
from features.core.v11.state import Track, state_features, tracks_of
from features.core.v11.static import static_features
from features.core.v11.switching import SwitchRates, behaviour_features
from features.core.v11.team_model import TeamModel
from interfaces import Feature

SEED = 42
N_JOBS = 4
CUTS = (12, 14, 16, 18, 20, 22, 24)
TARGETS = {10: ("fd", "hd", "f1", "f2"), 5: ("fd", "hd")}
OUTPUTS = tuple(f"q_{t}_h{h}" for h, targets in TARGETS.items() for t in targets)
PARAMS: dict[str, Any] = dict(
    n_estimators=400,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=50,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.5,
    deterministic=True,
    force_row_wise=True,
    random_state=SEED,
    n_jobs=N_JOBS,
    verbose=-1,
)

Key = tuple[Any, str]


class Helpers(NamedTuple):
    """The forecast's own fitted state of the input groups (fitted on the full battles)."""

    moves: MoveTable
    teams: TeamModel
    rates: SwitchRates


def battle_key(context: BattleContext) -> Key:
    """(battle id, content hash): training duplicates have distinct ids, and no test battle of
    the competition files shares both with a training one."""
    return context.record["battle_id"], fingerprint(context.record)


def inputs(context: BattleContext, helpers: Helpers, cut: float) -> dict[str, float]:
    """The regressors' inputs: the non-bag groups in matrix order (``s_`` … ``c_``), ``cut``."""
    b, (t1, t2) = battle_of(context), tracks_of(context)
    row: dict[str, float] = {}
    for values in (
        static_features(b),
        state_features(b, t1, t2),
        active_features(b),
        event_features(b),
        matchup_features(b, t1, t2),
        wincon_features(b, t1, t2),
        sleep_features(b, t1, t2),
        duel_features(b, t1, t2, helpers.moves),
        behaviour_features(b, helpers.rates),
        simulation_features(b, t1, t2, helpers.moves, helpers.teams),
        condition_features(b, t1, t2, helpers.teams),
    ):
        row.update((name, float(value)) for name, value in values.items())
    row["cut"] = cut
    return row


def paths(tracks: Sequence[tuple[Track, Track]], n: int) -> dict[str, np.ndarray]:
    """Per-entry trajectories: P1 and P2 faints (estimated hidden P2 faints included), their
    difference and the HP difference."""
    out = {k: np.zeros((len(tracks), n)) for k in ("f1", "f2", "hd")}
    for k, (t1, t2) in enumerate(tracks):
        out["f1"][k], out["f2"][k] = t1["faint_cum"], t2["faint_cum"]
        out["hd"][k] = t1["hp_tot"] - t2["hp_tot"]
    out["fd"] = out["f2"] - out["f1"]
    return out


def _regressor() -> Any:
    return LGBMRegressor(**PARAMS)


class V11Forecast(Feature[BattleContext]):
    """``q_*``: the forecast of the faint and HP differences over the next 5 and 10 entries."""

    requires_fit = True

    def __init__(self) -> None:
        super().__init__()
        self._helpers: Helpers | None = None
        self._cut = float("nan")
        self._regressors: dict[str, Any] = {}
        self._train_rows: dict[Key, list[float]] = {}
        self._predicted: dict[Key, list[float]] = {}

    @property
    def output_names(self) -> tuple[str, ...]:
        return OUTPUTS

    def _fit(self, battles: Sequence[BattleContext], labels: Sequence[int] | None) -> None:
        views = [battle_of(c) for c in battles]
        helpers = Helpers(MoveTable(views), TeamModel(views), SwitchRates(views))
        self._helpers, self._cut = helpers, float(len(battles[0].record["battle_timeline"]))
        ids = np.array([c.battle_id for c in battles])
        x = pd.DataFrame([inputs(c, helpers, self._cut) for c in battles], index=ids)
        truncated = pd.concat(
            [
                pd.DataFrame(
                    [inputs(c.at_turn(cut), helpers, float(cut)) for c in battles], index=ids
                )
                for cut in CUTS
            ]
        )
        traj = paths([tracks_of(c) for c in battles], views[0].n)
        oof = self._fit_regressors(x, truncated, traj)
        self._train_rows = {battle_key(c): oof[k].tolist() for k, c in enumerate(battles)}
        self._predicted = {}

    def _fit_regressors(
        self, x: pd.DataFrame, truncated: pd.DataFrame, traj: Mapping[str, np.ndarray]
    ) -> np.ndarray:
        """The out-of-fold ``q_*`` of the fitted battles (rows) and the regressors on all rows."""
        cols = list(x.columns)
        ids = x.index.to_numpy()
        row_of = {bid: k for k, bid in enumerate(ids)}
        folds = list(KFold(5, shuffle=True, random_state=SEED).split(ids))
        oof = np.zeros((len(ids), len(OUTPUTS)))
        for h, targets in TARGETS.items():
            z: Any = truncated[truncated["cut"] + h <= traj["fd"].shape[1]]
            bid = z.index.to_numpy()
            rows = np.array([row_of[b] for b in bid])
            start = z["cut"].to_numpy().astype(int) - 1
            z = z[cols]
            for t in targets:
                name = f"q_{t}_h{h}"
                y = traj[t][rows, start + h] - traj[t][rows, start]
                for tr, va in folds:
                    m = np.isin(bid, ids[tr])
                    oof[va, OUTPUTS.index(name)] = (
                        _regressor().fit(z[m], y[m]).predict(x.iloc[va][cols])
                    )
                self._regressors[name] = _regressor().fit(z, y)
        return oof

    def prepare(self, battles: Sequence[BattleContext]) -> None:
        """Predict in one batch the battles that are neither fitted nor already predicted."""
        new: dict[Key, BattleContext] = {}
        for context in battles:
            key = battle_key(context)
            if key not in self._train_rows and key not in self._predicted:
                new.setdefault(key, context)
        if not new:
            return
        assert self._helpers is not None
        x = pd.DataFrame([inputs(c, self._helpers, self._cut) for c in new.values()])
        predicted = np.column_stack([self._regressors[name].predict(x) for name in OUTPUTS])
        self._predicted.update(zip(new, predicted.tolist(), strict=True))

    def compute(self, battle: BattleContext) -> Mapping[str, float]:
        key = battle_key(battle)
        row = self._train_rows.get(key)
        if row is None:
            if key not in self._predicted:
                self.prepare([battle])
            row = self._predicted[key]
        return dict(zip(OUTPUTS, row, strict=True))
