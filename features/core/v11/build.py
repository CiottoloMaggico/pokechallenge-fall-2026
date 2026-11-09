"""v11's feature rows and matrices, as the original's ``build.py`` computes them.

``battle_features`` is the original's without ``usage_bags`` and ``hidden_features``
(``bt_ bc_ bf_ bg_ u_``), which no pipeline of ``blend_v11`` reads. ``fit_training`` runs the
original ``build()`` with the given records as train and no test; ``new_matrix`` treats new
battles as that ``build()`` treats its test battles, with the fitted state kept in a
``V11State``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import pandas as pd

from features.core.v11.active import active_features
from features.core.v11.bags import Vocab, bag_features
from features.core.v11.condition import condition_features
from features.core.v11.events import event_features
from features.core.v11.faint import _collect, _fit, fit_detector
from features.core.v11.forecast import CUTS, Regressors, fit_forecast, inputs, predict_forecast
from features.core.v11.matchup import matchup_features, wincon_features
from features.core.v11.mechanics import Pokedex, Record
from features.core.v11.replay import Battle, parse
from features.core.v11.simulation import MoveTable, duel_features
from features.core.v11.simulation2 import simulation_features
from features.core.v11.sleep import sleep_features
from features.core.v11.state import p1_track, p2_track, state_features
from features.core.v11.static import static_features
from features.core.v11.switching import SwitchRates, behaviour_features
from features.core.v11.team import team_keys
from features.core.v11.team_model import TeamModel

Helpers = tuple[Vocab, TeamModel, MoveTable, SwitchRates]


def battle_features(
    b: Battle, vocab: Vocab, team_model: TeamModel, moves: MoveTable, rates: SwitchRates
) -> dict[str, Any]:
    """``battle_id`` and every feature group of a battle whose ``p2_faint`` is already set."""
    t1, t2 = p1_track(b), p2_track(b)
    f: dict[str, Any] = {"battle_id": b.id}
    f.update(static_features(b))
    f.update(state_features(b, t1, t2))
    f.update(active_features(b))
    f.update(event_features(b))
    f.update(matchup_features(b, t1, t2))
    f.update(wincon_features(b, t1, t2))
    f.update(sleep_features(b, t1, t2))
    f.update(bag_features(b, vocab, t1, t2))
    f.update(team_keys(b))
    f.update(duel_features(b, t1, t2, moves))
    f.update(behaviour_features(b, rates))
    f.update(simulation_features(b, t1, t2, moves, team_model))
    f.update(condition_features(b, t1, t2, team_model))
    return f


def feature_matrix(battles: Sequence[Battle], helpers: Helpers) -> pd.DataFrame:
    """One float row of ``battle_features`` per battle, indexed by battle id."""
    rows = [battle_features(b, *helpers) for b in battles]
    return pd.DataFrame(rows).set_index("battle_id").astype(float)


def truncated_features(
    records: Sequence[Record],
    dex: Pokedex,
    helpers: Helpers,
    cuts: Sequence[int],
    columns: list[str],
) -> pd.DataFrame:
    """Training features recomputed on timelines truncated to ``cut`` entries (column ``cut``),
    with the helpers fitted on the full training battles and the hidden-faint detector refitted
    on the truncated ones."""
    frames = []
    for cut in cuts:
        battles = parse(records, dex, cut=cut)
        fit_detector(battles, [])
        X = feature_matrix(battles, helpers)
        X["cut"] = float(cut)
        frames.append(X[columns])
    return pd.concat(frames)


@dataclass
class V11State:
    """What ``new_matrix`` needs: v11's Pokédex, the detector fitted on all P1 rows, the
    helpers, the forecast regressors, the training ``cut`` and the ``battle_features`` columns."""

    dex: Pokedex
    detector: Any
    detector_columns: list[str]
    helpers: Helpers
    regressors: Regressors
    cut: float
    columns: list[str]


def fit_training(records: Sequence[Record]) -> tuple[pd.DataFrame, V11State]:
    """The training matrix (cross-fitted detector, out-of-fold forecast; ``cut`` and ``q_*``
    included) and the fitted state."""
    dex = Pokedex(records)
    train = parse(records, dex)
    fit_detector(train, [])
    p1, detector_columns = _collect(train, "p1")
    detector = _fit(cast(pd.DataFrame, p1[p1["final"] & p1["x_hp"].notna()]), detector_columns)
    helpers = Vocab(train), TeamModel(train), MoveTable(train), SwitchRates(train)
    X_train = feature_matrix(train, helpers)
    columns = list(X_train.columns)
    cut = float(len(records[0]["battle_timeline"]))
    X_train["cut"] = cut
    truncated = truncated_features(records, dex, helpers, CUTS, inputs(X_train.columns))
    Q_train, regressors = fit_forecast(train, X_train, truncated)
    X_train[list(Q_train.columns)] = Q_train.to_numpy()
    state = V11State(dex, detector, detector_columns, helpers, regressors, cut, columns)
    return X_train, state


def detect_new(state: V11State, battles: Sequence[Battle]) -> None:
    """Set ``p2_faint`` of new battles: the full detector on P2's candidates, 1 on recorded
    faints, 0 on the other P2 switches (the test branch of ``fit_detector``)."""
    p2, _ = _collect(battles, "p2")
    if p2.empty:
        return
    p2["p"] = 0.0
    cand = cast(pd.Series, (~p2["recorded"]) & p2["final"])
    if cand.any():
        p2.loc[cand, "p"] = state.detector.predict_proba(p2.loc[cand, state.detector_columns])[:, 1]
    p2.loc[p2["recorded"], "p"] = 1.0
    for b, i, p in zip(p2["b"], p2["i"], p2["p"], strict=True):
        battles[b].p2_faint[int(i)] = float(p)


def new_matrix(state: V11State, records: Sequence[Record]) -> pd.DataFrame:
    """The matrix of new battles, as the original's test matrix (columns of the training one)."""
    battles = parse(records, state.dex)
    detect_new(state, battles)
    X = cast(pd.DataFrame, feature_matrix(battles, state.helpers)[state.columns])
    X["cut"] = state.cut
    Q = predict_forecast(state.regressors, X)
    X[list(Q.columns)] = Q.to_numpy()
    return X
