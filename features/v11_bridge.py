"""The v11 model (FDS-Pokemon-Battle-Prediction repository) as a feature.

``V11Probability`` outputs ``v11_proba``: the probability that P1 wins given
by a v11 blend (default ``blend_v11`` of ``configs/blends.json``) trained on
the feature's own training battles only. It lets a model of this repository
be blended with v11 inside ``main.py evaluate / train / predict``, with the
gym's rules: in every fold v11 is rebuilt from scratch on the training part
(its Pokédex, hidden-faint detector, helpers, forecasts, stacked logits and
team effects included) and only predicts the validation battles; for the
submission it is trained on the whole training set and predicts the test.

Training battles get NaN (``inner_oof = False``): the feature never gives
in-sample values, so models must not use it as an ordinary column; the
blend (``models/logit_blend.py``) only reads it on predicted battles.

Setup: the v11 repository next to this one (or ``V11_REPO=/path``) with its
requirements installed (lightgbm, xgboost, catboost). v11 reads its data
folder at import: its own ``fds-pokemon-battles-prediction-2026/`` if
present, else ``FDS_DATA_DIR`` if set, else ``results/v11_data/`` (links to
``input/*.jsonl`` and ``assets/type_chart_gen2.csv``). Only the battles
passed to the feature are used for fitting: v11's own train / test loaders
are not called. Threads per v11 model: ``FDS_N_JOBS``.

Cost: by default (``V11_FEATURES=fold``, exact) one complete v11 build and
fit per call of ``prepare`` with new battles, i.e. once per fold and once for
the submission: v11 recomputes its features on 7 truncations of every
training battle for its forecasts, so a fold takes a long time.
``V11_FEATURES=shared`` follows v11's own CV protocol instead: the label-free
steps are fitted once on the whole training file (cached in
``results/v11_cache/shared_features.pkl``) and each fold only refits the
members (see ``_shared_rows``). For the submission the two modes coincide
(train fitted on all of train, test only predicted).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from domain.battle import BattleContext
from feature_set import Feature

DEFAULT_REPO = Path(__file__).resolve().parents[2] / "FDS-Pokemon-Battle-Prediction-main"
DEFAULT_DATA = Path(__file__).resolve().parents[1] / "input"
OUTPUT = "v11_proba"


def configure() -> Path:
    """Make the v11 package (``src``) importable; return its root."""
    repo = Path(os.environ.get("V11_REPO", DEFAULT_REPO)).resolve()
    if not (repo / "src" / "cli.py").exists():
        raise FileNotFoundError(
            f"v11 repository not found at {repo}; set V11_REPO to its folder"
        )
    if "FDS_DATA_DIR" not in os.environ and not (
        repo / "fds-pokemon-battles-prediction-2026" / "train.jsonl"
    ).exists():
        os.environ["FDS_DATA_DIR"] = str(_data_folder())
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    return repo


def _data_folder() -> Path:
    """A folder laid out as v11 expects (train.jsonl, test.jsonl and
    type_chart_gen2.csv), linking this repository's files."""
    root = Path(__file__).resolve().parents[1]
    folder = root / "results" / "v11_data"
    folder.mkdir(parents=True, exist_ok=True)
    for name, source in (
        ("train.jsonl", DEFAULT_DATA / "train.jsonl"),
        ("test.jsonl", DEFAULT_DATA / "test.jsonl"),
        ("type_chart_gen2.csv", root / "assets" / "type_chart_gen2.csv"),
    ):
        link = folder / name
        if source.exists() and not link.exists():
            try:
                link.symlink_to(source.resolve())
            except OSError:  # no symlinks (Windows without rights): copy
                link.write_bytes(source.read_bytes())
    return folder


def fingerprint(record: Mapping[str, Any]) -> str:
    """Identity of a battle independent of its id and label (train and test
    ids overlap): hash of the teams, the lead and the timeline."""
    content = {k: v for k, v in record.items() if k not in ("battle_id", "player_won")}
    return hashlib.sha1(json.dumps(content, sort_keys=True).encode()).hexdigest()


def v11_probabilities(
    train_records: Sequence[Mapping[str, Any]],
    labels: Sequence[int],
    apply_records: Sequence[Mapping[str, Any]],
    blend: str = "blend_v11",
    verbose: bool = True,
) -> np.ndarray:
    """P(P1 wins) of the v11 blend fitted on ``train_records`` only, for
    ``apply_records``. Same steps as v11's ``features.build.build`` and
    ``cli.predict_test`` with the given battles in place of train / test."""
    if os.environ.get("V11_FAKE") == "1":
        return _fake_probabilities(train_records, labels, apply_records)
    configure()
    started = time.perf_counter()
    mode = os.environ.get("V11_FEATURES", "fold")
    x_train = x_other = None
    if mode == "shared":
        x_train, x_other = _shared_rows(train_records, apply_records)
    if x_train is None:
        x_train, x_other = _v11_matrices(train_records, apply_records, verbose)
    p = _fit_members(x_train, labels, x_other, blend)
    if verbose:
        print(f"    v11 {blend} ({mode} features) fitted on {len(x_train)}, predicted "
              f"{len(x_other)} ({time.perf_counter() - started:.0f} s)", flush=True)
    return p


def _v11_matrices(train_records, apply_records, verbose=True):
    """v11's feature tables: everything fitted on ``train_records`` only."""
    import pandas as pd
    from src.data import Pokedex
    from src.faint import fit_detector
    from src.features import forecast
    from src.features.bags import Vocab
    from src.features.build import battle_features, truncated_features
    from src.features.hidden import TeamModel
    from src.features.simulation import MoveTable
    from src.features.switching import SwitchRates
    from src.replay import parse

    started = time.perf_counter()
    # v11 indexes rows by battle_id: give both sets unique ids of their own
    train_raw = [{**r, "battle_id": i} for i, r in enumerate(train_records)]
    apply_raw = [{**r, "battle_id": i} for i, r in enumerate(apply_records)]
    dex = Pokedex(train_raw)  # built from the training battles only
    train, other = parse(train_raw, dex), parse(apply_raw, dex)
    fit_detector(train, other)  # fitted on train, applied to the others
    helpers = Vocab(train), TeamModel(train), MoveTable(train), SwitchRates(train)
    x_train = pd.DataFrame([battle_features(b, *helpers) for b in train]).set_index("battle_id")
    x_other = pd.DataFrame([battle_features(b, *helpers) for b in other]).set_index("battle_id")
    x_train = x_train.astype(float)
    x_other = x_other.reindex(columns=x_train.columns).astype(float)
    x_train["cut"] = x_other["cut"] = float(len(train_raw[0]["battle_timeline"]))
    truncated = truncated_features(
        train_raw, dex, helpers, forecast.CUTS, forecast.inputs(x_train.columns)
    )
    q_train, q_other = forecast.forecast_features(train, x_train, x_other, truncated)
    x_train, x_other = x_train.join(q_train), x_other.join(q_other)
    if verbose:
        print(f"    v11 features {x_train.shape} / {x_other.shape} "
              f"({time.perf_counter() - started:.0f} s)", flush=True)
    return x_train, x_other


def _fit_members(x_train, labels, x_other, blend):
    """Fit the members of ``blend`` on ``x_train`` and blend their logits."""
    from src import ensemble, pipelines
    from src.models import fit_weighted

    repo = configure()
    spec = json.loads((repo / "configs" / "blends.json").read_text())[blend]
    members, weights = list(spec["members"]), np.array(list(spec["members"].values()))
    all_specs = pipelines.specs()
    if any(all_specs[m].get("aug") for m in members):
        raise ValueError("members with 'aug' read v11's global cache: not supported")
    y = np.asarray(labels, dtype=int)
    probs = []
    for name in members:
        member = all_specs[name]
        cols = pipelines.columns(member, x_train.columns)
        xf, yf, wf = pipelines.training_set(member, x_train[cols], y)
        model = fit_weighted(pipelines.factory(member, cols)(), xf, yf, wf)
        probs.append(model.predict_proba(x_other[cols])[:, 1])
    return np.asarray(ensemble.blend(np.stack(probs), weights), dtype=float)


_SHARED_TABLES: dict[str, Any] = {}


def _shared_rows(train_records, apply_records):
    """``V11_FEATURES=shared``: v11's own CV protocol. Its label-free steps
    (Pokédex, detector, helpers, forecasts) are fitted once on the whole
    training file and its features computed for train and test; each call
    then only refits the members (label-aware, cross-fitted inside) on the
    given training rows. Faster by hours, but in cross-validation the
    validation battles take part in those label-free fits (never their
    labels), as in v11's reported CV. Returns (None, None) when some battle
    is not in the tables (then the exact per-call build is used)."""
    if not _SHARED_TABLES:
        import pandas as pd

        root = Path(__file__).resolve().parents[1]
        path = root / "results" / "v11_cache" / "shared_features.pkl"
        if path.exists():
            _SHARED_TABLES.update(pd.read_pickle(path))
        else:
            train = [json.loads(line) for line in (DEFAULT_DATA / "train.jsonl").open()]
            test_path = DEFAULT_DATA / "test.jsonl"
            test = [json.loads(line) for line in test_path.open()] if test_path.exists() else []
            x_train, x_test = _v11_matrices(train, test or train[:1])
            x_train.index = [fingerprint(r) for r in train]
            x_test.index = [fingerprint(r) for r in (test or train[:1])]
            table = pd.concat([x_train, x_test[~x_test.index.isin(x_train.index)]])
            table = table[~table.index.duplicated()]
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.to_pickle({"table": table}, path)
            _SHARED_TABLES["table"] = table
    table = _SHARED_TABLES["table"]
    keys_train = [fingerprint(r) for r in train_records]
    keys_other = [fingerprint(r) for r in apply_records]
    if not set(keys_train) <= set(table.index) or not set(keys_other) <= set(table.index):
        return None, None
    return table.loc[keys_train], table.loc[keys_other]


def _fake_probabilities(train_records, labels, apply_records) -> np.ndarray:
    """Test stand-in for v11 (``V11_FAKE=1``): logistic regression on the last
    row's HP of both actives. Only for checking the plumbing without v11."""
    from sklearn.linear_model import LogisticRegression

    def row(r):
        last = r["battle_timeline"][-1]
        return [last["p1_pokemon_state"]["hp_pct"], last["p2_pokemon_state"]["hp_pct"]]

    model = LogisticRegression().fit([row(r) for r in train_records], labels)
    return model.predict_proba([row(r) for r in apply_records])[:, 1]


class V11Probability(Feature[BattleContext]):
    """``v11_proba`` (see the module docstring)."""

    requires_fit = True
    uses_labels = True
    inner_oof = False  # NaN on its own training battles: no inner folds needed

    def __init__(self, blend: str = "blend_v11") -> None:
        super().__init__("v11_probability")
        self.blend = blend
        self._records: list[dict[str, Any]] = []
        self._labels: list[int] = []
        self._train_keys: set[str] = set()
        self._cache: dict[str, float] = {}

    @property
    def output_names(self) -> tuple[str, ...]:
        return (OUTPUT,)

    def _fit(self, battles: Sequence[BattleContext], labels: Sequence[int] | None) -> None:
        if labels is None:
            raise ValueError("v11_probability needs the labels")
        self._records = [dict(b.record) for b in battles]
        self._labels = [int(v) for v in labels]
        self._train_keys = {fingerprint(r) for r in self._records}
        self._cache = {}

    def reset(self) -> None:
        super().reset()
        self._records, self._labels, self._train_keys, self._cache = [], [], set(), {}

    def prepare(self, battles: Sequence[BattleContext]) -> None:
        """Run v11 once on every battle not seen yet (training battles excluded)."""
        if not self._records:
            return
        todo: dict[str, dict[str, Any]] = {}
        for b in battles:
            key = fingerprint(b.record)
            if key not in self._train_keys and key not in self._cache:
                todo.setdefault(key, dict(b.record))
        if not todo:
            return
        p = v11_probabilities(self._records, self._labels, list(todo.values()), self.blend)
        self._cache.update(zip(todo, map(float, p), strict=True))

    def compute(self, battle: BattleContext) -> Mapping[str, float]:
        key = fingerprint(battle.record)
        if key in self._train_keys:
            return {OUTPUT: math.nan}
        if key not in self._cache:
            self.prepare([battle])
        return {OUTPUT: self._cache[key]}
