"""The knockout detector as a task of the gym, and the artifact the repair addon loads.

Rows: P1's switches whose leaver never comes back (``TransitionExtractor.examples``), described
from the row before a recorded faint so that a P1 knockout looks like a hidden P2 one; the
label is the recorded faint, the group the battle's duplicate group (``fingerprint``), so a
battle and its copies stay in one fold. No outcome is read: the records lose ``player_won``.

The checks run on P2's candidate switches (``TransitionExtractor.candidates``, scored with the
stats each battle carries) with the model fitted on all P1 rows:

* ``p2_repairs`` against ``symmetry_estimate``: P1's recorded faints minus P2's (double
  knockouts cancel), times the share of P1's faints outside double knockouts that become rows;
* ``adversarial_auc``: out of fold (first seed), how well the model separates P1's rows from
  P2's candidates; 0.5 when the two sides look alike;
* ``agreement``: share of P2 candidates on which a reference artifact takes the same decision.

``detector train`` writes ``results/addons/knockouts/<version>/``: the ``TrainedRun`` files
and ``info.json`` with the settings, the hashes, the evaluation summary and the checks.

``detector train-v11`` writes v11's artifact, ``results/addons/knockouts_v11/<version>/``
(``train_v11``): rows are P1's described switches of ``domain.switches`` (``ko_v11``,
``lgbm_ko_v11``), grouped by duplicate group like ``v3``. Each training battle gets the mean over
the seeds of the out-of-fold probabilities of its P2 candidates (the folds of the standard
``FoldPlan``, one gym run per seed; a probe scores the candidates of the battles whose P1 rows are
in the validation part); a battle with no P1 row is in no validation part and gets the full
model's. The same tables are built on the training battles truncated at each cut of v11's
forecast; the model fitted on all P1 rows of the full battles is the artifact's ``TrainedRun``.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import usecases
from addons.knockouts_model import INFO_FILE, load_artifact
from addons.knockouts_v11 import THRESHOLD
from artifacts import TrainedRun
from data import load_records
from domain.damage import DamageCalculator
from domain.pokedex import Missing, Pokedex
from domain.records import fingerprint, without_label
from domain.switches import COLUMNS, Switch, SwitchExtractor
from domain.transition_mechanics import FAINTED
from domain.transitions import Transition, TransitionExtractor
from features import transitions
from features.common import type_chart
from features.core.v11.forecast import CUTS
from features.transitions import ko_v11
from folds import FoldPlan
from gym import Gym, GymResult
from interfaces import FeatureSource, FitContext, LabelledBattles
from models import MODELS, build
from settings import Settings
from tuning.tuner import Tuner

ADDON = "knockouts"
ADDON_V11 = "knockouts_v11"
MODEL_V11 = "lgbm_ko_v11"
V11_TABLES = "tables.pkl"
KEEP_BATTLE = Missing()
Record = dict[str, Any]
Cell = tuple[int, int]  # (battle position, timeline index)


@dataclass(frozen=True)
class DetectorTask:
    """P1's rows and P2's candidates of one training file, with the symmetry estimate."""

    p1: LabelledBattles
    p2: list[Transition]
    p2_groups: np.ndarray
    symmetry: float


def load_task(train: Path, pokedex: Pokedex) -> DetectorTask:
    """The detector's rows of ``train``; P1 is described with ``pokedex.find``, as it is
    fitted, P2 with each battle's own stats, as the repair scores it."""
    records = [without_label(r) for r in load_records(train)]
    damage = DamageCalculator(type_chart())
    fitting = TransitionExtractor(pokedex.find, damage)
    group_of: dict[str, int] = {}
    items: list[Transition] = []
    ids: list[str] = []
    groups: list[int] = []
    p2: list[Transition] = []
    p2_groups: list[int] = []
    for record in records:
        group = group_of.setdefault(fingerprint(record), len(group_of))
        for t in fitting.examples(record):
            items.append(t)
            ids.append(f"{record['battle_id']}:{t.index}")
            groups.append(group)
        scoring = TransitionExtractor(pokedex.lookup(record, KEEP_BATTLE), damage)
        candidates = scoring.candidates(record)
        p2.extend(candidates)
        p2_groups.extend([group] * len(candidates))
    labels = np.array([int(t.fainted) for t in items])
    p1 = LabelledBattles(items, labels, np.array(ids), np.array(groups))
    return DetectorTask(p1, p2, np.array(p2_groups), symmetry_estimate(records, int(labels.sum())))


def symmetry_estimate(records: Sequence[Record], described_faints: int) -> float:
    """Hidden P2 faints a detector can find, if both sides lose Pokémon alike."""
    p1 = p2 = doubles = 0
    for record in records:
        timeline = record["battle_timeline"]
        p1_rows, p2_rows = _first_faints(timeline, "p1"), _first_faints(timeline, "p2")
        p1, p2 = p1 + len(p1_rows), p2 + len(p2_rows)
        doubles += sum(
            timeline[row]["p2_pokemon_state"]["status"] == FAINTED for row in p1_rows.values()
        )
    return (p1 - p2) * described_faints / (p1 - doubles)


def _first_faints(timeline: Sequence[Record], side: str) -> dict[str, int]:
    """Name -> first row where that Pokémon of ``side`` is recorded as fainted."""
    rows: dict[str, int] = {}
    for i, turn in enumerate(timeline):
        state = turn[f"{side}_pokemon_state"]
        if state["status"] == FAINTED:
            rows.setdefault(state["name"], i)
    return rows


def checks(
    run: TrainedRun,
    task: DetectorTask,
    threshold: float,
    plan: FoldPlan,
    reference: Path | None = None,
) -> dict[str, float]:
    """The checks of the module docstring for a detector fitted on all P1 rows."""
    decided = run.predict_proba(task.p2) >= threshold
    found = {
        "p2_repairs": int(decided.sum()),
        "symmetry_estimate": task.symmetry,
        "adversarial_auc": adversarial_auc(run, task, plan),
    }
    if reference is not None:
        other, info = load_artifact(reference)
        agree = decided == (other.predict_proba(task.p2) >= float(info["threshold"]))
        found["agreement"] = float(agree.mean())
    return found


def adversarial_auc(run: TrainedRun, task: DetectorTask, plan: FoldPlan) -> float:
    """Out-of-fold AUC of the run's model and features telling P2's candidates from P1's rows."""
    n_p1, n_p2 = len(task.p1.items), len(task.p2)
    sides = LabelledBattles(
        [*task.p1.items, *task.p2],
        np.repeat([0, 1], [n_p1, n_p2]),
        np.arange(n_p1 + n_p2),
        np.concatenate([task.p1.groups, task.p2_groups]),
    )
    first_seed = FoldPlan(plan.n_folds, plan.seeds[:1])
    result = Gym([run.model.clone()], [run.source], first_seed).evaluate(sides)
    return float(np.mean(result.folds["roc_auc"].to_numpy()))


def evaluate(
    settings: Settings,
    features: Sequence[str],
    models: Sequence[str],
    params: Sequence[Path] | None = None,
    reference: str | None = None,
    threshold: float = 0.5,
    agreement: Path | None = None,
    out: Path | None = None,
) -> int:
    """Cross-validate detectors on P1's rows (the tables of ``evaluate``) and add the checks."""
    task = load_task(settings.train, Pokedex.from_json(settings.pokedex))
    print(f"detector rows: {len(task.p1.items)} P1 switches, {len(task.p2)} P2 candidates")
    gym = _gym(settings, [transitions.build(name) for name in features], models, params)
    result = gym.evaluate(task.p1)
    target = out or settings.results.evaluation(time.strftime("detector_%Y-%m-%d_%H%M%S"))
    usecases.report(result, reference, False, target)
    rows = [
        {"feature_set": run.feature_set, "model": run.model_name}
        | checks(run, task, threshold, settings.plan, agreement)
        for run in gym.train(task.p1)
    ]
    table = pd.DataFrame(rows)
    table.to_csv(target / "checks.csv", index=False)
    print(f"\nchecks on P2 (threshold {threshold}):")
    print(table.to_string(index=False, float_format="{:.4f}".format))
    return 0


def train(
    settings: Settings,
    feature_set: str,
    model: str,
    version: str,
    threshold: float = 0.5,
    params: Sequence[Path] | None = None,
    agreement: Path | None = None,
) -> int:
    """Evaluate one detector, fit it on all P1 rows and save it as a new artifact version."""
    target = settings.results.addon(ADDON, version)
    if target.exists():
        raise SystemExit(f"{target} exists: an artifact version is never overwritten")
    pokedex = Pokedex.from_json(settings.pokedex)
    task = load_task(settings.train, pokedex)
    gym = _gym(settings, [transitions.build(feature_set)], [model], params)
    summary = gym.evaluate(task.p1).summary()
    print(summary.to_string(index=False, float_format="{:.4f}".format))
    (run,) = gym.train(task.p1)
    found = checks(run, task, threshold, settings.plan, agreement)
    print(json.dumps(found, indent=2))
    run.save(target)
    info = {
        "version": version,
        "feature_set": feature_set,
        "model": run.model_name,
        "params": _jsonable(run.model.get_params()),
        "threshold": threshold,
        "train_sha256": _sha256(settings.train),
        "pokedex_sha256": pokedex.source_sha256,
        "seeds": list(settings.plan.seeds),
        "evaluation": json.loads(summary.iloc[[0]].to_json(orient="records"))[0],
        "checks": found,
        "saved": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (target / INFO_FILE).write_text(json.dumps(info, indent=2) + "\n")
    print(f"detector saved to {target}")
    return 0


@dataclass(frozen=True)
class SwitchTask:
    """P1's described switches of some battles (the gym's rows, ``battle`` of each row) and every
    P2 switch of the same battles (``p2``, per battle)."""

    p1: LabelledBattles
    battle: np.ndarray
    p2: list[list[Switch]]

    @property
    def candidates(self) -> list[tuple[int, Switch]]:
        """(battle, switch) of the P2 switches that may hide a knockout."""
        return [
            (k, s) for k, rows in enumerate(self.p2) for s in rows if s.final and not s.recorded
        ]


def switch_task(records: Sequence[Record], groups: Sequence[int], pokedex: Pokedex) -> SwitchTask:
    """The rows of v11's detector; each battle is described with its own species."""
    chart = type_chart()
    items: list[Switch] = []
    ids: list[str] = []
    battle: list[int] = []
    p2: list[list[Switch]] = []
    for k, record in enumerate(records):
        extractor = SwitchExtractor(pokedex.lookup(record, KEEP_BATTLE), chart)
        for switch in extractor.examples(record):
            items.append(switch)
            ids.append(f"{record['battle_id']}:{switch.index}")
            battle.append(k)
        p2.append(extractor.extract(record, "p2"))
    labels = np.array([s.label for s in items])
    p1 = LabelledBattles(items, labels, np.array(ids), np.array([groups[k] for k in battle]))
    return SwitchTask(p1, np.array(battle), p2)


def cross_fit(task: SwitchTask, plan: FoldPlan) -> tuple[dict[Cell, float], GymResult, TrainedRun]:
    """P(faint) of every candidate, by (battle, index): the mean over the seeds of its out-of-fold
    probability, or the full model's for a battle with no P1 row; the out-of-fold results of the
    P1 rows and the run fitted on all of them."""
    features = ko_v11.build()
    candidates = task.candidates
    table = features.to_frame([s for _, s in candidates])
    battle_of = np.array([k for k, _ in candidates])

    def probe(fit: FitContext) -> list[dict[str, Any]]:
        """The candidates of the battles whose P1 rows are in this fold's validation part (the
        table of a stateless set keeps the task's row positions as index)."""
        rows = np.flatnonzero(np.isin(battle_of, task.battle[fit.x_valid.index.to_numpy()]))
        p = fit.model.predict_proba(table.iloc[rows]) if rows.size else []
        return [
            {"battle": candidates[j][0], "i": candidates[j][1].index, "p": float(v)}
            for j, v in zip(rows, p, strict=True)
        ]

    by_seed: dict[int, GymResult] = {}
    found: dict[Cell, list[float]] = {}
    for seed in plan.seeds:
        plan_seed = FoldPlan(plan.n_folds, (seed,))
        gym = Gym([build(MODEL_V11)], [features], plan_seed, probes={"p2": probe})
        by_seed[seed] = gym.evaluate(task.p1)
        for k, i, p in by_seed[seed].probes["p2"][["battle", "i", "p"]].itertuples(index=False):
            found.setdefault((int(k), int(i)), []).append(float(p))
    (run,) = Gym([build(MODEL_V11)], [features]).train(task.p1)
    alone = [j for j, (k, s) in enumerate(candidates) if (k, s.index) not in found]
    full = run.model.predict_proba(table.iloc[alone]) if alone else []
    values = {cell: float(np.mean(p)) for cell, p in found.items()}
    values.update(
        {
            (candidates[j][0], candidates[j][1].index): float(v)
            for j, v in zip(alone, full, strict=True)
        }
    )
    return values, GymResult.combine(by_seed), run


def faint_tables(
    records: Sequence[Record], task: SwitchTask, values: Mapping[Cell, float]
) -> dict[tuple[Any, str], dict[int, float]]:
    """``(battle id, fingerprint)`` -> every P2 switch: 1.0 recorded, P(faint) candidate, else 0."""
    return {
        (record["battle_id"], fingerprint(record)): {
            s.index: 1.0 if s.recorded else values.get((k, s.index), 0.0) for s in task.p2[k]
        }
        for k, record in enumerate(records)
    }


def pooled_auc(result: GymResult) -> float:
    """Out-of-fold ROC AUC on all P1 rows, mean over the seeds."""
    by_seed = result.predictions.groupby("seed")
    return float(np.mean([roc_auc_score(g["y_true"], g["proba"]) for _, g in by_seed]))


def train_v11(settings: Settings, version: str) -> int:
    """Build v11's detector artifact (see the module docstring); never overwritten."""
    target = settings.results.addon(ADDON_V11, version)
    if target.exists():
        raise SystemExit(f"{target} exists: an artifact version is never overwritten")
    pokedex = Pokedex.from_json(settings.pokedex)
    records = [without_label(r) for r in load_records(settings.train)]
    group_of: dict[str, int] = {}
    groups = [group_of.setdefault(fingerprint(r), len(group_of)) for r in records]
    task = switch_task(records, groups, pokedex)
    values, result, run = cross_fit(task, settings.plan)
    tables: dict[str, Any] = {"full": faint_tables(records, task, values), "cuts": {}}
    cut_auc: dict[int, float] = {}
    for cut in CUTS:
        cut_records = [{**r, "battle_timeline": r["battle_timeline"][:cut]} for r in records]
        cut_task = switch_task(cut_records, groups, pokedex)
        cut_values, cut_result, _ = cross_fit(cut_task, settings.plan)
        tables["cuts"][cut] = faint_tables(cut_records, cut_task, cut_values)
        cut_auc[cut] = pooled_auc(cut_result)
        print(f"cut {cut}: AUC {cut_auc[cut]:.4f}", flush=True)
    run.save(target)
    (target / V11_TABLES).write_bytes(pickle.dumps(tables))
    summary = result.summary()
    info = {
        "version": version,
        "feature_set": ko_v11.NAME,
        "model": run.model_name,
        "params": _jsonable(run.model.get_params()),
        "columns": list(COLUMNS),
        "cuts": list(CUTS),
        "seeds": list(settings.plan.seeds),
        "threshold": THRESHOLD,
        "train_sha256": _sha256(settings.train),
        "pokedex_sha256": pokedex.source_sha256,
        "detector_auc": pooled_auc(result),
        "detector_pos_rate": float(task.p1.labels.mean()),
        "p2_hidden_per_battle": float(sum(values.values()) / len(records)),
        "battles_without_p1_rows": len(records) - len(set(task.battle.tolist())),
        "cut_auc": cut_auc,
        "evaluation": json.loads(summary.iloc[[0]].to_json(orient="records"))[0],
        "saved": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (target / INFO_FILE).write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps({k: info[k] for k in ("detector_auc", "p2_hidden_per_battle")}, indent=2))
    print(f"v11 detector saved to {target}")
    return 0


def tune(
    settings: Settings,
    feature_set: str,
    model: str,
    trials: int,
    timeout: float | None = None,
    metric: str = "log_loss",
    study: str | None = None,
    storage_dir: Path | None = None,
    out: Path | None = None,
) -> int:
    """Search the hyperparameters of one detector model on the first seed of the plan."""
    source = transitions.build(feature_set)
    task = load_task(settings.train, Pokedex.from_json(settings.pokedex))
    seed = settings.plan.seeds[0]
    tables = Gym.fold_tables(task.p1, source, FoldPlan(settings.plan.n_folds, (seed,)))
    tuner = Tuner(MODELS[model], tables[seed], metric=metric)
    name = study or f"detector_{source.name}__{model}"
    usecases.run_study(settings, tuner, source.name, name, trials, timeout, storage_dir, out)
    return 0


def _gym(
    settings: Settings,
    sources: Sequence[FeatureSource],
    models: Sequence[str],
    params: Sequence[Path] | None,
) -> Gym:
    jobs = usecases.fork_safe_jobs(settings.jobs, [], models)
    return Gym(usecases.build_models(models, params), sources, settings.plan, jobs=jobs)


def _jsonable(values: Mapping[str, Any]) -> dict[str, Any]:
    """``values`` as JSON reads them back, so a manifest compares equal after a save."""
    return json.loads(json.dumps(values, default=str))


def _sha256(path: Path) -> str:
    with path.open("rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()
