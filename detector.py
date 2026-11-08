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
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import usecases
from addons.knockouts_model import INFO_FILE, load_artifact
from artifacts import TrainedRun
from data import load_records
from domain.damage import DamageCalculator
from domain.pokedex import Missing, Pokedex
from domain.records import fingerprint, without_label
from domain.transition_mechanics import FAINTED
from domain.transitions import Transition, TransitionExtractor
from features import transitions
from features.common import type_chart
from folds import FoldPlan
from gym import Gym
from interfaces import FeatureSource, LabelledBattles
from models import MODELS
from settings import Settings
from tuning.tuner import Tuner

ADDON = "knockouts"
KEEP_BATTLE = Missing()
Record = dict[str, Any]


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
