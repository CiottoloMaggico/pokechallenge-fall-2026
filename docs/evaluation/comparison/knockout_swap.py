"""Our ``compact_v5`` models with each team's hidden P2 knockouts in place of our detector's.

    python docs/evaluation/comparison/knockout_swap.py run --arm second --results results/comparison
    python docs/evaluation/comparison/knockout_swap.py tables --results results/comparison

The knockout repair is the one thing that changes between the arms: the feature set
(``compact_v5``), the models (``logreg`` and ``ensemble_from_lr`` with their default parameters,
as in the repository's regression check) and the folds (seeds 33, 66, 133) are the same.

``run``: one arm. ``none`` applies no repair; the others read a team's scores of P2's last exits
from ``<results>/knockouts/events.csv`` (``knockouts.py``; the names of the arms are its
teams): the score is the probability of the repair, and the exits the team counts as knockouts
are marked as fainted in the replay, in the whole battle and in its first 20 turns. Like our
detector's artifact, every team's scores are fitted once on the training file, without the
outcome. Writes the gym's tables to ``<results>/knockouts/swap/<arm>``.

``tables``: ``data/knockout_swap.csv``, the mean of every metric per arm and model and its
paired difference from the arm ``v3`` (corrected repeated k-fold t-test), and
``data/knockout_reverse.csv``, the opposite exchange: the 2nd place's models on our seed-33
folds with their own knockout scores (``second_place.py cv``, ``stage``) and with those of our
``v3`` (``second_place.py cv --knockouts``), and the paired difference.

The first comparison favours ``v3``, because the features of ``compact_v5`` were selected with
it; the second favours the 2nd place's own scores, for the same reason.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))

from data import Dataset  # noqa: E402
from domain.pokedex import Pokedex  # noqa: E402
from domain.repair import NO_REPAIR, TimelineRepair  # noqa: E402
from features import sets as feature_sets  # noqa: E402
from folds import FoldPlan  # noqa: E402
from gym import Gym, GymResult  # noqa: E402
from settings import DEFAULT_POKEDEX, DEFAULT_TRAIN  # noqa: E402
from stats_tables import corrected_ttest  # noqa: E402
from usecases import build_models  # noqa: E402

ARMS = ["none", "rule", "v3", "v11", "second", "spikes"]
REFERENCE = "v3"
FEATURES, MODELS, SEEDS = "compact_v5", ["logreg", "ensemble_from_lr"], (33, 66, 133)
METRICS = ["accuracy", "log_loss", "roc_auc"]
SCORES: dict[str, Any] = {"accuracy": lambda y, p: accuracy_score(y, p >= 0.5),
                          "log_loss": log_loss, "roc_auc": roc_auc_score}  # fmt: skip
SECOND = ["five_view", "team_similarity", "tactical", "tactical_team_spikes"]


class TableRepair:
    """``TimelineRepair`` that reads one team's scores of P2's last exits from a table.

    A record is found by its battle id and the length of its timeline (the training ids are
    unique), so the cut copies replayed by the features get the scores of their own cut."""

    def __init__(self, events: pd.DataFrame, team: str, battle_ids: list[int]) -> None:
        self._team = team
        self._scores: dict[tuple[int, int], dict[int, float]] = {}
        self._faints: dict[tuple[int, int], dict[int, str]] = {}
        groups: Any = events.groupby(["row", "turns"])
        for (row, turns), part in groups:
            key = (battle_ids[row], turns)
            counted = part[part[f"{team}_ko"] == 1]
            self._scores[key] = dict(zip(part["index"], part[team], strict=True))
            self._faints[key] = dict(zip(counted["index"], counted["leaver"], strict=True))

    def probabilities(self, record: Mapping[str, Any]) -> dict[int, float]:
        """Turn index -> the team's score of the P2 Pokémon that left at that row."""
        return dict(self._scores.get(self._key(record), {}))

    def repairs(self, record: Mapping[str, Any]) -> dict[int, str]:
        """Turn index -> name of the P2 Pokémon the team counts as fainted there."""
        return dict(self._faints.get(self._key(record), {}))

    def describe(self) -> dict[str, Any]:
        """Identify this repair in a run."""
        return {"name": "knockouts_table", "version": self._team, "params": {}}

    @staticmethod
    def _key(record: Mapping[str, Any]) -> tuple[int, int]:
        return record["battle_id"], len(record["battle_timeline"])


def repair_of(arm: str, results: Path) -> TimelineRepair:
    """The repair of an arm: none, or a team's column of ``events.csv``."""
    if arm == "none":
        return NO_REPAIR
    ids = [json.loads(line)["battle_id"] for line in DEFAULT_TRAIN.read_text().splitlines()]
    if len(set(ids)) != len(ids):
        raise SystemExit("the training battle ids are not unique")
    return TableRepair(pd.read_csv(results / "knockouts" / "events.csv"), arm, ids)


def run(args: argparse.Namespace) -> None:
    pokedex = Pokedex.from_json(DEFAULT_POKEDEX)
    data = Dataset.from_jsonl(DEFAULT_TRAIN, pokedex, repair=repair_of(args.arm, args.results))
    gym = Gym(build_models(MODELS, None), [feature_sets.build(FEATURES)], FoldPlan(5, SEEDS))
    result = gym.evaluate(data.labelled())
    result.export(args.results / "knockouts" / "swap" / args.arm)
    print(result.summary().to_string(index=False, float_format="{:.4f}".format))


def reverse_table(results: Path) -> pd.DataFrame:
    """The 2nd place's models with their knockout scores and with ours: fold means, difference
    (ours minus theirs) and corrected t-test on our seed-33 folds."""
    own, swapped = (
        pd.read_csv(results / "our_folds" / name / "second_oof.csv").sort_values("row")
        for name in ("fixed", "fixed_v3")
    )
    if not all(own[key].equals(swapped[key]) for key in ("row", "fold", "y_true")):
        raise SystemExit("the two runs of the 2nd place's models do not share the folds")
    folds = sorted(set(own["fold"]))
    rows = []
    for model in SECOND:
        for metric, score in SCORES.items():
            values = np.array([[float(score(t.loc[t["fold"] == k, "y_true"],
                                            t.loc[t["fold"] == k, model])) for k in folds]
                               for t in (own, swapped)])  # fmt: skip
            diff = values[1] - values[0]
            p = corrected_ttest(diff, 1 / (len(folds) - 1))[1]
            rows.append({"model": model, "metric": metric, "theirs": values[0].mean(),
                         "ours": values[1].mean(), "delta": diff.mean(), "p": p})  # fmt: skip
    return pd.DataFrame(rows)


def tables(args: argparse.Namespace) -> None:
    folds: Any = pd.concat(
        GymResult.load(args.results / "knockouts" / "swap" / arm).folds.assign(arm=arm)
        for arm in ARMS
    )
    ratio = float((folds["n_valid"] / folds["n_train"]).mean())
    wide = folds.set_index(["model", "seed", "fold", "arm"])[METRICS].unstack("arm")
    rows = []
    for model in MODELS:
        for metric in METRICS:
            values: Any = wide.loc[model, metric]
            for arm in ARMS:
                diff = (values[arm] - values[REFERENCE]).to_numpy()
                p = corrected_ttest(diff, ratio)[1] if arm != REFERENCE else float("nan")
                rows.append({"model": model, "arm": arm, "metric": metric,
                             "mean": values[arm].mean(), "delta": diff.mean(), "p": p})  # fmt: skip
    table = pd.DataFrame(rows)
    table.to_csv(args.out / "knockout_swap.csv", index=False)
    print(table.to_string(index=False, float_format="{:.4f}".format))
    reverse = reverse_table(args.results)
    reverse.to_csv(args.out / "knockout_reverse.csv", index=False)
    print(reverse.to_string(index=False, float_format="{:.4f}".format))


if __name__ == "__main__":
    root = argparse.ArgumentParser(description="compact_v5 with each team's hidden knockouts")
    commands = root.add_subparsers(dest="command", required=True)
    for name, handler in (("run", run), ("tables", tables)):
        sub = commands.add_parser(name)
        sub.add_argument("--results", type=Path, required=True, help="results/comparison")
        sub.set_defaults(handler=handler)
    commands.choices["run"].add_argument("--arm", choices=ARMS, required=True)
    commands.choices["tables"].add_argument("--out", type=Path, default=HERE / "data")
    arguments = root.parse_args()
    arguments.handler(arguments)
