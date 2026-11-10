"""``detector train-v11``: the gym's task of v11's detector and what its probe sees."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import detector
from domain.pokedex import PokedexBuilder
from domain.records import fingerprint, read_records, without_label
from features.transitions import ko_v11
from folds import FoldPlan, make_splits
from gym import Gym
from interfaces import FitContext
from models import build
from settings import Settings
from storage import ResultsLayout

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "battles.jsonl"


@pytest.fixture(scope="module")
def task() -> detector.SwitchTask:
    records = [without_label(r) for r in read_records(FIXTURES)]
    pokedex = PokedexBuilder().add_records(records).build()
    group_of: dict[str, int] = {}
    groups = [group_of.setdefault(fingerprint(r), len(group_of)) for r in records]
    return detector.switch_task(records, groups, pokedex)


def test_the_task_has_p1_rows_grouped_by_battle(task: detector.SwitchTask) -> None:
    rows = task.p1
    assert all(s.final and s.features for s in rows.items)
    assert rows.labels.tolist() == [int(s.recorded) for s in rows.items]
    assert (np.diff(task.battle) >= 0).all(), "battles in file order, rows in order"
    for k in np.unique(task.battle):
        assert len(set(rows.groups[task.battle == k].tolist())) == 1
    assert all(s.final and not s.recorded for _, s in task.candidates)


def test_the_probe_sees_the_task_positions_of_the_validation_rows(
    task: detector.SwitchTask,
) -> None:
    seen: list[np.ndarray] = []

    def probe(fit: FitContext) -> dict[str, int]:
        seen.append(fit.x_valid.index.to_numpy())
        return {"rows": len(fit.x_valid)}

    plan = FoldPlan(5, (7,))
    model = build("lgbm_ko_v11", n_estimators=5)
    Gym([model], [ko_v11.build()], plan, probes={"p": probe}).evaluate(task.p1)
    (splitter,) = (s for _, s in plan.splitters())
    splits = make_splits(splitter, task.p1.labels, task.p1.groups)
    assert [v.tolist() for v in seen] == [valid.tolist() for _, valid in splits]


def test_an_artifact_version_is_never_overwritten(tmp_path: Path) -> None:
    (tmp_path / "addons" / "knockouts_v11" / "v1").mkdir(parents=True)
    settings = Settings(FIXTURES, results=ResultsLayout(tmp_path))
    with pytest.raises(SystemExit, match="never overwritten"):
        detector.train_v11(settings, "v1")
