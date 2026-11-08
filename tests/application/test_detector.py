"""The detector task, its checks and its artifact, on the fixtures."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import detector
import main
import usecases
from addons.knockouts_model import ModelKnockoutRepair, load_artifact
from domain.damage import DamageCalculator
from domain.pokedex import Pokedex, PokedexBuilder
from domain.records import read_records
from domain.transitions import TransitionExtractor
from features.common import type_chart
from folds import FoldPlan
from settings import Settings
from storage import ResultsLayout

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "battles.jsonl"
SMALL = {"model": "hgb_ko", "params": {"max_iter": 30}}


@pytest.fixture(scope="module")
def settings(tmp_path_factory: pytest.TempPathFactory) -> Settings:
    folder = tmp_path_factory.mktemp("detector")
    PokedexBuilder().add_records(read_records(FIXTURES)).build("fixtures").to_json(
        folder / "species.json"
    )
    (folder / "small.json").write_text(json.dumps(SMALL))
    return Settings(
        train=FIXTURES,
        test=FIXTURES,
        pokedex=folder / "species.json",
        plan=FoldPlan(2, (1,)),
        results=ResultsLayout(folder / "results"),
    )


@pytest.fixture(scope="module")
def artifact(settings: Settings) -> Path:
    params = [settings.pokedex.parent / "small.json"]
    assert detector.train(settings, "ko_domain30", "hgb_ko", "v1", params=params) == 0
    return settings.results.addon("knockouts", "v1")


def test_the_rows_are_p1_switches_grouped_by_battle(settings: Settings) -> None:
    pokedex = Pokedex.from_json(settings.pokedex)
    task = detector.load_task(FIXTURES, pokedex)
    extractor = TransitionExtractor(pokedex.find, DamageCalculator(type_chart()))
    records = list(read_records(FIXTURES))
    assert len(task.p1.items) == sum(len(extractor.examples(r)) for r in records)
    assert set(task.p1.labels) == {0, 1}
    assert len(set(task.p1.groups)) <= len(records)
    assert len(task.p2_groups) == len(task.p2) > 0


def turn(p1: tuple[str, str], p2: tuple[str, str]) -> dict[str, Any]:
    def state(name: str, status: str) -> dict[str, Any]:
        return {"name": name, "status": status}

    return {"p1_pokemon_state": state(*p1), "p2_pokemon_state": state(*p2)}


def test_the_symmetry_estimate() -> None:
    timeline = [
        turn(("a", "nostatus"), ("x", "nostatus")),
        turn(("a", "fnt"), ("x", "nostatus")),  # P1 faint, recorded
        turn(("b", "fnt"), ("x", "fnt")),  # double knockout: never detectable
        turn(("c", "fnt"), ("y", "nostatus")),
        turn(("c", "fnt"), ("y", "nostatus")),  # the same faint again
    ]
    # 3 P1 faints, 1 P2 faint, 1 double: (3 - 1) * described / (3 - 1)
    assert detector.symmetry_estimate([{"battle_timeline": timeline}], 1) == 1.0


def test_train_writes_an_artifact_the_addon_loads(settings: Settings, artifact: Path) -> None:
    assert {p.name for p in artifact.iterdir()} == {"run.json", "model.pkl", "info.json"}
    run, info = load_artifact(artifact)
    assert info["version"] == "v1"
    assert info["threshold"] == 0.5
    assert info["params"]["max_iter"] == 30
    assert info["pokedex_sha256"] == "fixtures"
    assert set(info["checks"]) == {"p2_repairs", "symmetry_estimate", "adversarial_auc"}
    assert 0 <= info["evaluation"]["roc_auc_mean"] <= 1
    pokedex = Pokedex.from_json(settings.pokedex)
    repair = ModelKnockoutRepair(run, info, pokedex, DamageCalculator(type_chart()))
    repaired = sum(len(repair.repairs(r)) for r in read_records(FIXTURES))
    assert repaired == info["checks"]["p2_repairs"]
    assert repair.describe()["version"] == "v1"


def test_a_version_is_never_overwritten(settings: Settings, artifact: Path) -> None:
    with pytest.raises(SystemExit, match="never overwritten"):
        detector.train(settings, "ko_domain30", "hgb_ko", "v1")


def test_training_again_gives_the_same_repairs(settings: Settings, artifact: Path) -> None:
    params = [settings.pokedex.parent / "small.json"]
    detector.train(settings, "ko_domain30", "hgb_ko", "v1-again", params=params)
    first = json.loads((artifact / "info.json").read_text())["checks"]
    again = settings.results.addon("knockouts", "v1-again") / "info.json"
    assert json.loads(again.read_text())["checks"] == first


def test_evaluate_exports_the_tables_and_the_checks(
    settings: Settings, artifact: Path, tmp_path: Path
) -> None:
    code = detector.evaluate(
        settings,
        ["ko_domain30", "ko_raw50"],
        ["logreg"],
        reference="ko_raw50/logreg",
        agreement=artifact,
        out=tmp_path,
    )
    assert code == 0
    checks = pd.read_csv(tmp_path / "checks.csv")
    assert list(checks["feature_set"]) == ["ko_domain30", "ko_raw50"]
    assert checks["agreement"].between(0, 1).all()
    assert (tmp_path / "paired.csv").exists()


def test_main_runs_with_the_artifact_and_refuses_another_repair(
    settings: Settings, artifact: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with_model = replace(settings, repair=f"model={artifact}")
    assert usecases.train(with_model, ["compact_v5"], ["logreg"]) == 0
    run = settings.results.run("compact_v5", "logreg")
    assert json.loads((run / "run.json").read_text())["repair"]["version"] == "v1"
    assert usecases.predict(with_model, run) == 0
    line = ["--pokedex", str(settings.pokedex), "predict", "--run", str(run), "--test"]
    assert main.main([*line, str(FIXTURES), "--repair", "rule"]) == 2
    assert "trained with repair" in capsys.readouterr().err
