"""Layout of the outputs, the options of each command, and the commands on the fixtures."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

import main
import usecases
from artifacts import RepairMismatchError
from domain.errors import DomainError
from domain.pokedex import PokedexBuilder
from domain.records import read_records
from gym import FoldPlan
from registry import fork_unsafe_libraries
from settings import Settings
from storage import ResultsLayout

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "battles.jsonl"


def test_every_output_is_under_the_results_root() -> None:
    layout = ResultsLayout(Path("r"))
    assert layout.run("s", "m") == Path("r/runs/s__m")
    assert layout.evaluation("e") == Path("r/evaluations/e")
    assert layout.tuning() == Path("r/tuning")
    assert layout.cache("folds", "x.pkl") == Path("r/cache/folds/x.pkl")
    assert layout.submission("sub") == Path("r/submissions/sub.csv")


@pytest.mark.parametrize(
    ("line", "seeds"),
    [
        ("--seeds 33 66 33", (33, 66)),
        ("", (42,)),
        ("--n-seeds 3", (42, 43, 44)),
        ("--seed 7 --n-seeds 2", (7, 8)),
    ],
)
def test_the_seeds_of_evaluate(line: str, seeds: tuple[int, ...]) -> None:
    args = main.build_parser().parse_args(
        ["evaluate", "--features", "compact_v5", "--models", "logreg", *line.split()]
    )
    assert main._seeds(args) == seeds


def test_random_seeds_are_drawn() -> None:
    args = main.build_parser().parse_args(
        ["evaluate", "--features", "hp30", "--models", "logreg", "--n-seeds", "2", "--random-seeds"]
    )
    assert len(set(main._seeds(args))) == 2


def test_no_seeds_is_refused() -> None:
    args = main.build_parser().parse_args(
        ["evaluate", "--features", "hp30", "--models", "logreg", "--n-seeds", "0"]
    )
    with pytest.raises(SystemExit):
        main._seeds(args)


@pytest.mark.parametrize(
    "line",
    [
        "check --train t --test u",
        "evaluate --train t --test u --features hp30 --models logreg hgb --turns 5 10 "
        "--params p --folds 3 --seed 1 --n-seeds 2 --seeds 1 --random-seeds --jobs 2 "
        "--reference hp30/logreg --out o",
        "train --train t --test u --features hp30 --models logreg --turns 5 --params p --out o",
        "tune --train t --test u --features hp30 --model hgb --trials 1 --timeout 1 "
        "--metric accuracy --folds 3 --seed 1 --study s --storage-dir d --out o --refresh",
        "predict --run r --test u --out o --proba p --threshold 0.4",
        "predict --run r --repair none",
        "evaluate --features hp30 --models logreg --repair model=results/addons/knockouts/v1",
        "replay --battle 3 --file f --repair rule",
        "detector evaluate --features ko_domain30 ko_raw50 --models hgb_ko logreg --seeds 42 1 2 "
        "--reference ko_raw50/hgb_ko --threshold 0.4 --agreement a --params p --jobs 2 --out o",
        "detector train --features ko_raw50 --models hgb_ko --version v9 --threshold 0.6 "
        "--n-seeds 2 --agreement a",
        "detector tune --features ko_domain30 --model hgb_ko --trials 1 --seed 1",
    ],
)
def test_the_commands_keep_their_options(line: str) -> None:
    args = main.build_parser().parse_args(line.split())
    assert callable(args.handler)


def test_an_unknown_repair_is_refused_by_the_parser() -> None:
    with pytest.raises(SystemExit):
        main.build_parser().parse_args(["check", "--repair", "detector"])


def test_the_detector_commands_default_to_three_seeds() -> None:
    args = main.build_parser().parse_args(["detector", "train", "--version", "v9"])
    assert (args.features, args.models, main._seeds(args)) == (
        "ko_domain30",
        "hgb_ko",
        (42, 43, 44),
    )


def test_jobs_are_kept_for_models_that_can_fork(capsys: pytest.CaptureFixture[str]) -> None:
    assert usecases.fork_safe_jobs(4, ["compact_v5"], ["logreg", "hgb"]) == 4
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
    ("features", "models"),
    [
        (["compact_v5"], ["ensemble_from_lr"]),
        (["compact_v5"], ["logreg", "lightgbm"]),
        (["compact_v5_v11"], ["logreg"]),
        (["compact_v5_v11"], ["blend_v11"]),
    ],
)
def test_jobs_are_disabled_with_a_message_for_openmp_libraries(
    features: list[str], models: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    assert usecases.fork_safe_jobs(4, features, models) == 1
    assert "--jobs 4 ignored" in capsys.readouterr().out


def test_one_job_needs_no_message(capsys: pytest.CaptureFixture[str]) -> None:
    assert usecases.fork_safe_jobs(1, ["compact_v5"], ["catboost"]) == 1
    assert capsys.readouterr().out == ""


def test_only_the_fork_unsafe_libraries_are_reported() -> None:
    assert fork_unsafe_libraries(["hgb", "xgboost"], ["compact_v5"]) == ["xgboost"]


def test_a_domain_error_is_one_line_and_exit_code_2(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fails(settings: Settings) -> int:
        raise DomainError("broken battle")

    monkeypatch.setattr(usecases, "check", fails)
    assert main.main(["check"]) == 2
    assert capsys.readouterr().err == "error: broken battle\n"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    pokedex = tmp_path / "species.json"
    PokedexBuilder().add_records(read_records(FIXTURES)).build().to_json(pokedex)
    return Settings(
        train=FIXTURES,
        test=FIXTURES,
        pokedex=pokedex,
        plan=FoldPlan(2, (1, 2)),
        results=ResultsLayout(tmp_path / "results"),
        repair="rule",
    )


def test_evaluate_exports_every_table(settings: Settings, tmp_path: Path) -> None:
    out = tmp_path / "out"
    code = usecases.evaluate(
        settings,
        ["compact_v5"],
        ["logreg"],
        turns=[5, 10],
        reference="compact_v5@t5/logreg",
        out=out,
    )
    assert code == 0
    names = {path.name for path in out.iterdir()}
    assert names == {
        "predictions.csv",
        "folds.csv",
        "meta.json",
        "summary.csv",
        "accuracy_by_seed.csv",
        "paired.csv",
        "accuracy_by_turn.csv",
    }
    assert json.loads((out / "meta.json").read_text())["seeds"] == [1, 2]
    assert pd.read_csv(out / "summary.csv").columns[0] == "feature_set"


def test_evaluate_without_out_writes_a_dated_evaluation(settings: Settings) -> None:
    usecases.evaluate(settings, ["compact_v5"], ["logreg"])
    (folder,) = (settings.results.root / "evaluations").iterdir()
    assert folder.name.startswith("evaluate_")
    assert (folder / "summary.csv").exists()


def test_train_then_predict_write_under_results(settings: Settings) -> None:
    assert usecases.train(settings, ["compact_v5"], ["logreg"]) == 0
    run = settings.results.run("compact_v5", "logreg")
    assert json.loads((run / "run.json").read_text())["repair"]["name"] == "knockouts_rule"
    assert usecases.predict(settings, run) == 0
    submission = pd.read_csv(settings.results.submission("submission"))
    assert list(submission.columns) == ["battle_id", "player_won"]
    assert len(submission) == 200
    assert set(submission["player_won"]) <= {0, 1}


def test_a_run_refuses_to_predict_with_another_repair(settings: Settings) -> None:
    usecases.train(settings, ["compact_v5"], ["logreg"])
    run = settings.results.run("compact_v5", "logreg")
    with pytest.raises(RepairMismatchError, match="knockouts_rule"):
        usecases.predict(replace(settings, repair="none"), run)
    assert not settings.results.submission("submission").exists()


def test_replay_prints_every_turn(settings: Settings, capsys: pytest.CaptureFixture[str]) -> None:
    battle_id = next(read_records(FIXTURES))["battle_id"]
    assert usecases.replay(settings, FIXTURES, battle_id) == 0
    assert capsys.readouterr().out.count("turn ") == 30


def test_train_with_out_keeps_the_run_folder_name(settings: Settings, tmp_path: Path) -> None:
    usecases.train(settings, ["compact_v5"], ["logreg"], out=tmp_path / "elsewhere")
    assert (tmp_path / "elsewhere" / "compact_v5__logreg" / "run.json").exists()


def test_tune_caches_the_fold_tables_and_saves_the_parameters(settings: Settings) -> None:
    pytest.importorskip("optuna")
    plan = FoldPlan(2, (2027,))
    tune_settings = replace(settings, test=None, plan=plan)
    assert usecases.tune(tune_settings, "compact_v5", "logreg", trials=1) == 0
    assert settings.results.cache("folds", "compact_v5__k2_s2027.pkl").exists()
    assert (settings.results.tuning() / "optuna.db").exists()
    saved = json.loads((settings.results.tuning() / "compact_v5__logreg.json").read_text())
    assert saved["model"] == "logreg"
