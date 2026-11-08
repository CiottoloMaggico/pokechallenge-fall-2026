"""evaluate -> train -> predict through ``main.main`` on the 200 fixture battles (seconds).

Only the fixtures are read and only ``tmp_path`` is written: the Pokédex is built from the
fixtures and the repair is the deterministic rule, so no artifact of ``results/`` is needed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import main
from domain.pokedex import PokedexBuilder
from domain.records import read_records

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "battles.jsonl"


def test_evaluate_train_predict(tmp_path: Path) -> None:
    pokedex = tmp_path / "species.json"
    PokedexBuilder().add_records(read_records(FIXTURES)).build().to_json(pokedex)
    common = ["--pokedex", str(pokedex)]
    data = ["--train", str(FIXTURES), "--repair", "rule"]
    model = ["--features", "compact_v5", "--models", "logreg"]

    evaluation = tmp_path / "evaluation"
    assert main.main([*common, "evaluate", *data, *model, "--out", str(evaluation)]) == 0
    summary = pd.read_csv(evaluation / "summary.csv")
    assert summary[["feature_set", "model"]].values.tolist() == [["compact_v5", "logreg"]]
    assert 0.5 < summary.loc[0, "accuracy_mean"] <= 1.0

    runs = tmp_path / "runs"
    assert main.main([*common, "train", *data, *model, "--out", str(runs)]) == 0
    run = runs / "compact_v5__logreg"
    assert json.loads((run / "run.json").read_text())["repair"]["name"] == "knockouts_rule"

    submission, proba = tmp_path / "submission.csv", tmp_path / "proba.csv"
    predict = ["predict", "--run", str(run), "--test", str(FIXTURES), "--repair", "rule"]
    assert main.main([*common, *predict, "--out", str(submission), "--proba", str(proba)]) == 0
    predicted = pd.read_csv(proba)
    assert list(pd.read_csv(submission).columns) == ["battle_id", "player_won"]
    assert predicted["battle_id"].tolist() == [r["battle_id"] for r in read_records(FIXTURES)]
    expected = (predicted["p_player_won"] > 0.5).astype(int)
    assert predicted["player_won"].tolist() == expected.tolist()
