"""v11's detector as a repair addon, on a tiny artifact built from the fixtures in ``tmp_path``."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import detector
import models.ko_v11
from addons import check_spec, load_repair
from addons.knockouts_model import ArtifactError
from addons.knockouts_v11 import V11KnockoutRepair, battle_key, load_artifact
from domain.pokedex import Missing, Pokedex, PokedexBuilder
from domain.records import read_records, without_label
from domain.switches import SwitchExtractor
from features.common import type_chart
from folds import FoldPlan
from settings import Settings
from storage import ResultsLayout

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "battles.jsonl"
CUT = 12

Record = dict[str, Any]


@pytest.fixture(scope="module")
def artifact(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """``detector train-v11`` on the fixtures: few trees, one cut, two seeds."""
    root = tmp_path_factory.mktemp("results")
    PokedexBuilder().add_records(read_records(FIXTURES)).build("fixtures").to_json(
        root / "species.json"
    )
    settings = Settings(
        FIXTURES,
        pokedex=root / "species.json",
        plan=FoldPlan(5, (42, 1)),
        results=ResultsLayout(root),
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(detector, "CUTS", (CUT,))
        patch.setitem(models.ko_v11.PARAMS, "n_estimators", 20)
        assert detector.train_v11(settings, "t1") == 0
    yield root / "addons" / "knockouts_v11" / "t1"


@pytest.fixture(scope="module")
def pokedex(artifact: Path) -> Pokedex:
    return Pokedex.from_json(artifact.parents[2] / "species.json")


@pytest.fixture(scope="module")
def repair(artifact: Path, pokedex: Pokedex) -> V11KnockoutRepair:
    return V11KnockoutRepair.from_artifact(artifact, pokedex, type_chart())


@pytest.fixture(scope="module")
def records() -> list[Record]:
    return [without_label(r) for r in read_records(FIXTURES)]


def p2_switches(record: Record, pokedex: Pokedex) -> list[Any]:
    return SwitchExtractor(pokedex.lookup(record, Missing()), type_chart()).extract(record, "p2")


def test_training_battles_get_their_cross_fitted_row(
    artifact: Path, repair: V11KnockoutRepair, records: list[Record], pokedex: Pokedex
) -> None:
    _, tables, _ = load_artifact(artifact)
    candidates = 0
    for record in records:
        found = repair.probabilities(record)
        assert found == tables["full"][battle_key(record)]
        switches = p2_switches(record, pokedex)
        assert list(found) == [s.index for s in switches]
        for s in switches:
            if s.recorded:
                assert found[s.index] == 1.0
            elif not s.final:
                assert found[s.index] == 0.0
            else:
                assert 0.0 < found[s.index] < 1.0
                candidates += 1
    assert candidates > 0


def test_truncated_training_battles_get_the_cut_table(
    artifact: Path, repair: V11KnockoutRepair, records: list[Record]
) -> None:
    _, tables, _ = load_artifact(artifact)
    for record in records[:50]:
        cut = {**record, "battle_timeline": record["battle_timeline"][:CUT]}
        assert repair.probabilities(cut) == tables["cuts"][CUT][battle_key(cut)]


def test_other_battles_get_the_full_detector(
    artifact: Path, repair: V11KnockoutRepair, records: list[Record], pokedex: Pokedex
) -> None:
    run, tables, _ = load_artifact(artifact)
    for record in records[:30]:
        new = {**record, "battle_id": -1}
        assert battle_key(new) not in tables["full"]
        found = repair.probabilities(new)
        switches = p2_switches(new, pokedex)
        candidates = [s for s in switches if s.final and not s.recorded]
        expected = run.predict_proba(candidates) if candidates else np.array([])
        assert [found[s.index] for s in candidates] == expected.tolist()
        assert all(found[s.index] == 1.0 for s in switches if s.recorded)


def test_battles_without_p1_rows_get_the_full_model(
    artifact: Path, repair: V11KnockoutRepair, records: list[Record], pokedex: Pokedex
) -> None:
    run, _, info = load_artifact(artifact)
    chart = type_chart()
    alone = [
        r
        for r in records
        if not SwitchExtractor(pokedex.lookup(r, Missing()), chart).examples(r)
        and SwitchExtractor(pokedex.lookup(r, Missing()), chart).candidates(r)
    ]
    assert info["battles_without_p1_rows"] >= len(alone) > 0
    for record in alone:
        candidates = SwitchExtractor(pokedex.lookup(record, Missing()), chart).candidates(record)
        found = repair.probabilities(record)
        assert [found[s.index] for s in candidates] == run.predict_proba(candidates).tolist()


def test_repairs_describe_and_specs(
    artifact: Path, repair: V11KnockoutRepair, records: list[Record], pokedex: Pokedex
) -> None:
    for record in records[:30]:
        probabilities = repair.probabilities(record)
        timeline = record["battle_timeline"]
        assert repair.repairs(record) == {
            i: timeline[i - 1]["p2_pokemon_state"]["name"]
            for i, p in probabilities.items()
            if p >= 0.5
        }
    described = json.loads(json.dumps(repair.describe()))
    assert described["name"] == "knockouts_v11" and described["version"] == "t1"
    assert described["params"]["cuts"] == [CUT] and described["params"]["seeds"] == [42, 1]
    assert check_spec(f"v11={artifact}") == f"v11={artifact}"
    assert isinstance(load_repair(f"v11={artifact}", pokedex, type_chart()), V11KnockoutRepair)
    with pytest.raises(ValueError, match="unknown repair"):
        check_spec("v11=")


def test_a_missing_artifact_says_how_to_build_it(pokedex: Pokedex, tmp_path: Path) -> None:
    with pytest.raises(ArtifactError, match="detector train-v11 --version v9"):
        load_repair(f"v11={tmp_path / 'v9'}", pokedex, type_chart())
