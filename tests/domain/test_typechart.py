"""The type chart read from the competition file."""

from __future__ import annotations

from pathlib import Path

import pytest

from domain.errors import DomainError, TypeChartError
from domain.typechart import TypeChart

CHART = Path(__file__).resolve().parents[2] / "assets" / "type_chart_gen2.csv"


@pytest.fixture(scope="module")
def chart() -> TypeChart:
    return TypeChart.from_csv(CHART)


@pytest.mark.parametrize(
    ("move_type", "defender", "expected"),
    [
        ("fire", ["grass"], 2.0),
        ("normal", ["ghost"], 0.0),
        ("fire", ["water"], 0.5),
        ("normal", ["water"], 1.0),
        ("ice", ["dragon", "flying"], 4.0),
        ("ground", ["fire", "flying"], 0.0),
        (None, ["ghost"], 1.0),
    ],
)
def test_effectiveness(
    chart: TypeChart, move_type: str | None, defender: list[str], expected: float
) -> None:
    assert chart.effectiveness(move_type, defender) == expected


def test_unknown_type(chart: TypeChart) -> None:
    with pytest.raises(KeyError):
        chart.multiplier("fire", "shadow")


def test_malformed_files_are_domain_errors(tmp_path: Path) -> None:
    path = tmp_path / "chart.csv"
    path.write_text("attacking_type,defending_type,multiplier\nfire,grass,3.0\n", encoding="utf-8")
    with pytest.raises(TypeChartError) as caught:
        TypeChart.from_csv(path)
    assert isinstance(caught.value, DomainError)
    assert isinstance(caught.value, ValueError)
