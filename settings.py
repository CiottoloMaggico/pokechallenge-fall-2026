"""What every command needs to know about its environment."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from folds import FoldPlan
from storage import ResultsLayout

DEFAULT_TRAIN = Path("input/train.jsonl")
DEFAULT_TEST = Path("input/test.jsonl")
DEFAULT_POKEDEX = ResultsLayout().cache("species.json")


@dataclass(frozen=True)
class Settings:
    """Inputs, folds and output layout of one command; ``test`` is optional except in predict."""

    train: Path = DEFAULT_TRAIN
    test: Path | None = None
    pokedex: Path = DEFAULT_POKEDEX
    plan: FoldPlan = field(default_factory=FoldPlan)
    jobs: int = 1
    results: ResultsLayout = field(default_factory=ResultsLayout)
