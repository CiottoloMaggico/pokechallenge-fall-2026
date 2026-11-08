"""Every path the program writes to: the only place that knows the layout of ``results/``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ResultsLayout:
    """Where each kind of output lives; nothing is created until something is written there."""

    root: Path = Path("results")

    def run(self, feature_set: str, model: str) -> Path:
        """Folder of a trained run."""
        return self.root / "runs" / f"{feature_set}__{model}"

    def evaluation(self, name: str) -> Path:
        """Folder of the tables of one ``evaluate``."""
        return self.root / "evaluations" / name

    def tuning(self) -> Path:
        """Folder of the Optuna study and of the best parameters."""
        return self.root / "tuning"

    def cache(self, *parts: str) -> Path:
        """Files that can be rebuilt: the Pokédex, fold tables."""
        return self.root.joinpath("cache", *parts)

    def submission(self, name: str) -> Path:
        """A submission file (``name`` without the extension)."""
        return self.root / "submissions" / f"{name}.csv"
