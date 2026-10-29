"""Weighted average of the probabilities of other models."""

from __future__ import annotations

from collections.abc import Sequence

from model import EnsembleModel, Model

DEFAULT_MEMBERS = ("logreg", "hgb")


def build(
    name: str, members: Sequence[Model], weights: Sequence[float] | None = None
) -> Model:
    return EnsembleModel(name, members, weights)
