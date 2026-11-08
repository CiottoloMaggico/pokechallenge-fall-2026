"""Feature sets of the knockout detector's task: one row per switch (``Transition``)."""

from __future__ import annotations

from collections.abc import Callable

from domain.transitions import Transition
from feature_set import FeatureSet
from features.transitions import ko_domain30, ko_raw50
from registry import Registry

TRANSITION_SETS: Registry[Callable[[], FeatureSet[Transition]]] = Registry("detector feature set")
TRANSITION_SETS.add("ko_raw50", ko_raw50.build)
TRANSITION_SETS.add("ko_domain30", ko_domain30.build)


def build(name: str) -> FeatureSet[Transition]:
    """The detector feature set called ``name``."""
    return TRANSITION_SETS[name]()
