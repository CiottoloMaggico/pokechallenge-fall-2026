"""Imports every feature set file, which registers its builders, and exposes the registry.

::

    features = build("compact_v5")        # sampled at the last recorded turn
    features = build("compact_v5@t10")    # the same features at turn 10

The ``@t<N>`` suffix samples every feature of the set on the battle truncated after
turn N (see ``features.core.turn``); column names stay the same, so results at
different turns can be compared and plotted.
"""

from features import (  # noqa: F401  (imported for the registration)
    compact,
    compact_ko,
    compact_plus,
    compact_v2,
    compact_v3,
    compact_v4,
    compact_v5,
    core_set,
    hazards,
    hp30,
    material,
    selection_pool,
    switch,
)
from features.common import type_chart
from registry import FEATURE_SETS, parse_name
from registry import build_feature_set as build

__all__ = ["FEATURE_SETS", "build", "parse_name", "type_chart"]
