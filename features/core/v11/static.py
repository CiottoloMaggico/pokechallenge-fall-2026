"""v11's static descriptors: Player 1 team stats, levels, items and type weaknesses, Player 2 lead
and the lead matchup. A copy of the original's ``features/static.py``."""

from __future__ import annotations

from typing import Any

import numpy as np

from domain import expected_damage as ed
from features.core.v11.battle import Battle
from features.core.v11.moves import ALL_TYPES, CHART, STATS

ITEMS = ("leftovers", "thickclub", "miracleberry", "mintberry", "lightball")


def static_features(b: Battle) -> dict[str, Any]:
    """The ``s_`` columns of a battle."""
    f: dict[str, Any] = {}
    team = b.team
    stats = np.array([[p[f"base_{s}"] for s in STATS] for p in team], dtype=float)
    for k, s in enumerate(STATS):
        f[f"s_p1_{s}_mean"] = stats[:, k].mean()
        f[f"s_p1_{s}_max"] = stats[:, k].max()
    bst = stats.sum(1)
    f["s_p1_bst_mean"], f["s_p1_bst_min"], f["s_p1_bst_std"] = bst.mean(), bst.min(), bst.std()
    f["s_p1_level_min"] = min(p["level"] for p in team)
    f["s_p1_n_lowlevel"] = sum(p["level"] < 100 for p in team)
    items = [p["item"] for p in team]
    for it in ITEMS:
        f[f"s_p1_item_{it}"] = items.count(it)
    f["s_p1_item_other"] = sum(it not in ITEMS for it in items)
    types = [b.stats(p["name"]).types for p in team]
    weak = {t: sum(ed.effectiveness(CHART, t, ts) > 1 for ts in types) for t in ALL_TYPES}
    resist = {t: sum(ed.effectiveness(CHART, t, ts) < 1 for ts in types) for t in ALL_TYPES}
    f["s_p1_weak_max"] = max(weak.values())
    f["s_p1_weak_sum"] = sum(weak.values())
    f["s_p1_resist_sum"] = sum(resist.values())
    f["s_p1_unresisted_types"] = sum(v == 0 for v in resist.values())
    f["s_p1_type_diversity"] = len({t for ts in types for t in ts})

    lead1, lead2 = team[0], b.lead2
    d1, d2 = b.stats(lead1["name"]), b.stats(lead2["name"])
    for s in STATS:
        f[f"s_p2_lead_{s}"] = lead2[f"base_{s}"]
    f["s_p2_lead_bst"] = d2.base_stat_total
    f["s_p2_lead_level"] = lead2["level"]
    f["s_lead_bst_diff"] = d1.base_stat_total - d2.base_stat_total
    f["s_lead_stab12"] = ed.stab_effectiveness(CHART, d1.types, d2.types)
    f["s_lead_stab21"] = ed.stab_effectiveness(CHART, d2.types, d1.types)
    f["s_lead_speed_diff"] = d1.base_spe - d2.base_spe
    return f
