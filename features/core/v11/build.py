"""v11's feature row of one battle, in the column order of the original's ``build.py``.

``battle_features`` is the original's without ``usage_bags`` and ``hidden_features``
(``bt_ bc_ bf_ bg_ u_``), which no pipeline of ``blend_v11`` reads.
"""

from __future__ import annotations

from typing import Any

from features.core.v11.active import active_features
from features.core.v11.bags import Vocab, bag_features
from features.core.v11.condition import condition_features
from features.core.v11.events import event_features
from features.core.v11.matchup import matchup_features, wincon_features
from features.core.v11.replay import Battle
from features.core.v11.simulation import MoveTable, duel_features
from features.core.v11.simulation2 import simulation_features
from features.core.v11.sleep import sleep_features
from features.core.v11.state import p1_track, p2_track, state_features
from features.core.v11.static import static_features
from features.core.v11.switching import SwitchRates, behaviour_features
from features.core.v11.team import team_keys
from features.core.v11.team_model import TeamModel


def battle_features(
    b: Battle, vocab: Vocab, team_model: TeamModel, moves: MoveTable, rates: SwitchRates
) -> dict[str, Any]:
    """``battle_id`` and every feature group of a battle whose ``p2_faint`` is already set."""
    t1, t2 = p1_track(b), p2_track(b)
    f: dict[str, Any] = {"battle_id": b.id}
    f.update(static_features(b))
    f.update(state_features(b, t1, t2))
    f.update(active_features(b))
    f.update(event_features(b))
    f.update(matchup_features(b, t1, t2))
    f.update(wincon_features(b, t1, t2))
    f.update(sleep_features(b, t1, t2))
    f.update(bag_features(b, vocab, t1, t2))
    f.update(team_keys(b))
    f.update(duel_features(b, t1, t2, moves))
    f.update(behaviour_features(b, rates))
    f.update(simulation_features(b, t1, t2, moves, team_model))
    f.update(condition_features(b, t1, t2, team_model))
    return f
