"""The groups of v11_fs without fitted state, remembered on the battle context.

Static, state, actives, events, matchup, win conditions, sleep and team keys depend only on the
battle and its repair, so each is computed once per context (``context.cached`` under
``("v11_fs", <group>)``) and shared by the group features (``groups.py``), the forecast's inputs
(``forecast.inputs``, truncated contexts included) and every fold of an evaluation. The groups
with fitted state (bags, duel, behaviour, simulation, condition) are never remembered.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from domain.battle import BattleContext
from features.core.v11_fs.active import active_features
from features.core.v11_fs.battle import battle_of
from features.core.v11_fs.events import event_features
from features.core.v11_fs.matchup import matchup_features, wincon_features
from features.core.v11_fs.sleep import sleep_features
from features.core.v11_fs.state import state_features, tracks_of
from features.core.v11_fs.static import static_features
from features.core.v11_fs.team import team_keys

Group = dict[str, Any]


def _remembered(context: BattleContext, name: str, compute: Callable[[], Group]) -> Group:
    return context.cached(("v11_fs", name), compute)


def static_of(c: BattleContext) -> Group:
    """``static_features`` of the battle."""
    return _remembered(c, "static", lambda: static_features(battle_of(c)))


def state_of(c: BattleContext) -> Group:
    """``state_features`` of the battle."""
    return _remembered(c, "state", lambda: state_features(battle_of(c), *tracks_of(c)))


def active_of(c: BattleContext) -> Group:
    """``active_features`` of the battle."""
    return _remembered(c, "active", lambda: active_features(battle_of(c)))


def events_of(c: BattleContext) -> Group:
    """``event_features`` of the battle."""
    return _remembered(c, "events", lambda: event_features(battle_of(c)))


def matchup_of(c: BattleContext) -> Group:
    """``matchup_features`` of the battle."""
    return _remembered(c, "matchup", lambda: matchup_features(battle_of(c), *tracks_of(c)))


def wincon_of(c: BattleContext) -> Group:
    """``wincon_features`` of the battle."""
    return _remembered(c, "wincon", lambda: wincon_features(battle_of(c), *tracks_of(c)))


def sleep_of(c: BattleContext) -> Group:
    """``sleep_features`` of the battle."""
    return _remembered(c, "sleep", lambda: sleep_features(battle_of(c), *tracks_of(c)))


def team_keys_of(c: BattleContext) -> Group:
    """``team_keys`` of the battle."""
    return _remembered(c, "team_keys", lambda: team_keys(battle_of(c)))
