"""v11's feature groups, one ``Feature`` per group, on ``BattleContext`` and any repair.

Each feature runs one group function of the original's ``battle_features`` on the battle's view
(``battle.battle_of``) and its tracks (``state.tracks_of``), and returns its columns as floats
in the group's order (the original's ``astype(float)``). The fitted ones learn label-free state
from the training battles only: the vocabularies of the bags, the typical moves of each species,
the team co-occurrences, the switch rates. ``features/v11_fs.py`` registers them in v11's order
(the bags and team keys only as inputs of ``label_aware``), then the forecast
(``forecast.V11Forecast``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from domain.battle import BattleContext
from features.core.v11_fs.active import active_features
from features.core.v11_fs.bags import Vocab, bag_features
from features.core.v11_fs.battle import battle_of
from features.core.v11_fs.condition import condition_features
from features.core.v11_fs.events import event_features
from features.core.v11_fs.matchup import matchup_features, wincon_features
from features.core.v11_fs.simulation import MoveTable, duel_features
from features.core.v11_fs.simulation2 import simulation_features
from features.core.v11_fs.sleep import sleep_features
from features.core.v11_fs.state import state_features, tracks_of
from features.core.v11_fs.static import static_features
from features.core.v11_fs.switching import SwitchRates, behaviour_features
from features.core.v11_fs.team import team_keys
from features.core.v11_fs.team_model import TeamModel
from interfaces import Feature


def floats(values: Mapping[str, Any]) -> dict[str, float]:
    """A group's columns as floats, in its order (NaN kept)."""
    return {name: float(value) for name, value in values.items()}


class V11Static(Feature[BattleContext]):
    """``s_``: P1's team stats, levels, items and weaknesses, P2's lead, the lead matchup."""

    def compute(self, battle: BattleContext) -> Mapping[str, float]:
        return floats(static_features(battle_of(battle)))


class V11State(Feature[BattleContext]):
    """``f_``, ``t_``: faints (hidden P2 ones from the repair), HP, status and their trajectory."""

    def compute(self, battle: BattleContext) -> Mapping[str, float]:
        return floats(state_features(battle_of(battle), *tracks_of(battle)))


class V11Active(Feature[BattleContext]):
    """``a_``: the two actives, speed order, damage and turns-to-KO both ways."""

    def compute(self, battle: BattleContext) -> Mapping[str, float]:
        return floats(active_features(battle_of(battle)))


class V11Events(Feature[BattleContext]):
    """``e_``: each player's move classes, effectiveness, switches, damage, hazards, weather."""

    def compute(self, battle: BattleContext) -> Mapping[str, float]:
        return floats(event_features(battle_of(battle)))


class V11Matchup(Feature[BattleContext]):
    """``m_``: the matchup matrix of the remaining Pokémon of both players."""

    def compute(self, battle: BattleContext) -> Mapping[str, float]:
        return floats(matchup_features(battle_of(battle), *tracks_of(battle)))


class V11WinConditions(Feature[BattleContext]):
    """``w_``: spinners, set-up, phazers, healers, boomers, sleepers, Spikes, trapping."""

    def compute(self, battle: BattleContext) -> Mapping[str, float]:
        return floats(wincon_features(battle_of(battle), *tracks_of(battle)))


class V11Sleep(Feature[BattleContext]):
    """``x_``: longest current sleep, Rest and induced sleepers of each player."""

    def compute(self, battle: BattleContext) -> Mapping[str, float]:
        return floats(sleep_features(battle_of(battle), *tracks_of(battle)))


class V11Bags(Feature[BattleContext]):
    """``bs_``, ``bh_``, ``bm_``: species, alive HP and move bags over the training vocabularies."""

    requires_fit = True

    def __init__(self) -> None:
        super().__init__()
        self.vocab: Vocab | None = None

    def _fit(self, battles: Sequence[BattleContext], labels: Sequence[int] | None) -> None:
        self.vocab = Vocab([battle_of(c) for c in battles])

    def compute(self, battle: BattleContext) -> Mapping[str, float]:
        assert self.vocab is not None
        return floats(bag_features(battle_of(battle), self.vocab, *tracks_of(battle)))


class V11TeamKeys(Feature[BattleContext]):
    """``k_``: ids of the teams, sub-teams and species of both players (read by the team effect)."""

    def compute(self, battle: BattleContext) -> Mapping[str, float]:
        return floats(team_keys(battle_of(battle)))


class V11Duel(Feature[BattleContext]):
    """``d_``: the duel of the actives and a chain of duels, with the typical moves of training."""

    requires_fit = True

    def __init__(self) -> None:
        super().__init__()
        self.moves: MoveTable | None = None

    def _fit(self, battles: Sequence[BattleContext], labels: Sequence[int] | None) -> None:
        self.moves = MoveTable([battle_of(c) for c in battles])

    def compute(self, battle: BattleContext) -> Mapping[str, float]:
        assert self.moves is not None
        return floats(duel_features(battle_of(battle), *tracks_of(battle), self.moves))


class V11Behaviour(Feature[BattleContext]):
    """``v_``: switch probabilities and excess switching against the training switch rates."""

    requires_fit = True

    def __init__(self) -> None:
        super().__init__()
        self.rates: SwitchRates | None = None

    def _fit(self, battles: Sequence[BattleContext], labels: Sequence[int] | None) -> None:
        self.rates = SwitchRates([battle_of(c) for c in battles])

    def compute(self, battle: BattleContext) -> Mapping[str, float]:
        assert self.rates is not None
        return floats(behaviour_features(battle_of(battle), self.rates))


class V11Simulation(Feature[BattleContext]):
    """``z_``: the second simulation, with the unrevealed P2 Pokémon guessed from team models."""

    requires_fit = True

    def __init__(self) -> None:
        super().__init__()
        self.moves: MoveTable | None = None
        self.teams: TeamModel | None = None

    def _fit(self, battles: Sequence[BattleContext], labels: Sequence[int] | None) -> None:
        views = [battle_of(c) for c in battles]
        self.moves, self.teams = MoveTable(views), TeamModel(views)

    def compute(self, battle: BattleContext) -> Mapping[str, float]:
        assert self.moves is not None and self.teams is not None
        b, (t1, t2) = battle_of(battle), tracks_of(battle)
        return floats(simulation_features(b, t1, t2, self.moves, self.teams))


class V11Condition(Feature[BattleContext]):
    """``c_``: Spikes, boosts, healthy and statused shares, Snorlax, bulk left."""

    requires_fit = True

    def __init__(self) -> None:
        super().__init__()
        self.teams: TeamModel | None = None

    def _fit(self, battles: Sequence[BattleContext], labels: Sequence[int] | None) -> None:
        self.teams = TeamModel([battle_of(c) for c in battles])

    def compute(self, battle: BattleContext) -> Mapping[str, float]:
        assert self.teams is not None
        return floats(condition_features(battle_of(battle), *tracks_of(battle), self.teams))
