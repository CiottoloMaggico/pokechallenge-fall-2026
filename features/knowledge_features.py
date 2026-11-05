"""Features that use the team archive (``domain.team_knowledge``).

They are the archive versions of existing columns: the same formulas, but
each Pokémon is assumed to carry the moves its team usually shows, and P2's
hidden Pokémon come from the archive when its team can be identified.

* ``kb_ttk_adv``: ``act_ttk_adv`` with likely moves for the two actives;
* ``kb_sim_seen_left``: ``sim_seen_left`` with likely moves for every
  Pokémon, against P2's revealed Pokémon only;
* ``kb_sim_full_win``: ``sim_full_win`` with the same moves, also against
  P2's hidden Pokémon (identified team, otherwise the naive Bayes guess);
* ``kb_hidden_bst``: ``hidden_bst`` for the same hidden Pokémon;
* ``kb_recovery``: ``wc_healers`` as expected counts: a Pokémon with
  probability ``p`` of having a recovery move adds ``p``. P2's hidden
  Pokémon count only when its team is identified;
* ``kb_setup``: the same with boosting moves (Curse for non-Ghosts).

All six lie in [-1, 1] and are positive when P1 is ahead, as their
originals. The archive is fitted on the training part of each fold only;
a battle is scored without its own contribution (leave-one-out), so
training rows see the archive as validation rows do.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence

from domain.battle import BattleContext, PlayerState, PokemonState
from domain.moves import Move
from domain.records import fingerprint
from domain.team_knowledge import (
    MIN_TEAM_SHARE,
    SMOOTHING,
    THRESHOLD,
    KnowledgeView,
    Team,
    TeamKnowledge,
    complete_team,
)
from domain.typechart import TypeChart
from feature_set import Feature
from features.matchup_features import ttk_advantage
from features.simulation_features import (
    TEAM_SIZE,
    DuelSimulator,
    Fighter,
    is_attack,
    make_fighter,
    outcome,
    stats_missing,
)
from features.team_features import MAX_BST, HiddenTeamStrength
from features.turn_features import check_turn
from features.wincon_features import CURSE, HEALING, SETUP

OUTPUTS = (
    "kb_ttk_adv",
    "kb_sim_seen_left",
    "kb_sim_full_win",
    "kb_hidden_bst",
    "kb_recovery",
    "kb_setup",
)
FINGERPRINT_KEY = "kb_fingerprint"
GHOST = "ghost"

MoveSet = Callable[[PokemonState], frozenset[str]]


def _healing(pokemon: PokemonState) -> frozenset[str]:
    return frozenset(HEALING)


def _setup(pokemon: PokemonState) -> frozenset[str]:
    """Boosting moves this Pokémon could have (Curse only if not a Ghost)."""
    if GHOST in pokemon.info.types:
        return frozenset(SETUP)
    return frozenset({*SETUP, CURSE})


class TeamKnowledgeFeatures(Feature[BattleContext]):
    """The six archive features at the sampled turn (see the module docstring)."""

    requires_fit = True

    def __init__(
        self,
        chart: TypeChart,
        turn: int | None = None,
        smoothing: float = SMOOTHING,
        min_share: float = MIN_TEAM_SHARE,
        threshold: float = THRESHOLD,
    ) -> None:
        super().__init__("team_knowledge")
        self.chart = chart
        self.turn = check_turn(turn)
        self.smoothing = smoothing
        self.min_share = min_share
        self.threshold = threshold
        self.knowledge: TeamKnowledge | None = None
        self.guesser = HiddenTeamStrength(turn)
        self.simulator = DuelSimulator(chart)

    @property
    def output_names(self) -> tuple[str, ...]:
        return OUTPUTS

    def _fit(
        self, battles: Sequence[BattleContext], labels: Sequence[int] | None
    ) -> None:
        self.knowledge = TeamKnowledge.build(battles, self.smoothing)
        self.guesser.fit(battles)

    def compute(self, battle: BattleContext) -> Mapping[str, float]:
        if self.knowledge is None:
            raise RuntimeError("team knowledge is not fitted")
        # a property of the whole battle, so it is safe to remember it there
        digest = battle.cached(FINGERPRINT_KEY, lambda: fingerprint(battle.record))
        view = self.knowledge.excluding(digest)
        context = battle.upto(self.turn)
        p1, p2 = context.side("p1"), context.side("p2")
        team_1 = complete_team(p1)
        team_2, hidden = self._p2_team(view, p2)
        identified = team_2 is not None

        def active_moves(ctx: BattleContext, side: str) -> tuple[Move, ...]:
            team = team_1 if side == "p1" else team_2
            return self._attacks(view, team, ctx.side(side).active.pokemon)

        values: dict[str, float] = {
            "kb_ttk_adv": ttk_advantage(self.chart, active_moves)(context),
        }

        mine = [self._fighter(p1, p, team_1, view) for p in _alive(p1)]
        seen = [self._fighter(p2, p, team_2, view) for p in _alive(p2)]
        unseen = [self._hidden_fighter(context, name, team_2, view) for name in hidden]
        actives = (p1.active.pokemon.name, p2.active.pokemon.name)

        values["kb_sim_seen_left"] = values["kb_sim_full_win"] = math.nan
        if not stats_missing(mine, seen):
            left_a, left_b = self.simulator.simulate(mine, seen, actives)
            values["kb_sim_seen_left"] = (len(left_a) - len(left_b)) / TEAM_SIZE
        if not stats_missing(mine, seen, unseen):
            win_a, win_b = self.simulator.simulate(mine, [*seen, *unseen], actives)
            values["kb_sim_full_win"] = outcome(win_a, win_b)

        values["kb_hidden_bst"] = _hidden_bst(context, hidden)
        counted = hidden if identified else []
        values["kb_recovery"] = self._expected_difference(
            (p1, team_1), (p2, team_2), view, context, counted, _healing
        )
        values["kb_setup"] = self._expected_difference(
            (p1, team_1), (p2, team_2), view, context, counted, _setup
        )
        # _setup needs the types (Curse only boosts a non-Ghost)
        if stats_missing(mine, seen, unseen):
            values["kb_setup"] = math.nan
        return {name: float(value) for name, value in values.items()}

    def _p2_team(
        self, view: KnowledgeView, p2: PlayerState
    ) -> tuple[Team | None, list[str]]:
        """P2's team (None if unknown) and the species of its hidden Pokémon."""
        revealed = frozenset(p2.team)
        if len(revealed) == TEAM_SIZE:
            return revealed, []
        team = view.identify(revealed, self.min_share)
        if team is None:
            return None, self.guesser.guess(list(revealed), p2.team_size)
        return team, sorted(team - revealed)

    def _attacks(
        self, view: KnowledgeView, team: Team | None, pokemon: PokemonState
    ) -> tuple[Move, ...]:
        likely = view.likely_moves(team, pokemon, self.threshold)
        return tuple(move for move in likely if is_attack(move))

    def _fighter(
        self,
        player: PlayerState,
        pokemon: PokemonState,
        team: Team | None,
        view: KnowledgeView,
    ) -> Fighter:
        boosts = player.active.boosts if pokemon is player.active.pokemon else None
        moves = self._attacks(view, team, pokemon)
        return make_fighter(pokemon, moves, boosts)

    def _hidden_fighter(
        self, context: BattleContext, name: str, team: Team | None, view: KnowledgeView
    ) -> Fighter:
        pokemon = context.new_pokemon(name)
        return make_fighter(pokemon, self._attacks(view, team, pokemon), None)

    def _expected_difference(
        self,
        side_1: tuple[PlayerState, Team | None],
        side_2: tuple[PlayerState, Team | None],
        view: KnowledgeView,
        context: BattleContext,
        hidden: Sequence[str],
        moveset: MoveSet,
    ) -> float:
        """``(E1 - E2) / 6``, E = expected alive Pokémon holding a move of the set."""
        player_1, team_1 = side_1
        player_2, team_2 = side_2
        expected_1 = sum(_has_move(view, team_1, p, moveset) for p in _alive(player_1))
        expected_2 = sum(_has_move(view, team_2, p, moveset) for p in _alive(player_2))
        for name in hidden:
            pokemon = context.new_pokemon(name)
            expected_2 += _has_move(view, team_2, pokemon, moveset)
        return (expected_1 - expected_2) / TEAM_SIZE


def _alive(player: PlayerState) -> list[PokemonState]:
    return [p for p in player.team.values() if not p.fainted]


def _has_move(
    view: KnowledgeView,
    team: Team | None,
    pokemon: PokemonState,
    moveset: MoveSet,
) -> float:
    """Probability that ``pokemon`` has at least one move of the set.

    A move already revealed counts as certain; the others are independent.
    """
    names = moveset(pokemon)
    if names & set(pokemon.known_moves):
        return 1.0
    p_none = 1.0
    for move in sorted(names):
        p_none *= 1.0 - view.move_probability(team, pokemon.name, move)
    return 1.0 - p_none


def _hidden_bst(context: BattleContext, hidden: Sequence[str]) -> float:
    if not hidden:
        return 0.0
    bst = [context.new_pokemon(name).info.base_stat_total for name in hidden]
    return -sum(bst) / len(bst) / MAX_BST
