"""Features that use the pool of teams seen in the training battles.

Teams repeat a lot: 4,167 distinct P1 teams in 10,000 battles, and 71%
of the P2 teams that are fully revealed by turn 30 are exactly a P1 team
of another battle. So the pool of known teams (every P1 team plus every
P2 team revealed in full, fitted on the training battles) usually tells
which team P2 is playing, and therefore which Pokémon it still hides.

* ``tr_p1`` / ``tr_p2`` (labels): win rate of P1's team and of P2's
  team, each counted from both sides (a team wins as P1 when P1 wins and
  as P2 when P1 loses), smoothed towards 0.5 with ``m`` pseudo battles,
  minus 0.5. P2's team is the mixture of the pool teams containing every
  P2 Pokémon revealed so far, weighted by how often they occur (ignored
  when more than ``max_candidates`` match). Fitted inside each fold and
  out of fold on the training rows, like the other encodings.
* ``phpw_spd``: ``hpw_spd`` with P2's unrevealed slots filled with the
  Pokémon of its most likely pool team (at full HP) instead of the mean
  of its revealed ones; unchanged when no pool team matches.
* ``smat_hp``: ``material_hp`` where each P2 Pokémon that may have
  fainted unseen counts with ``(1 - p)`` times its last HP, ``p`` being
  the knockout detector's probability, instead of all (p < 0.5) or
  nothing (p >= 0.5).

Other compact helpers:

* ``p2_fainted``: P2 Pokémon knocked out (recorded or repaired), over 6.
* ``alive_gain_10_30``: change of ``P1 alive - P2 alive`` between turn
  10 and the sampled turn, over 6.
* ``b_stat_diff``: share of P1's moves that are status moves minus the
  same share for P2.
"""

from __future__ import annotations

import collections
from collections.abc import Mapping, Sequence

from domain.battle import BattleContext, PlayerState
from feature_set import Feature
from features.encoding_features import TeamEncoding
from features.turn_features import check_turn

TEAM_SIZE = 6
FAINTED = "fnt"

Team = frozenset[str]


def p2_revealed(context: BattleContext) -> list[str]:
    """P2's Pokémon seen so far, in order of appearance."""
    return list(context.side("p2").team)


class TeamPool:
    """Teams seen in some battles and their frequencies (no labels)."""

    def __init__(self) -> None:
        self.count: collections.Counter[Team] = collections.Counter()
        self.by_member: dict[str, set[Team]] = {}

    @classmethod
    def from_battles(cls, battles: Sequence[BattleContext]) -> TeamPool:
        pool = cls()
        for context in battles:
            pool.count[frozenset(context.side("p1").team)] += 1
            revealed = frozenset(context.side("p2").team)
            if len(revealed) == context.side("p2").team_size:
                pool.count[revealed] += 1
        by_member: dict[str, set[Team]] = collections.defaultdict(set)
        for team in pool.count:
            for name in team:
                by_member[name].add(team)
        pool.by_member = dict(by_member)
        return pool

    def candidates(self, revealed: Sequence[str]) -> list[tuple[Team, float]]:
        """Pool teams containing every revealed Pokémon, with their weights."""
        found: set[Team] | None = None
        for name in revealed:
            teams = self.by_member.get(name, set())
            found = set(teams) if found is None else found & teams
            if not found:
                return []
        if not found:
            return []
        total = sum(self.count[t] for t in found)
        ranked = sorted(found, key=lambda t: (-self.count[t], sorted(t)))
        return [(t, self.count[t] / total) for t in ranked]

    def most_likely(self, revealed: Sequence[str]) -> Team | None:
        found = self.candidates(revealed)
        return found[0][0] if found else None


class TeamRating(Feature[BattleContext]):
    """``tr_p1`` and ``tr_p2`` (see the module docstring)."""

    requires_fit = True
    uses_labels = True

    def __init__(
        self, turn: int | None = None, smoothing: float = 2.0, max_candidates: int = 50
    ) -> None:
        super().__init__("team_rating")
        if smoothing <= 0:
            raise ValueError("smoothing must be positive")
        self.turn = check_turn(turn)
        self.smoothing = smoothing
        self.max_candidates = max_candidates
        self.pool = TeamPool()
        self.wins: dict[Team, float] = {}
        self.games: dict[Team, float] = {}

    @property
    def output_names(self) -> tuple[str, ...]:
        return ("tr_p1", "tr_p2")

    def _fit(
        self, battles: Sequence[BattleContext], labels: Sequence[int] | None
    ) -> None:
        assert labels is not None
        contexts = [b.at_turn(self.turn) for b in battles]
        self.pool = TeamPool.from_battles(contexts)
        wins: dict[Team, float] = collections.defaultdict(float)
        games: dict[Team, float] = collections.defaultdict(float)
        for context, label in zip(contexts, labels, strict=True):
            own = frozenset(context.side("p1").team)
            wins[own] += label
            games[own] += 1
            for team, weight in self._p2_teams(context):
                wins[team] += (1 - label) * weight
                games[team] += weight
        self.wins, self.games = dict(wins), dict(games)

    def _p2_teams(self, context: BattleContext) -> list[tuple[Team, float]]:
        found = self.pool.candidates(p2_revealed(context))
        return found if len(found) <= self.max_candidates else []

    def _rate(self, team: Team) -> float:
        m = self.smoothing
        return (self.wins.get(team, 0.0) + 0.5 * m) / (self.games.get(team, 0.0) + m)

    def compute(self, battle: BattleContext) -> Mapping[str, float]:
        context = battle.at_turn(self.turn)
        p1 = self._rate(frozenset(context.side("p1").team)) - 0.5
        p2_teams = self._p2_teams(context)
        p2 = sum(w * (self._rate(t) - 0.5) for t, w in p2_teams) if p2_teams else 0.0
        return {"tr_p1": p1, "tr_p2": p2}


class PoolBulk(Feature[BattleContext]):
    """``phpw_spd`` (see the module docstring)."""

    requires_fit = True

    def __init__(self, turn: int | None = None, stat: str = "spd") -> None:
        super().__init__(f"pool_bulk_{stat}")
        self.turn = check_turn(turn)
        self.stat = stat
        self.pool = TeamPool()

    @property
    def output_names(self) -> tuple[str, ...]:
        return (f"phpw_{self.stat}",)

    def _fit(
        self, battles: Sequence[BattleContext], labels: Sequence[int] | None
    ) -> None:
        self.pool = TeamPool.from_battles([b.at_turn(self.turn) for b in battles])

    def compute(self, battle: BattleContext) -> Mapping[str, float]:
        context = battle.at_turn(self.turn)
        p1, p2 = context.side("p1"), context.side("p2")
        a = _alive_bulk(p1, self.stat)
        b = _alive_bulk(p2, self.stat)
        unrevealed = p2.team_size - len(p2.team)
        if unrevealed:
            team = self.pool.most_likely(p2_revealed(context))
            hidden = [n for n in team if n not in p2.team] if team else []
            values = []
            for name in hidden[:unrevealed]:
                pokemon = context.new_pokemon(name)
                values.append(pokemon.max_hp * pokemon.stat(self.stat))
            if values:
                b += sum(values) + (unrevealed - len(values)) * sum(values) / len(values)
            elif p2.team:
                known = list(p2.team.values())
                b += unrevealed * sum(q.max_hp * q.stat(self.stat) for q in known) / len(known)
        value = (a - b) / (a + b) if a + b else 0.0
        return {f"phpw_{self.stat}": value}


def _alive_bulk(player: PlayerState, stat: str) -> float:
    return sum(q.hp * q.stat(stat) for q in player.team.values() if not q.fainted)


def soft_material_hp(context: BattleContext) -> float:
    """``smat_hp`` (see the module docstring)."""
    timeline = context.record["battle_timeline"]
    probabilities = context.repair.probabilities(context.record)
    p_ko: dict[str, float] = {}
    last_hp: dict[str, float] = {}
    for index, p in probabilities.items():
        state = timeline[index - 1]["p2_pokemon_state"]
        p_ko[state["name"]] = max(p_ko.get(state["name"], 0.0), p)
        last_hp[state["name"]] = state["hp_pct"]
    recorded = {
        t["p2_pokemon_state"]["name"]
        for t in timeline
        if t["p2_pokemon_state"]["status"] == FAINTED
    }
    p1, p2 = context.side("p1"), context.side("p2")
    own = sum(q.hp_pct for q in p1.team.values() if not q.fainted)
    other = 0.0
    for name, pokemon in p2.team.items():
        if name in recorded:
            continue
        if name in p_ko:
            other += (1.0 - p_ko[name]) * last_hp[name]
        else:
            other += pokemon.hp_pct
    other += p2.team_size - len(p2.team)
    return (own - other) / TEAM_SIZE


def p2_fainted(context: BattleContext) -> float:
    return context.side("p2").n_fainted / TEAM_SIZE


def alive_gain_10_30(context: BattleContext) -> float:
    history = context.history
    start = history[min(9, len(history) - 1)]
    end = history[-1]
    diff = lambda t: t.p1.n_alive - t.p2.n_alive  # noqa: E731
    return (diff(end) - diff(start)) / TEAM_SIZE


def status_move_share_diff(context: BattleContext) -> float:
    shares = []
    for side in ("p1", "p2"):
        moves = [t.side(side).move for t in context.history if t.side(side).move]
        shares.append(
            sum(m.category == "status" for m in moves) / len(moves) if moves else 0.0
        )
    return shares[0] - shares[1]


class PairEncoding(TeamEncoding):
    """Only ``te_pair_mean`` of :class:`TeamEncoding` (``tr_p1`` replaces
    ``te_p1_team``)."""

    @property
    def output_names(self) -> tuple[str, ...]:
        return ("te_pair_mean",)

    def compute(self, battle: BattleContext) -> Mapping[str, float]:
        return {"te_pair_mean": super().compute(battle)["te_pair_mean"]}

