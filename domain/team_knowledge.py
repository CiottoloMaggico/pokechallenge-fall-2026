"""Which moves each Pokémon showed in each team, learned from past battles.

A battle reveals the moves of the Pokémon that were used, and the teams of
both players when they are complete (P1 always; P2 once all six of its
Pokémon have been shown). The archive counts these observations over the
training battles, keyed by the team (a frozenset of species names), the
species and the move:

* ``team`` counts the battles in which a complete team appears, once per
  player (a mirror match counts twice);
* ``team_moves`` and ``team_seen``: for a team and a species, how often
  each move was used, and in how many battles the Pokémon used any move;
* ``species_moves`` and ``species_seen``: the same over all teams, plus
  the Pokémon whose team is not complete; ``move_users`` and ``pokemon``
  give the overall frequency of each move.

Probabilities are smoothed toward the species, and the species toward the
overall frequency of the move, with smoothing ``a`` (default 3)::

    p(m | s)   = (n(s, m) + a * g(m)) / (N(s) + a)     g(m) = users(m) / pokemon
    p(m | T,s) = (n(T, s, m) + a * p(m | s)) / (N(T, s) + a)

The same team can belong to different players with different sets, so the
archive never merges moves across teams.

:meth:`TeamKnowledge.excluding` removes a battle, and every exact copy of
it (same content, any battle id), before a leave-one-out score. Only
integer counts are subtracted, so the view is identical to an archive
built without those battles.

This module does not import ``sklearn`` nor the feature layer.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from domain.battle import BattleContext, PlayerState, PokemonState
from domain.moves import Move
from domain.records import fingerprint

TEAM_SIZE = 6
SMOOTHING = 3.0
THRESHOLD = 0.5
MIN_TEAM_SHARE = 0.5
MAX_MOVES = 4

Team = frozenset[str]


@dataclass(frozen=True)
class Observation:
    """The moves one Pokémon showed in one battle.

    ``team`` is None when its player's team is not complete.
    """

    team: Team | None
    species: str
    moves: tuple[Move, ...]


@dataclass(frozen=True)
class Contribution:
    """What one battle adds to the archive."""

    teams: tuple[Team, ...]
    observations: tuple[Observation, ...]


@dataclass
class Counts:
    """Integer tallies of observations (see the module docstring)."""

    pokemon: int = 0
    team: Counter[Team] = field(default_factory=Counter)
    team_seen: Counter[tuple[Team, str]] = field(default_factory=Counter)
    team_moves: Counter[tuple[Team, str, str]] = field(default_factory=Counter)
    species_seen: Counter[str] = field(default_factory=Counter)
    species_moves: Counter[tuple[str, str]] = field(default_factory=Counter)
    move_users: Counter[str] = field(default_factory=Counter)

    def add(self, other: Counts) -> None:
        self.pokemon += other.pokemon
        self.team.update(other.team)
        self.team_seen.update(other.team_seen)
        self.team_moves.update(other.team_moves)
        self.species_seen.update(other.species_seen)
        self.species_moves.update(other.species_moves)
        self.move_users.update(other.move_users)


def complete_team(player: PlayerState) -> Team | None:
    """The player's six species names, or None while the team is incomplete."""
    names = frozenset(player.team)
    return names if len(names) == player.team_size else None


def observations(context: BattleContext) -> tuple[Observation, ...]:
    """One :class:`Observation` per Pokémon that used at least one move.

    Read from the final state of the battle, so pass a complete context.
    """
    result: list[Observation] = []
    for side in ("p1", "p2"):
        player = context.side(side)
        team = complete_team(player)
        for pokemon in _by_name(player.team.values()):
            moves = tuple(sorted(pokemon.known_moves.values(), key=_move_name))
            if moves:
                result.append(Observation(team, pokemon.name, moves))
    return tuple(result)


def contribution(context: BattleContext) -> Contribution:
    """The teams and observations of one complete battle."""
    teams = tuple(
        team
        for team in (complete_team(context.side(s)) for s in ("p1", "p2"))
        if team is not None
    )
    return Contribution(teams, observations(context))


def _tally(entry: Contribution, times: int) -> Counts:
    counts = Counts()
    for team in entry.teams:
        counts.team[team] += times
    for seen in entry.observations:
        counts.pokemon += times
        counts.species_seen[seen.species] += times
        if seen.team is not None:
            counts.team_seen[(seen.team, seen.species)] += times
        for move in seen.moves:
            counts.species_moves[(seen.species, move.name)] += times
            counts.move_users[move.name] += times
            if seen.team is not None:
                key = (seen.team, seen.species, move.name)
                counts.team_moves[key] += times
    return counts


class TeamKnowledge:
    """The archive built from a set of complete battles.

    Use :meth:`build`; then :meth:`excluding` gives the view to query.
    """

    def __init__(
        self,
        entries: Mapping[str, Contribution],
        times: Mapping[str, int],
        smoothing: float = SMOOTHING,
    ) -> None:
        if smoothing < 0:
            raise ValueError(f"smoothing must be >= 0, got {smoothing}")
        self.smoothing = smoothing
        self._entries = dict(entries)
        self._times = dict(times)
        self.counts = Counts()
        for digest in sorted(self._entries):
            self.counts.add(
                _tally(self._entries[digest], self._times[digest])
            )
        self.index, self.species_moves, self.catalog = _indexes(
            self.counts, self._entries
        )

    @classmethod
    def build(
        cls, contexts: Iterable[BattleContext], smoothing: float = SMOOTHING
    ) -> TeamKnowledge:
        entries: dict[str, Contribution] = {}
        times: Counter[str] = Counter()
        for context in contexts:
            digest = fingerprint(context.record)
            times[digest] += 1
            if digest not in entries:
                entries[digest] = contribution(context)
        return cls(entries, times, smoothing)

    def excluding(self, fingerprint: str) -> KnowledgeView:
        """The archive as if the battle with this fingerprint, and its copies,
        had never been seen. Unknown fingerprints give the whole archive."""
        if fingerprint not in self._entries:
            return KnowledgeView(self, Counts())
        removed = _tally(self._entries[fingerprint], self._times[fingerprint])
        return KnowledgeView(self, removed)


def _indexes(
    counts: Counts, entries: Mapping[str, Contribution]
) -> tuple[dict[str, frozenset[Team]], dict[str, tuple[str, ...]], dict[str, Move]]:
    teams_of: dict[str, set[Team]] = {}
    for team in counts.team:
        for species in team:
            teams_of.setdefault(species, set()).add(team)
    moves_of: dict[str, set[str]] = {}
    for species, move in counts.species_moves:
        moves_of.setdefault(species, set()).add(move)
    catalog: dict[str, Move] = {}
    for digest in sorted(entries):
        for seen in entries[digest].observations:
            for move in seen.moves:
                catalog.setdefault(move.name, move)
    index = {s: frozenset(teams) for s, teams in teams_of.items()}
    species_moves = {s: tuple(sorted(moves)) for s, moves in moves_of.items()}
    return index, species_moves, catalog


class KnowledgeView:
    """The archive minus some removed counts, queried as if it were rebuilt."""

    def __init__(self, knowledge: TeamKnowledge, removed: Counts) -> None:
        self._base = knowledge.counts
        self._removed = removed
        self._alpha = knowledge.smoothing
        self._knowledge = knowledge

    # ------------------------------------------------------------ counts
    def team_count(self, team: Team) -> int:
        return self._base.team[team] - self._removed.team[team]

    def _team_seen(self, team: Team, species: str) -> int:
        key = (team, species)
        return self._base.team_seen[key] - self._removed.team_seen[key]

    def _team_moves(self, team: Team, species: str, move: str) -> int:
        key = (team, species, move)
        return self._base.team_moves[key] - self._removed.team_moves[key]

    def _species_seen(self, species: str) -> int:
        return self._base.species_seen[species] - self._removed.species_seen[species]

    def _species_moves(self, species: str, move: str) -> int:
        key = (species, move)
        return self._base.species_moves[key] - self._removed.species_moves[key]

    def _users(self, move: str) -> int:
        return self._base.move_users[move] - self._removed.move_users[move]

    def _pokemon(self) -> int:
        return self._base.pokemon - self._removed.pokemon

    # ------------------------------------------------------------ teams
    def team_weights(self, revealed: Iterable[str]) -> dict[Team, float]:
        """Share of each archive team that could be the one of a player whose
        revealed Pokémon are ``revealed``.

        A complete team is its own answer (weight 1). Otherwise the candidates
        are the teams containing every revealed species, with a count above
        zero; the weight is the team's count over the candidates' total.
        """
        names = frozenset(revealed)
        if not names:
            return {}
        if len(names) == TEAM_SIZE:
            return {names: 1.0}
        candidates = self._candidates(names)
        total = sum(self.team_count(team) for team in candidates)
        if total == 0:
            return {}
        return {t: self.team_count(t) / total for t in candidates}

    def identify(
        self, revealed: Iterable[str], min_share: float = MIN_TEAM_SHARE
    ) -> Team | None:
        """The team of a player, or None if no candidate is dominant enough.

        Ties go to the lexicographically smallest sorted team.
        """
        names = frozenset(revealed)
        if len(names) == TEAM_SIZE:
            return names
        weights = self.team_weights(names)
        if not weights:
            return None
        best = min(weights, key=lambda t: (-weights[t], tuple(sorted(t))))
        return best if weights[best] >= min_share else None

    def _candidates(self, names: frozenset[str]) -> list[Team]:
        sets = [self._knowledge.index.get(s, frozenset()) for s in sorted(names)]
        common = frozenset.intersection(*sets)
        return sorted(
            (t for t in common if self.team_count(t) > 0),
            key=lambda t: tuple(sorted(t)),
        )

    # ------------------------------------------------------------ moves
    def move_probability(self, team: Team | None, species: str, move: str) -> float:
        """Probability that a Pokémon of ``species`` in ``team`` has ``move``.

        ``team=None`` (or a team the archive does not know) gives the species
        estimate. A species never seen has no moves: 0.
        """
        if self._species_seen(species) == 0:
            return 0.0
        prior = self._species_probability(species, move)
        if team is None:
            return prior
        hits = self._team_moves(team, species, move)
        seen = self._team_seen(team, species)
        return (hits + self._alpha * prior) / (seen + self._alpha)

    def _species_probability(self, species: str, move: str) -> float:
        share = self._users(move) / self._pokemon()
        hits = self._species_moves(species, move)
        seen = self._species_seen(species)
        return (hits + self._alpha * share) / (seen + self._alpha)

    def move_probabilities(self, team: Team | None, species: str) -> dict[str, float]:
        """Probability of each move the archive has seen for ``species``."""
        return {
            move: self.move_probability(team, species, move)
            for move in self._seen_moves(species)
        }

    def _seen_moves(self, species: str) -> list[str]:
        """Moves of ``species`` with a count above zero in this view."""
        return [
            move
            for move in self._knowledge.species_moves.get(species, ())
            if self._species_moves(species, move) > 0
        ]

    def likely_moves(
        self,
        team: Team | None,
        pokemon: PokemonState,
        threshold: float = THRESHOLD,
        limit: int = MAX_MOVES,
    ) -> tuple[Move, ...]:
        """The moves to assume for ``pokemon``: every move revealed in this
        battle, plus the archive moves with probability at least
        ``threshold``, most probable first (ties by name), up to ``limit``."""
        revealed = _by_name(pokemon.known_moves.values())
        names = {move.name for move in revealed}
        room = max(0, limit - len(revealed))
        scored = sorted(
            (-p, move)
            for move in self._seen_moves(pokemon.name)
            if move not in names
            for p in [self.move_probability(team, pokemon.name, move)]
            if p >= threshold
        )
        added = [self._knowledge.catalog[move] for _, move in scored[:room]]
        return (*revealed, *added)


def _by_name(items: Iterable[PokemonState | Move]) -> list:
    return sorted(items, key=_name)


def _name(item: PokemonState | Move) -> str:
    return item.name


def _move_name(move: Move) -> str:
    return move.name
