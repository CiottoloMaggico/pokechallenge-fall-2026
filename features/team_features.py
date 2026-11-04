"""Strength of what is left of each team.

* ``hpw_spd`` / ``hpw_def``: special / physical bulk left, as a share
  difference. In the Gen 2 formula the hits a Pokémon can take grow
  with ``HP x defence``, so each alive Pokémon counts with its current
  HP (in points) times its real SpD / Def at its level; P2's unrevealed
  slots count as the mean bulk of its revealed Pokémon at full HP.
* ``snorlax_diff``: HP of P1's Snorlax minus P2's (0 if absent or
  fainted), in [-1, 1]. Snorlax is the pillar of most Gen 2 OU teams;
  while P2 has not shown it, it counts as at full HP with the
  probability that it is among P2's unrevealed slots.
* ``bst_hp_left``: base stat total weighted by the HP left, summed over
  alive Pokémon (P2's unrevealed slots as the mean of its revealed
  Pokémon at full HP), as a share difference in [-1, 1].
* ``hidden_bst``: how strong P2's unrevealed Pokémon probably are. A
  label-free model learned from P1 teams (which species appear
  together) completes P2's revealed team; the feature is minus the mean
  base stat total of the guessed Pokémon over 600 (0 when P2 has
  revealed everyone), so stronger hidden threats are more negative.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from domain.battle import BattleContext, PlayerState
from feature_set import Feature
from features.turn_features import AtTurn, check_turn

MAX_BST = 600
SNORLAX = "snorlax"
SNORLAX_IN_TEAM = 0.85  # share of teams with a Snorlax, for unrevealed slots
TEAM_CANDIDATES = 120  # most frequent species considered when guessing


def hp_weighted(player: PlayerState, stat: str) -> float:
    """Sum of ``current HP x stat`` over alive Pokémon, unrevealed as mean."""
    known = list(player.team.values())
    total = sum(p.hp * p.stat(stat) for p in known if not p.fainted)
    unrevealed = player.team_size - len(known)
    if unrevealed and known:
        total += unrevealed * sum(p.max_hp * p.stat(stat) for p in known) / len(known)
    return total


def hpw_spd(context: BattleContext) -> float:
    return _hpw_share(context, "spd")


def hpw_def(context: BattleContext) -> float:
    """Physical bulk left (HP x Defense), as a share difference."""
    return _hpw_share(context, "def")


def snorlax_diff(context: BattleContext) -> float:
    """``snorlax_diff`` (see the module docstring)."""
    p1, p2 = context.side("p1"), context.side("p2")
    return _snorlax_hp(p1) - _snorlax_hp(p2, guess_hidden=True)


def _snorlax_hp(player: PlayerState, guess_hidden: bool = False) -> float:
    snorlax = player.team.get(SNORLAX)
    if snorlax is not None:
        return 0.0 if snorlax.fainted else snorlax.hp_pct
    if not guess_hidden:
        return 0.0
    unrevealed = player.team_size - len(player.team)
    return SNORLAX_IN_TEAM * unrevealed / (player.team_size - 1)


def bst_hp_left(context: BattleContext) -> float:
    """``bst_hp_left`` (see the module docstring)."""
    a, b = (_strength(context.side(s)) for s in ("p1", "p2"))
    return (a - b) / (a + b) if a + b else 0.0


def _strength(player: PlayerState) -> float:
    known = list(player.team.values())
    total = sum(p.hp_pct * p.info.base_stat_total for p in known if not p.fainted)
    unrevealed = player.team_size - len(known)
    if unrevealed and known:
        mean = sum(p.info.base_stat_total for p in known) / len(known)
        total += unrevealed * mean
    return total


def _hpw_share(context: BattleContext, stat: str) -> float:
    a, b = (hp_weighted(context.side(s), stat) for s in ("p1", "p2"))
    return (a - b) / (a + b) if a + b else 0.0


class HiddenTeamStrength(Feature[BattleContext]):
    """``hidden_bst`` at the sampled turn (see the module docstring)."""

    requires_fit = True

    def __init__(self, turn: int | None = None) -> None:
        super().__init__("hidden_team_strength")
        self.turn = check_turn(turn)
        self.n_teams = 0
        self.frequency: dict[str, int] = {}
        self.together: dict[str, dict[str, int]] = {}

    @property
    def output_names(self) -> tuple[str, ...]:
        return ("hidden_bst",)

    def _fit(
        self, battles: Sequence[BattleContext], labels: Sequence[int] | None
    ) -> None:
        frequency: dict[str, int] = {}
        together: dict[str, dict[str, int]] = {}
        for context in battles:
            team = list(context.side("p1").team)
            for a in team:
                frequency[a] = frequency.get(a, 0) + 1
                row = together.setdefault(a, {})
                for b in team:
                    if a != b:
                        row[b] = row.get(b, 0) + 1
        self.n_teams, self.frequency, self.together = len(battles), frequency, together

    def guess(self, revealed: Sequence[str], team_size: int) -> list[str]:
        """Greedy most likely completion of a team (naive Bayes scores)."""
        team = list(revealed)
        candidates = sorted(self.frequency, key=lambda s: -self.frequency[s])
        candidates = candidates[:TEAM_CANDIDATES]
        while len(team) < team_size:
            best = max(
                (s for s in candidates if s not in team),
                key=lambda s: self._score(s, team),
                default=None,
            )
            if best is None:
                break
            team.append(best)
        n_revealed = len(revealed)
        return team[n_revealed:]

    def compute(self, battle: BattleContext) -> Mapping[str, float]:
        context = battle.upto(self.turn)
        p2 = context.side("p2")
        hidden = self.guess(list(p2.team), p2.team_size)
        if not hidden:
            return {"hidden_bst": 0.0}
        bst = [context.species(name).base_stat_total for name in hidden]
        return {"hidden_bst": -sum(bst) / len(bst) / MAX_BST}

    def _score(self, species: str, team: Sequence[str]) -> float:
        n = self.n_teams
        count = self.frequency[species]
        score = math.log(count / n)
        for member in team:
            pair = self.together.get(member, {}).get(species, 0)
            member_count = self.frequency.get(member, 0)
            score += math.log((pair + 1) / (member_count + 2))
            score -= math.log((count + 1) / (n + 2))
        return score


def team_features(turn: int | None = None) -> list[Feature[BattleContext]]:
    return [AtTurn("hpw_spd", hpw_spd, turn), HiddenTeamStrength(turn)]
