"""A quick simulation of the rest of the battle.

From the sampled turn, the alive Pokémon of the two sides fight a
sequence of one-on-one duels until one side runs out (at most 15 duels):

* a duel is won by whoever knocks the other out first: hits needed with
  the Gen 2 damage formula and the best expected damage among the
  Pokémon's moves (moves seen so far, the typical attacking moves of its
  species learned from the training battles, and a power-80 STAB move
  per type as a floor), plus the turns it will still lose asleep or
  frozen (:func:`turns_lost`); speed breaks ties; the winner keeps the HP
  left after the hits it took, and is awake afterwards;
* the next Pokémon sent in is the one with the best matchup against the
  winner (faster first on equal matchups); the actives at the sampled
  turn start.

Outputs (positive when P1 comes out ahead):

* ``sim_seen_left``: Pokémon left to P1 minus Pokémon left to P2, over
  6, simulating only against P2's revealed Pokémon;
* ``sim_full_win``: +1 if P1 wins, -1 if P2 wins, 0 if undecided, also
  fighting P2's unrevealed Pokémon as guessed by ``HiddenTeamStrength``.

The move statistics and the team guesses are learned without labels.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace

from domain.battle import BattleContext, PokemonState
from domain.damage import Combatant, DamageCalculator
from domain.moves import SELF_KO_MOVES, DamageKind, Move, damage_kind
from domain.typechart import TypeChart
from features.core.team import HiddenTeamStrength

OUTPUTS = ("sim_seen_left", "sim_full_win")
MAX_DUELS = 15
MAX_HITS = 30
MIN_HP_LEFT = 0.01
TEAM_SIZE = 6
TYPICAL_MOVES = 4
MIN_MOVE_COUNT = 3
MIN_MOVE_SHARE = 0.04
FALLBACK_POWER = 80
# turns a Pokémon spends asleep or frozen, as seen in the training records:
# Rest and other sleep last about 3 turns, a freeze about 2
SLEEP_TURNS = 3
FREEZE_TURNS = 2
SLEEP_MOVES = frozenset({"sleeptalk"})
PHYSICAL_TYPES = frozenset(
    {
        "normal",
        "fighting",
        "flying",
        "ground",
        "rock",
        "bug",
        "ghost",
        "poison",
        "steel",
    }
)


@dataclass(frozen=True)
class Fighter:
    name: str
    hp: float
    combatant: Combatant
    moves: tuple[Move, ...]
    delay: int = 0  # turns before it can attack (asleep or frozen)


def turns_lost(pokemon: PokemonState) -> int:
    """Turns ``pokemon`` will still spend unable to attack.

    Asleep without Sleep Talk: about 3 turns from falling asleep; frozen:
    about 2 (``status_turns`` counts the turns already spent so).
    """
    if pokemon.status == "slp" and not SLEEP_MOVES & set(pokemon.known_moves):
        return max(0, SLEEP_TURNS - pokemon.status_turns)
    if pokemon.status == "frz":
        return max(0, FREEZE_TURNS - pokemon.status_turns)
    return 0


def is_attack(move: Move) -> bool:
    """A move that deals predictable damage and keeps the user standing.

    Explosion and Self-Destruct are left out: in a duel they trade the
    user for the target instead of winning it.
    """
    if move.name in SELF_KO_MOVES:
        return False
    return damage_kind(move) not in (DamageKind.NONE, DamageKind.UNPREDICTABLE)


def stab_moves(types: Sequence[str]) -> tuple[Move, ...]:
    """One power-80 move per type, used when no move is known."""
    return tuple(
        Move(
            f"stab_{t}",
            t,
            "physical" if t in PHYSICAL_TYPES else "special",
            FALLBACK_POWER,
            1.0,
            0,
        )
        for t in types
    )


def make_fighter(
    pokemon: PokemonState,
    moves: Iterable[Move],
    boosts: Mapping[str, int] | None = None,
) -> Fighter:
    """A duel participant with its attacks (first name wins) and a STAB floor."""
    attacks: dict[str, Move] = {}
    for move in moves:
        if is_attack(move):
            attacks.setdefault(move.name, move)
    for move in stab_moves(pokemon.info.types):
        attacks.setdefault(move.name, move)
    return Fighter(
        pokemon.name,
        pokemon.hp_pct,
        Combatant.from_state(pokemon, boosts),
        tuple(attacks.values()),
        turns_lost(pokemon),
    )


class DuelSimulator:
    """Plays the one-on-one duels of a battle (see the module docstring)."""

    def __init__(self, chart: TypeChart) -> None:
        self.chart = chart

    def simulate(
        self, team_a: list[Fighter], team_b: list[Fighter], actives: tuple[str, str]
    ) -> tuple[list[Fighter], list[Fighter]]:
        a_side, b_side = list(team_a), list(team_b)
        damage: dict[tuple[str, str], tuple[float, float]] = {}
        ia = next((i for i, f in enumerate(a_side) if f.name == actives[0]), 0)
        ib = next((i for i, f in enumerate(b_side) if f.name == actives[1]), 0)
        for _ in range(MAX_DUELS):
            if not a_side or not b_side:
                break
            a, b = a_side[ia], b_side[ib]
            a_wins, hp = self._duel(a, b, damage)
            if a_wins:
                a_side[ia] = replace(a, hp=hp, delay=0)
                b_side.pop(ib)
                if b_side:
                    ib = max(
                        range(len(b_side)),
                        key=lambda j: self._margin(b_side[j], a_side[ia], damage),
                    )
            else:
                b_side[ib] = winner = replace(b, hp=hp, delay=0)
                a_side.pop(ia)
                if a_side:
                    ia = max(
                        range(len(a_side)),
                        key=lambda i: self._margin(a_side[i], winner, damage),
                    )
        return a_side, b_side

    def _damage(
        self, a: Fighter, b: Fighter, cache: dict[tuple[str, str], tuple[float, float]]
    ) -> tuple[float, float]:
        key = (a.name, b.name)
        if key not in cache:
            cache[key] = (self._best(a, b), self._best(b, a))
        return cache[key]

    def _best(self, attacker: Fighter, defender: Fighter) -> float:
        best = 0.0
        damage = DamageCalculator(self.chart)
        for move in attacker.moves:
            mean = damage.expected(attacker.combatant, defender.combatant, move, with_accuracy=True)
            if mean is not None:
                best = max(best, mean / defender.combatant.max_hp)
        return best

    def _hits(
        self, a: Fighter, b: Fighter, cache: dict[tuple[str, str], tuple[float, float]]
    ) -> tuple[int, int, float, float]:
        """Turns each needs to knock the other out (delay included)."""
        to_b, to_a = self._damage(a, b, cache)
        hits_a = min(MAX_HITS, math.ceil(b.hp / to_b)) if to_b > 0 else MAX_HITS
        hits_b = min(MAX_HITS, math.ceil(a.hp / to_a)) if to_a > 0 else MAX_HITS
        return hits_a + a.delay, hits_b + b.delay, to_b, to_a

    def _duel(
        self, a: Fighter, b: Fighter, cache: dict[tuple[str, str], tuple[float, float]]
    ) -> tuple[bool, float]:
        turns_a, turns_b, to_b, to_a = self._hits(a, b, cache)
        a_fast = _faster(a, b)
        if turns_a < turns_b or (turns_a == turns_b and a_fast):
            taken = _hits_taken(turns_a, a_fast, b.delay)
            return True, max(MIN_HP_LEFT, a.hp - taken * to_a)
        taken = _hits_taken(turns_b, not a_fast, a.delay)
        return False, max(MIN_HP_LEFT, b.hp - taken * to_b)

    def _margin(
        self, a: Fighter, b: Fighter, cache: dict[tuple[str, str], tuple[float, float]]
    ) -> tuple[float, bool]:
        """How well ``a`` matches up against ``b``; faster wins ties."""
        turns_a, turns_b, _, _ = self._hits(a, b, cache)
        return math.log(turns_b / turns_a), _faster(a, b)


class BattleSimulation(HiddenTeamStrength):
    """``sim_seen_left`` and ``sim_full_win`` at the sampled turn."""

    def __init__(
        self,
        chart: TypeChart,
        turn: int | None = None,
        outputs: Sequence[str] = OUTPUTS,
    ) -> None:
        super().__init__(turn)
        unknown = set(outputs) - set(OUTPUTS)
        if not outputs or unknown:
            raise ValueError(f"outputs must be a non-empty subset of {OUTPUTS}")
        self._feature_id = "battle_simulation"
        self.chart = chart
        self.outputs = tuple(outputs)
        self.typical: dict[str, list[Move]] = {}
        self.simulator = DuelSimulator(chart)

    @property
    def output_names(self) -> tuple[str, ...]:
        return self.outputs

    def _fit(self, battles: Sequence[BattleContext], labels: Sequence[int] | None) -> None:
        super()._fit(battles, labels)
        counts: dict[str, Counter[str]] = {}
        known: dict[str, Move] = {}
        for context in battles:
            for turn in context.history:
                for side in (turn.p1, turn.p2):
                    if side.move is not None and is_attack(side.move):
                        counts.setdefault(side.active, Counter())[side.move.name] += 1
                        known[side.move.name] = side.move
        self.typical = {
            species: [
                known[name]
                for name, n in counter.most_common(TYPICAL_MOVES)
                if n >= MIN_MOVE_COUNT and n / sum(counter.values()) >= MIN_MOVE_SHARE
            ]
            for species, counter in counts.items()
        }

    def _fighter(self, pokemon: PokemonState, boosts: Mapping[str, int] | None = None) -> Fighter:
        """Known and typical attacks of the species, then a STAB floor."""
        typical = self.typical.get(pokemon.name, [])
        return make_fighter(pokemon, (*pokemon.known_moves.values(), *typical), boosts)

    def compute(self, battle: BattleContext) -> Mapping[str, float]:
        context = battle.at_turn(self.turn)
        p1, p2 = context.side("p1"), context.side("p2")
        team_a = [
            self._fighter(p, p1.active.boosts if p is p1.active.pokemon else None)
            for p in p1.team.values()
            if not p.fainted
        ]
        seen_b = [
            self._fighter(p, p2.active.boosts if p is p2.active.pokemon else None)
            for p in p2.team.values()
            if not p.fainted
        ]
        hidden = self.guess(list(p2.team), p2.team_size)
        full_b = seen_b + [self._fighter(context.new_pokemon(name)) for name in hidden]
        actives = (p1.active.pokemon.name, p2.active.pokemon.name)
        values: dict[str, float] = {}
        if "sim_seen_left" in self.outputs:
            values["sim_seen_left"] = math.nan
            if not stats_missing(team_a, seen_b):
                left_a, left_b = self.simulator.simulate(team_a, seen_b, actives)
                values["sim_seen_left"] = (len(left_a) - len(left_b)) / TEAM_SIZE
        if "sim_full_win" in self.outputs:
            values["sim_full_win"] = math.nan
            if not stats_missing(team_a, full_b):
                win_a, win_b = self.simulator.simulate(team_a, full_b, actives)
                values["sim_full_win"] = outcome(win_a, win_b)
        return values


def stats_missing(*teams: Iterable[Fighter]) -> bool:
    """A fighter has no stats: the duels would treat its damage as 0, so they are not played."""
    return not all(f.combatant.species.has_stats for team in teams for f in team)


def _faster(a: Fighter, b: Fighter) -> bool:
    return a.combatant.stat("spe") > b.combatant.stat("spe")


def _hits_taken(winner_turns: int, winner_fast: bool, loser_delay: int) -> int:
    """Hits the loser lands before the winner's last hit."""
    turns = winner_turns - 1 if winner_fast else winner_turns
    return max(0, turns - loser_delay)


def outcome(left_a: Sequence[Fighter], left_b: Sequence[Fighter]) -> float:
    """+1 if only P1 has Pokémon left, -1 if only P2, 0 otherwise."""
    if left_a and not left_b:
        return 1.0
    if left_b and not left_a:
        return -1.0
    return 0.0
