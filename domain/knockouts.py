"""Switches that may hide a knockout, described for a knockout detector.

In the records a fainted P1 Pokémon appears with status ``fnt`` and 0 HP,
but a fainted P2 Pokémon usually does not: when only P2's Pokémon faints,
the turn already shows P2's replacement, which looks like a voluntary
switch. ``fnt`` appears for P2 only when both sides faint in the same
turn (Explosion, Destiny Bond, Perish Song).

P1's faints are always recorded, so they can teach a detector what a
hidden faint looks like: :func:`transitions` describes every change of
active Pokémon of one side, and for P1 it skips the row that shows the
faint, so a P1 knockout looks exactly like a hidden P2 one (the leaver
at the row before, the replacement at the next). Each transition gets
the same features on both sides:

* the leaver before the turn: HP, status, boosts, effects, base bulk;
* the incoming Pokémon: first appearance, HP and the HP it lost compared
  with when it was last seen (a voluntary switch-in takes the attack, a
  post-faint replacement enters at the end of the turn), whether the
  record shows a move of its own, base stat total, Spikes on its side;
* the opponent: whether it stayed in and the HP it lost, its move (new
  this turn or repeated, power, status move, phazing), the move's type
  effectiveness and expected damage on the leaver, and the leaver's HP
  minus that damage.

The detector itself (``knockout_detector.KnockoutDetector``) learns
P(faint | transition) on P1 and scores P2's transitions.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from domain.damage import Combatant, expected_damage
from domain.moves import Move
from domain.pokedex import TYPES, PokemonGen2
from domain.typechart import TypeChart

FAINTED = "fnt"
NO_EFFECT = "noeffect"
SPIKES = "spikes"
RAPID_SPIN = "rapidspin"
PHAZING_MOVES = frozenset({"roar", "whirlwind"})
MAX_STAGE = 6
MISSING = float("nan")
SIDES = ("p1", "p2")

Turn = Mapping[str, Any]
SpeciesLookup = Callable[[str], PokemonGen2 | None]

FEATURES = (
    "turn",
    "x_hp",
    "x_slp",
    "x_par",
    "x_frz",
    "x_psn",
    "x_brn",
    "x_boost",
    "x_trapped",
    "x_perish",
    "x_sub",
    "x_seed",
    "x_curse",
    "x_conf",
    "x_dbond",
    "x_bhp",
    "x_bdef",
    "x_bspd",
    "y_first",
    "y_hp",
    "y_prev_hp",
    "y_drop",
    "y_mv_none",
    "y_mv_new",
    "y_bst",
    "spikes",
    "a_same",
    "a_drop",
    "a_mv_changed",
    "a_bp",
    "a_status_mv",
    "a_eff",
    "a_dmg",
    "x_hp_minus_dmg",
    "a_phaze",
    "a_boost",
    "a_disabled",
)


@dataclass(frozen=True)
class Transition:
    """One change of active Pokémon of a side.

    ``index`` is the timeline row where the replacement appears and
    ``leaver`` the Pokémon that left. ``fainted`` is True for a recorded
    faint (always for P1's training examples, rarely for P2) and
    ``final`` tells whether the leaver never shows up again.
    """

    index: int
    leaver: str
    fainted: bool
    final: bool
    features: Mapping[str, float] = field(default_factory=dict)


class _Side:
    """One side's column of the timeline."""

    def __init__(self, timeline: Sequence[Turn], side: str) -> None:
        states = [t[f"{side}_pokemon_state"] for t in timeline]
        self.names = [s["name"] for s in states]
        self.hp = [float(s["hp_pct"]) for s in states]
        self.status = [s["status"] for s in states]
        self.effects = [frozenset(s["effects"]) - {NO_EFFECT} for s in states]
        self.boosts = [dict(s.get("boosts") or {}) for s in states]
        self.moves: list[Turn | None] = [t[f"{side}_move_details"] for t in timeline]
        self.last_index = {name: i for i, name in enumerate(self.names)}
        self.fresh = self._fresh_moves()

    def _fresh_moves(self) -> list[Turn | None]:
        """Moves really used at each row (the records repeat old ones)."""
        fresh: list[Turn | None] = [None] * len(self.names)
        last: dict[str, str] = {}
        for i, (name, move) in enumerate(zip(self.names, self.moves, strict=True)):
            if move is None:
                continue
            if i > 0 and self.names[i - 1] == name:
                previous = self.moves[i - 1]
                reference = previous["name"] if previous else None
            else:
                reference = last.get(name)
            if reference != move["name"]:
                fresh[i] = move
            last[name] = move["name"]
        return fresh


def transitions(
    timeline: Sequence[Turn],
    side: str,
    species: SpeciesLookup,
    chart: TypeChart,
) -> list[Transition]:
    """Every change of active Pokémon of ``side``, with its features.

    For P1 a recorded faint is described from the row before the faint
    row, so it looks like a hidden P2 faint; double faints (both sides
    ``fnt``) carry no features.
    """
    other = "p2" if side == "p1" else "p1"
    own, opp = _Side(timeline, side), _Side(timeline, other)
    spikes = _spikes_on(own, opp)
    last_hp: dict[str, float] = {}
    last_move: dict[str, str] = {}
    seen: set[str] = set()
    result = []
    for i in range(1, len(timeline)):
        j = i - 1
        leaver, incoming = own.names[j], own.names[i]
        last_hp[leaver] = own.hp[j]
        seen.add(leaver)
        if own.moves[j] is not None:
            last_move[leaver] = own.moves[j]["name"]
        if leaver == incoming:
            continue
        final = own.last_index[leaver] == j
        fainted = own.status[j] == FAINTED
        if fainted:
            usable = (
                side == "p1"
                and opp.status[j] != FAINTED
                and i >= 2
                and own.names[i - 2] == leaver
            )
            if not usable:
                result.append(Transition(i, leaver, True, final))
                continue
            before = i - 2
        else:
            before = i - 1
        context = _Context(own, opp, before, i, species, chart)
        features = context.features(last_hp, last_move, seen, spikes[j])
        result.append(Transition(i, leaver, fainted, final, features))
    return result


def _spikes_on(own: _Side, opp: _Side) -> list[bool]:
    """Whether ``own``'s side has Spikes after each row."""
    on, result = False, []
    for opp_move, own_move in zip(opp.fresh, own.fresh, strict=True):
        if opp_move is not None and opp_move["name"] == SPIKES:
            on = True
        if own_move is not None and own_move["name"] == RAPID_SPIN:
            on = False
        result.append(on)
    return result


@dataclass(frozen=True)
class _Context:
    own: _Side
    opp: _Side
    before: int
    index: int
    species: SpeciesLookup
    chart: TypeChart

    def features(
        self,
        last_hp: Mapping[str, float],
        last_move: Mapping[str, str],
        seen: set[str],
        spikes: bool,
    ) -> dict[str, float]:
        own, opp, b, i = self.own, self.opp, self.before, self.index
        leaver, incoming = own.names[b], own.names[i]
        x, y = self.species(leaver), self.species(incoming)
        attacker_same = opp.names[b] == opp.names[i]
        move = opp.moves[i]
        previous_move = opp.moves[b]
        status, effects = own.status[b], own.effects[b]
        y_prev = last_hp.get(incoming, 1.0)
        y_move = own.moves[i]
        damage = self._damage(move, x)
        effectiveness = (
            self.chart.effectiveness(move["type"], x.types)
            if move is not None and x is not None and move["type"] in TYPES
            else MISSING
        )
        values: dict[str, float | bool] = {
            "turn": i,
            "x_hp": own.hp[b],
            "x_slp": status == "slp",
            "x_par": status == "par",
            "x_frz": status == "frz",
            "x_psn": status in ("psn", "tox"),
            "x_brn": status == "brn",
            "x_boost": sum(own.boosts[b].values()),
            "x_trapped": "trapped" in effects,
            "x_perish": "perish" in effects,
            "x_sub": "substitute" in effects,
            "x_seed": "leechseed" in effects,
            "x_curse": "curse" in effects,
            "x_conf": "confusion" in effects,
            "x_dbond": "destinybond" in effects,
            "x_bhp": x.base_hp if x else MISSING,
            "x_bdef": x.base_def if x else MISSING,
            "x_bspd": x.base_spd if x else MISSING,
            "y_first": incoming not in seen,
            "y_hp": own.hp[i],
            "y_prev_hp": y_prev,
            "y_drop": y_prev - own.hp[i],
            "y_mv_none": y_move is None,
            "y_mv_new": y_move is not None
            and last_move.get(incoming) != y_move["name"],
            "y_bst": y.base_stat_total if y else MISSING,
            "spikes": spikes,
            "a_same": attacker_same,
            "a_drop": opp.hp[b] - opp.hp[i] if attacker_same else MISSING,
            "a_mv_changed": move is not None
            and (
                not attacker_same
                or previous_move is None
                or previous_move["name"] != move["name"]
            ),
            "a_bp": move["base_power"] if move else 0,
            "a_status_mv": move is not None and move["category"] == "status",
            "a_eff": effectiveness,
            "a_dmg": damage,
            "x_hp_minus_dmg": own.hp[b] - damage,
            "a_phaze": move is not None and move["name"] in PHAZING_MOVES,
            "a_boost": sum(opp.boosts[i].values()),
            "a_disabled": opp.status[i] in ("slp", "frz"),
        }
        return {name: float(values[name]) for name in FEATURES}

    def _damage(self, move: Turn | None, defender: PokemonGen2 | None) -> float:
        """Expected damage of the opponent's move on the leaver, in max HP."""
        attacker = self.species(self.opp.names[self.index])
        if move is None or move["category"] == "status":
            return 0.0
        if attacker is None or defender is None:
            return 0.0
        a = Combatant(
            attacker,
            boosts=_stages(self.opp.boosts[self.index]),
            status=_status(self.opp.status[self.index]),
        )
        d = Combatant(defender, boosts=_stages(self.own.boosts[self.before]))
        mean = expected_damage(
            a, d, Move.from_dict(dict(move)), self.chart, include_accuracy=True
        )
        return 0.0 if mean is None else mean / d.max_hp


def _stages(boosts: Mapping[str, int]) -> dict[str, int]:
    return {k: max(-MAX_STAGE, min(MAX_STAGE, v)) for k, v in boosts.items()}


def _status(status: str) -> str:
    return "nostatus" if status == FAINTED else status
