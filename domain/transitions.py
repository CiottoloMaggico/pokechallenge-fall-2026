"""Switches that may hide a knockout, described for a knockout detector.

In the records a fainted P1 Pokémon appears with status ``fnt`` and 0 HP,
but a fainted P2 Pokémon usually does not: when only P2's Pokémon faints,
the turn already shows P2's replacement, which looks like a voluntary
switch. ``fnt`` appears for P2 only when both sides faint in the same
turn (Explosion, Destiny Bond, Perish Song).

P1's faints are always recorded, so they can teach a detector what a
hidden faint looks like: :class:`TransitionExtractor` describes every change
of active Pokémon of one side, and for P1 it skips the row that shows the
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

Six more features weigh the physical evidence (``_Context._evidence``):
the Speed of the leaver against the attacker, the worst critical roll
of the opposing move on the leaver, and the damage that move should
have done to the incoming Pokémon had it switched in voluntarily
compared with the HP it really lost (an immune switch-in, e.g. a Ground
type into Thunder, loses nothing without being a replacement). On P1's
recorded faints they raise the out-of-fold AUC from 0.982 to 0.987 and
lower the log loss from 0.150 to 0.128; most of the gain comes from the
incoming Pokémon's expected damage. Seven more
(:mod:`domain.transition_mechanics`) add the Gen 2 field and end-of-turn
rules (screens, weather, residual damage, priority, Toxic and Perish Song
counters) and ``a_drop_any``, the damage taken by whichever opposing
Pokémon was on the field during the change: AUC 0.9915, log loss 0.099,
precision 0.935, recall 0.922 (50 features).

Reading the opponent of a recorded P1 faint at the faint row instead of
the replacement row was tested and rejected: it fits P1 better (log loss
0.078 vs 0.098) but repairs only ~7,000 P2 switches, while symmetry with
P1's recorded faints expects ~10,100 detectable hidden P2 faints; it also
raises the P1-vs-P2 adversarial AUC from 0.585 to 0.603.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from domain.damage import Combatant, DamageCalculator
from domain.moves import Move
from domain.pokedex import TYPES, PokemonGen2, SpeciesLookup
from domain.transition_mechanics import (
    FAINTED,
    MISSING,
    Turn,
    attacker_status,
    field_tracker,
    mechanics,
    stages,
    status_counters,
)

NO_EFFECT = "noeffect"
SPIKES = "spikes"
RAPID_SPIN = "rapidspin"
PHAZING_MOVES = frozenset({"roar", "whirlwind"})
STATUS_CODES = {"par": 1, "slp": 2, "frz": 3, "psn": 4, "tox": 5, "brn": 6}

ALL_COLUMNS = (
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
    # physical evidence added in compact_v5 (see the module docstring)
    "x_spe_ratio",
    "a_dmg_max",
    "x_hp_minus_max",
    "a_eff_in",
    "a_dmg_in",
    "drop_gap",
    # Gen 2 field and end-of-turn mechanics, and damage over the change
    "a_dmg_field",
    "x_residual",
    "x_hp_after",
    "a_priority",
    "x_perish_turns",
    "x_tox_turns",
    "a_drop_any",
    # priority of the opposing move, -2 for Roar / Whirlwind (forced switch)
    "a_prio_phaze",
    # lossless codes of the flags above (categorical for the detector)
    "x_status",
    "y_code",
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
    features: Mapping[str, float] = field(default_factory=dict[str, float])


class _Side:
    """One side's column of the timeline."""

    def __init__(self, timeline: Sequence[Turn], side: str) -> None:
        states = [t[f"{side}_pokemon_state"] for t in timeline]
        self.names: list[str] = [s["name"] for s in states]
        self.hp = [float(s["hp_pct"]) for s in states]
        self.status: list[str] = [s["status"] for s in states]
        self.effects = [frozenset[str](s["effects"]) - {NO_EFFECT} for s in states]
        self.boosts: list[dict[str, int]] = [dict(s.get("boosts") or {}) for s in states]
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


class TransitionExtractor:
    """Describes every change of active Pokémon of a side."""

    def __init__(self, pokedex: SpeciesLookup, damage: DamageCalculator) -> None:
        self._species = pokedex
        self._damage = damage

    def extract(self, record: Mapping[str, Any], side: Literal["p1", "p2"]) -> list[Transition]:
        """Every change of active Pokémon of ``side``, with its features.

        For P1 a recorded faint is described from the row before the faint
        row, so it looks like a hidden P2 faint; double faints (both sides
        ``fnt``) carry no features.
        """
        timeline: Sequence[Turn] = record["battle_timeline"]
        other = "p2" if side == "p1" else "p1"
        own, opp = _Side(timeline, side), _Side(timeline, other)
        fields = field_tracker(timeline)
        tox_turns, perish_turns = status_counters(timeline, side)
        spikes = _spikes_on(own, opp)
        last_hp: dict[str, float] = {}
        last_move: dict[str, str] = {}
        seen: set[str] = set()
        result: list[Transition] = []
        for i in range(1, len(timeline)):
            j = i - 1
            leaver, incoming = own.names[j], own.names[i]
            last_hp[leaver] = own.hp[j]
            seen.add(leaver)
            own_move = own.moves[j]
            if own_move is not None:
                last_move[leaver] = own_move["name"]
            if leaver == incoming:
                continue
            final = own.last_index[leaver] == j
            fainted = own.status[j] == FAINTED
            before = _row_before(own, opp, i, side)
            if before is None:
                result.append(Transition(i, leaver, True, final))
                continue
            context = _Context(own, opp, before, i, self._species, self._damage)
            features = context.features(last_hp, last_move, seen, spikes[j])
            features.update(
                mechanics(
                    timeline, side, before, i, self._species, self._damage,
                    fields[before], tox_turns[before], perish_turns[before],
                )
            )  # fmt: skip
            features["a_prio_phaze"] = features["a_priority"] - features["a_phaze"]
            result.append(Transition(i, leaver, fainted, final, features))
        return result

    def examples(self, record: Mapping[str, Any]) -> list[Transition]:
        """P1's described switches whose leaver never comes back: the detector's training rows,
        labelled by the recorded faint."""
        return [t for t in self.extract(record, "p1") if t.final and t.features]

    def candidates(self, record: Mapping[str, Any]) -> list[Transition]:
        """P2's described switches whose leaver never comes back and is not a recorded faint:
        the switches that may hide a knockout."""
        return [t for t in self.extract(record, "p2") if t.final and not t.fainted and t.features]


def _row_before(own: _Side, opp: _Side, i: int, side: str) -> int | None:
    """Row that describes the leaver of row ``i``; None when it cannot be described."""
    j = i - 1
    if own.status[j] != FAINTED:
        return j
    usable = (
        side == "p1" and opp.status[j] != FAINTED and i >= 2 and own.names[i - 2] == own.names[j]
    )
    return i - 2 if usable else None


def _spikes_on(own: _Side, opp: _Side) -> list[bool]:
    """Whether ``own``'s side has Spikes after each row."""
    on = False
    result: list[bool] = []
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
    damage: DamageCalculator

    def features(
        self,
        last_hp: Mapping[str, float],
        last_move: Mapping[str, str],
        seen: set[str],
        spikes: bool,
    ) -> dict[str, float]:
        """The columns read from the two sides' rows; ``mechanics`` adds the others."""
        own, b, i = self.own, self.before, self.index
        x, y = self.species(own.names[b]), self.species(own.names[i])
        y_prev = last_hp.get(own.names[i], 1.0)
        values: dict[str, float | bool] = {
            "turn": i,
            **self._leaver_values(x),
            **self._incoming_values(y, y_prev, last_move, seen),
            "spikes": spikes,
            **self._opponent_values(x),
            **self._evidence(self.opp.moves[i], x, y, own.hp[b], own.hp[i], y_prev),
        }
        return {name: float(values[name]) for name in ALL_COLUMNS if name in values}

    def _leaver_values(self, x: PokemonGen2 | None) -> dict[str, float | bool]:
        own, b = self.own, self.before
        status, effects = own.status[b], own.effects[b]
        return {
            "x_hp": own.hp[b],
            "x_slp": status == "slp",
            "x_par": status == "par",
            "x_frz": status == "frz",
            "x_psn": status in ("psn", "tox"),
            "x_brn": status == "brn",
            "x_status": STATUS_CODES.get(status, 0),
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
        }

    def _incoming_values(
        self,
        y: PokemonGen2 | None,
        y_prev: float,
        last_move: Mapping[str, str],
        seen: set[str],
    ) -> dict[str, float | bool]:
        own, i = self.own, self.index
        incoming, y_move = own.names[i], own.moves[i]
        return {
            "y_first": incoming not in seen,
            "y_hp": own.hp[i],
            "y_prev_hp": y_prev,
            "y_drop": y_prev - own.hp[i],
            "y_mv_none": y_move is None,
            "y_mv_new": y_move is not None and last_move.get(incoming) != y_move["name"],
            "y_bst": y.base_stat_total if y else MISSING,
            # first appearance x (no move / new move / repeated move)
            "y_code": 3 * (incoming not in seen)
            + (0 if y_move is None else 1 if last_move.get(incoming) != y_move["name"] else 2),
        }

    def _opponent_values(self, x: PokemonGen2 | None) -> dict[str, float | bool]:
        own, opp, b, i = self.own, self.opp, self.before, self.index
        attacker_same = opp.names[b] == opp.names[i]
        move, previous_move = opp.moves[i], opp.moves[b]
        damage = self._expected_on_leaver(move, x)
        effectiveness = (
            self.damage.chart.effectiveness(move["type"], x.types)
            if move is not None and x is not None and move["type"] in TYPES
            else MISSING
        )
        return {
            "a_same": attacker_same,
            "a_drop": opp.hp[b] - opp.hp[i] if attacker_same else MISSING,
            "a_mv_changed": move is not None
            and (
                not attacker_same or previous_move is None or previous_move["name"] != move["name"]
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

    def _evidence(
        self,
        move: Turn | None,
        leaver: PokemonGen2 | None,
        incoming: PokemonGen2 | None,
        x_hp: float,
        y_hp: float,
        y_prev: float,
    ) -> dict[str, float]:
        """Speed, worst-case damage on the leaver, and what the move should
        have done to the incoming Pokémon if it had switched in voluntarily.

        * ``x_spe_ratio``: log of the leaver's Speed over the attacker's
          (stages and paralysis included);
        * ``a_dmg_max`` / ``x_hp_minus_max``: highest critical roll of the
          opposing move on the leaver, in max HP, and the leaver's HP minus it;
        * ``a_eff_in`` / ``a_dmg_in``: type effectiveness and expected damage
          of the opposing move on the incoming Pokémon;
        * ``drop_gap``: ``a_dmg_in`` minus the HP the incoming Pokémon actually
          lost. A voluntary switch-in takes the hit (gap near 0, or 0 because
          it is immune, e.g. a Ground type into Thunder); a replacement sent
          in after a faint enters at the end of the turn (gap = full damage).
        """
        attacker = self.species(self.opp.names[self.index])
        values = {
            "x_spe_ratio": MISSING,
            "a_dmg_max": 0.0,
            "x_hp_minus_max": x_hp,
            "a_eff_in": MISSING,
            "a_dmg_in": 0.0,
            "drop_gap": 0.0,
        }
        attacks = move is not None and move["category"] != "status"
        if attacks and (attacker is None or leaver is None):
            values["a_dmg_max"] = values["x_hp_minus_max"] = MISSING
        if attacks and (attacker is None or incoming is None):
            values["a_dmg_in"] = values["drop_gap"] = MISSING
        if attacker is None:
            return values
        a = self._attacker(attacker)
        d = None
        if leaver is not None:
            d = Combatant(
                leaver,
                boosts=stages(self.own.boosts[self.before]),
                status=attacker_status(self.own.status[self.before]),
            )
            values["x_spe_ratio"] = math.log(d.stat("spe") / a.stat("spe"))
        if move is None or move["category"] == "status":
            return values
        parsed = Move.from_details(move)
        if d is not None:
            worst = self.damage.roll(a, d, parsed, crit=True)
            if worst is not None:
                values["a_dmg_max"] = worst.max / d.max_hp
                values["x_hp_minus_max"] = x_hp - values["a_dmg_max"]
        if incoming is not None:
            if move["type"] in TYPES:
                values["a_eff_in"] = self.damage.chart.effectiveness(move["type"], incoming.types)
            target = Combatant(incoming)
            mean = self.damage.expected(a, target, parsed, with_accuracy=True)
            if mean is not None:
                values["a_dmg_in"] = min(1.0, mean / target.max_hp)
                values["drop_gap"] = values["a_dmg_in"] - (y_prev - y_hp)
        return values

    def _attacker(self, attacker: PokemonGen2) -> Combatant:
        return Combatant(
            attacker,
            boosts=stages(self.opp.boosts[self.index]),
            status=attacker_status(self.opp.status[self.index]),
        )

    def _expected_on_leaver(self, move: Turn | None, defender: PokemonGen2 | None) -> float:
        """Expected damage of the opponent's move on the leaver, in max HP."""
        attacker = self.species(self.opp.names[self.index])
        if move is None or move["category"] == "status":
            return 0.0
        if attacker is None or defender is None:
            return MISSING
        d = Combatant(defender, boosts=stages(self.own.boosts[self.before]))
        mean = self.damage.expected(
            self._attacker(attacker), d, Move.from_details(move), with_accuracy=True
        )
        return 0.0 if mean is None else mean / d.max_hp
