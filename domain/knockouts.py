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

Six more features weigh the physical evidence (``_Context._evidence``):
the Speed of the leaver against the attacker, the worst critical roll
of the opposing move on the leaver, and the damage that move should
have done to the incoming Pokémon had it switched in voluntarily
compared with the HP it really lost (an immune switch-in, e.g. a Ground
type into Thunder, loses nothing without being a replacement). On P1's
recorded faints they raise the out-of-fold AUC from 0.982 to 0.987 and
lower the log loss from 0.150 to 0.128; most of the gain comes from the
incoming Pokémon's expected damage. Seven more (``_mechanics``) add the
Gen 2 field and end-of-turn rules (screens, weather, residual damage,
priority, Toxic and Perish Song counters) and ``a_drop_any``, the damage
taken by whichever opposing Pokémon was on the field during the change:
AUC 0.9915, log loss 0.099, precision 0.935, recall 0.922 (50 features).

The detector itself (``knockout_detector.KnockoutDetector``) learns
P(faint | transition) on P1 and scores P2's transitions.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import math

from domain.damage import Combatant, Field, calc_damage, expected_damage
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
    fields = _field_tracker(timeline)
    tox_turns, perish_turns = _status_counters(timeline, side)
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
        features.update(
            _mechanics(
                timeline, side, before, i, species, chart,
                fields[before], tox_turns[before], perish_turns[before],
            )
        )
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
            **self._evidence(move, x, y, own.hp[b], own.hp[i], y_prev),
        }
        # the Gen 2 mechanics features are added by ``transitions`` (``_mechanics``)
        return {name: float(values[name]) for name in FEATURES if name in values}

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
        if attacker is None:
            return values
        a = Combatant(
            attacker,
            boosts=_stages(self.opp.boosts[self.index]),
            status=_status(self.opp.status[self.index]),
        )
        if leaver is not None:
            d = Combatant(
                leaver,
                boosts=_stages(self.own.boosts[self.before]),
                status=_status(self.own.status[self.before]),
            )
            values["x_spe_ratio"] = math.log(d.stat("spe") / a.stat("spe"))
        if move is None or move["category"] == "status":
            return values
        parsed = Move.from_dict(dict(move))
        if leaver is not None:
            worst = calc_damage(a, d, parsed, self.chart, crit=True)
            if worst is not None:
                values["a_dmg_max"] = worst.max / d.max_hp
                values["x_hp_minus_max"] = x_hp - values["a_dmg_max"]
        if incoming is not None:
            if move["type"] in TYPES:
                values["a_eff_in"] = self.chart.effectiveness(move["type"], incoming.types)
            target = Combatant(incoming)
            mean = expected_damage(a, target, parsed, self.chart, include_accuracy=True)
            if mean is not None:
                values["a_dmg_in"] = min(1.0, mean / target.max_hp)
                values["drop_gap"] = values["a_dmg_in"] - (y_prev - y_hp)
        return values

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


# ----------------------------------------------------------------------
# Gen 2 field and end-of-turn mechanics
# ----------------------------------------------------------------------
SCREEN_MOVES = {"reflect": "reflect", "lightscreen": "lightscreen"}
WEATHER_MOVES = {"raindance": "rain", "sunnyday": "sun", "sandstorm": "sand"}
FIELD_TURNS = 5
PARTIAL_TRAPS = frozenset({"wrap", "bind", "clamp", "firespin", "whirlpool"})
SAND_IMMUNE = frozenset({"rock", "ground", "steel"})
LEFTOVERS_SHARE = 0.95  # share of known Pokémon holding Leftovers in the records


def _fresh_move(timeline: Sequence[Turn], side: str, i: int) -> Turn | None:
    """``side``'s move really used at row ``i`` (the records repeat stale ones)."""
    move = timeline[i][f"{side}_move_details"]
    if move is None or i == 0:
        return move
    before = timeline[i - 1]
    if before[f"{side}_pokemon_state"]["name"] != timeline[i][f"{side}_pokemon_state"]["name"]:
        return None
    previous = before[f"{side}_move_details"]
    return None if previous is not None and previous["name"] == move["name"] else move


def _field_tracker(
    timeline: Sequence[Turn],
) -> list[tuple[str | None, dict[str, frozenset[str]]]]:
    """Weather and each side's screens after every row (Gen 2: 5 turns)."""
    weather: str | None = None
    weather_left = 0
    screens: dict[str, dict[str, int]] = {"p1": {}, "p2": {}}
    result = []
    for i in range(len(timeline)):
        for side in SIDES:
            move = _fresh_move(timeline, side, i)
            if move is None:
                continue
            if move["name"] in WEATHER_MOVES:
                weather, weather_left = WEATHER_MOVES[move["name"]], FIELD_TURNS
            if move["name"] in SCREEN_MOVES:
                screens[side][SCREEN_MOVES[move["name"]]] = FIELD_TURNS
        active = {s: frozenset(k for k, v in d.items() if v > 0) for s, d in screens.items()}
        result.append((weather if weather_left > 0 else None, active))
        weather_left = max(0, weather_left - 1)
        for left in screens.values():
            for key in left:
                left[key] -= 1
    return result


def _status_counters(timeline: Sequence[Turn], side: str) -> tuple[list[int], list[int]]:
    """Rows the active Pokémon has been badly poisoned / under Perish Song."""
    tox, perish = [], []
    t = p = 0
    for i, row in enumerate(timeline):
        state = row[f"{side}_pokemon_state"]
        same = i > 0 and timeline[i - 1][f"{side}_pokemon_state"]["name"] == state["name"]
        t = (t + 1 if same else 1) if state["status"] == "tox" else 0
        p = (p + 1 if same else 1) if "perish" in state["effects"] else 0
        tox.append(t)
        perish.append(p)
    return tox, perish


def _mechanics(
    timeline: Sequence[Turn],
    side: str,
    before: int,
    index: int,
    species: SpeciesLookup,
    chart: TypeChart,
    field_state: tuple[str | None, Mapping[str, frozenset[str]]],
    tox_turns: int,
    perish_turns: int,
) -> dict[str, float]:
    """Gen 2 mechanics the base features ignore.

    * ``a_dmg_field``: expected damage of the opposing move on the leaver
      with the weather and the leaver's Reflect / Light Screen;
    * ``x_residual``: end-of-turn damage on the leaver (poison and burn
      1/8, Toxic n/16, Leech Seed 1/8, Curse 1/4, Nightmare 1/4, partial
      trapping 1/16, sandstorm 1/8 unless Rock / Ground / Steel) minus the
      expected Leftovers recovery;
    * ``x_hp_after``: leaver's HP minus both;
    * ``a_priority``: priority of the opposing move;
    * ``x_perish_turns`` / ``x_tox_turns``: rows under Perish Song / Toxic;
    * ``a_drop_any``: largest HP loss of any opposing Pokémon active over
      the rows of the change, against its last known HP. It sees the hit
      taken by an opponent that switched in on that turn (an Explosion,
      the leaver's last attack), which ``a_drop`` misses.
    """
    other = "p2" if side == "p1" else "p1"
    weather, screens = field_state
    leaver_state = timeline[before][f"{side}_pokemon_state"]
    attacker_state = timeline[index][f"{other}_pokemon_state"]
    move = timeline[index][f"{other}_move_details"]
    leaver = species(leaver_state["name"])
    attacker = species(attacker_state["name"])
    status, effects = leaver_state["status"], set(leaver_state["effects"])

    residual = 0.0
    if status in ("psn", "brn"):
        residual += 1 / 8
    if status == "tox":
        residual += min(15, tox_turns + 1) / 16
    if "leechseed" in effects:
        residual += 1 / 8
    if "curse" in effects:
        residual += 1 / 4
    if "nightmare" in effects and status == "slp":
        residual += 1 / 4
    if effects & PARTIAL_TRAPS:
        residual += 1 / 16
    if weather == "sand" and leaver is not None and not set(leaver.types) & SAND_IMMUNE:
        residual += 1 / 8
    residual -= LEFTOVERS_SHARE / 16

    damage = 0.0
    if (
        move is not None
        and move["category"] != "status"
        and leaver is not None
        and attacker is not None
    ):
        a = Combatant(
            attacker,
            boosts=_stages(attacker_state.get("boosts") or {}),
            status=_status(attacker_state["status"]),
        )
        d = Combatant(leaver, boosts=_stages(leaver_state.get("boosts") or {}))
        mean = expected_damage(
            a,
            d,
            Move.from_dict(dict(move)),
            chart,
            conditions=Field.of(weather, screens[side]),
            include_accuracy=True,
        )
        damage = 0.0 if mean is None else mean / d.max_hp

    drop_any = 0.0
    for r in range(before + 1, index + 1):
        state = timeline[r][f"{other}_pokemon_state"]
        last = 1.0
        for q in range(r - 1, -1, -1):
            earlier = timeline[q][f"{other}_pokemon_state"]
            if earlier["name"] == state["name"]:
                last = earlier["hp_pct"]
                break
        drop_any = max(drop_any, last - state["hp_pct"])

    return {
        "a_dmg_field": damage,
        "x_residual": residual,
        "x_hp_after": leaver_state["hp_pct"] - damage - residual,
        "a_priority": float(move["priority"]) if move else 0.0,
        "x_perish_turns": float(perish_turns),
        "x_tox_turns": float(tox_turns),
        "a_drop_any": drop_any,
    }
