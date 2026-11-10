"""Changes of active Pokémon described for a hidden-faint detector: the basic description.

P2's faints are recorded only when they coincide with a P1 faint; otherwise the next entry
already shows P2's replacement, which looks like a voluntary switch. P1's faints are always
recorded, so :class:`SwitchExtractor` describes every change of active Pokémon of a side the same
way on both sides, and describes a P1 faint from the entry before the faint entry, so that it
looks like a hidden P2 faint: the leaver before the change (HP, status, boosts, effects, base
bulk), the incoming Pokémon (first appearance, HP and the HP it lost since it was last seen,
own move, base stat total, Spikes on its side) and the opponent (whether it stayed in and the HP
it lost, its move, the move's effectiveness and :mod:`domain.expected_damage` estimate on the
leaver, the leaver's HP minus that estimate).

Unlike :mod:`domain.transitions` this is the basic description: no physical-evidence columns
(speed, worst roll, damage on the incoming Pokémon), no field or end-of-turn mechanics, and the
damage is the roll-free estimate of :mod:`domain.expected_damage` instead of the Gen 2 formula of
:class:`domain.damage.DamageCalculator`. The values keep the types they are computed with
(booleans, integers, floats, NaN where a species has no stats).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from domain.expected_damage import effectiveness, expected_damage
from domain.pokedex import PokemonGen2, SpeciesLookup
from domain.timeline import FAINTED, OPPONENT, ObservedTimeline
from domain.typechart import TypeChart

PHAZING_MOVES = frozenset({"roar", "whirlwind"})
MISSING = float("nan")

COLUMNS = (
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
class Switch:
    """One change of active Pokémon of a side.

    ``index`` is the entry where the replacement appears, ``leaver`` the Pokémon that left;
    ``recorded`` a recorded faint, ``final`` a leaver never seen again. ``features`` maps
    ``COLUMNS`` to values; it is empty for a recorded faint that cannot be described (every P2
    one, and P1's double knockouts).
    """

    index: int
    leaver: str
    recorded: bool
    final: bool
    features: Mapping[str, float] = field(default_factory=dict[str, float])

    @property
    def label(self) -> int:
        """1 for a recorded faint, else 0: what a detector learns on P1's switches."""
        return int(self.recorded)


class SwitchExtractor:
    """Describes the changes of active Pokémon of a record (species of that record's lookup)."""

    def __init__(self, species: SpeciesLookup, chart: TypeChart) -> None:
        self._species = species
        self._chart = chart

    def extract(self, record: Mapping[str, Any], side: Literal["p1", "p2"]) -> list[Switch]:
        """Every change of active Pokémon of ``side``, in timeline order."""
        timeline = ObservedTimeline(record)
        own, opp = timeline.sides[side], timeline.sides[OPPONENT[side]]
        result: list[Switch] = []
        last_hp: dict[str, float] = {}
        last_move: dict[str, str] = {}
        seen: set[str] = set()
        for i in range(1, timeline.n):
            j = i - 1
            leaver = own.names[j]
            last_hp[leaver] = own.hp[j]
            seen.add(leaver)
            move = own.moves[j]
            if move is not None:
                last_move[leaver] = move["name"]
            if leaver == own.names[i]:
                continue
            final = own.last_index[leaver] == j
            if own.status[j] == FAINTED:
                usable = (
                    side == "p1"
                    and opp.status[j] != FAINTED
                    and i >= 2
                    and own.names[i - 2] == leaver
                )
                if not usable:
                    result.append(Switch(i, leaver, True, True))
                    continue
                before, recorded = i - 2, True
            else:
                before, recorded = i - 1, False
            values = self._describe(timeline, side, before, i, last_hp, last_move, seen)
            result.append(Switch(i, leaver, recorded, final, values))
        return result

    def examples(self, record: Mapping[str, Any]) -> list[Switch]:
        """P1's described switches whose leaver never comes back: a detector's training rows,
        labelled by the recorded faint."""
        return [s for s in self.extract(record, "p1") if s.final and s.features]

    def candidates(self, record: Mapping[str, Any]) -> list[Switch]:
        """P2's switches whose leaver never comes back and is not a recorded faint: the switches
        that may hide a knockout."""
        return [s for s in self.extract(record, "p2") if s.final and not s.recorded]

    def _describe(
        self,
        timeline: ObservedTimeline,
        side: str,
        before: int,
        i: int,
        last_hp: Mapping[str, float],
        last_move: Mapping[str, str],
        seen: set[str],
    ) -> dict[str, float]:
        """The columns of the change at entry ``i``, the leaver read at entry ``before``."""
        own = timeline.sides[side]
        x = self._species(own.names[i - 1])
        return {
            "turn": i,
            **_leaver(timeline, side, before, x),
            **_incoming(timeline, side, i, self._species(own.names[i]), last_hp, last_move, seen),
            "spikes": timeline.spikes[side][i - 1],
            **self._opponent(timeline, side, before, i, x),
        }

    def _opponent(
        self, timeline: ObservedTimeline, side: str, before: int, i: int, x: PokemonGen2 | None
    ) -> dict[str, float]:
        """The opponent at the change: stayed in, HP lost, its move and the estimate on ``x``."""
        own, other = timeline.sides[side], OPPONENT[side]
        opp = timeline.sides[other]
        attacker, move, previous = opp.names[i], opp.moves[i], opp.moves[before]
        same = opp.names[before] == attacker
        damage = expected_damage(
            self._chart,
            move,
            self._species(attacker),
            x,
            att_level=timeline.level(other, attacker),
            def_level=timeline.level(side, own.names[i - 1]),
            att_boosts=opp.boosts[i],
            def_boosts=own.boosts[before],
            att_item=timeline.item(other, attacker),
            att_status=opp.status[i],
            att_name=attacker,
        )
        return {
            "a_same": same,
            "a_drop": opp.hp[before] - opp.hp[i] if same else MISSING,
            "a_mv_changed": move is not None
            and (not same or previous is None or previous["name"] != move["name"]),
            "a_bp": move["base_power"] if move else 0,
            "a_status_mv": move is not None and move["category"] == "status",
            "a_eff": effectiveness(self._chart, move["type"], x.types) if (move and x) else MISSING,
            "a_dmg": damage,
            "x_hp_minus_dmg": own.hp[before] - damage,
            "a_phaze": move is not None and move["name"] in PHAZING_MOVES,
            "a_boost": sum(opp.boosts[i].values()),
            "a_disabled": opp.status[i] in ("slp", "frz"),
        }


def _leaver(
    timeline: ObservedTimeline, side: str, before: int, x: PokemonGen2 | None
) -> dict[str, float]:
    """The leaver at entry ``before``: HP, status, boosts, effects and base bulk."""
    own = timeline.sides[side]
    status, effects = own.status[before], own.effects[before]
    return {
        "x_hp": own.hp[before],
        "x_slp": status == "slp",
        "x_par": status == "par",
        "x_frz": status == "frz",
        "x_psn": status in ("psn", "tox"),
        "x_brn": status == "brn",
        "x_boost": sum(own.boosts[before].values()),
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


def _incoming(
    timeline: ObservedTimeline,
    side: str,
    i: int,
    y: PokemonGen2 | None,
    last_hp: Mapping[str, float],
    last_move: Mapping[str, str],
    seen: set[str],
) -> dict[str, float]:
    """The incoming Pokémon at entry ``i``: first appearance, HP and HP lost, move, bulk."""
    own = timeline.sides[side]
    incoming, move = own.names[i], own.moves[i]
    y_prev = last_hp.get(incoming, 1.0)
    return {
        "y_first": incoming not in seen,
        "y_hp": own.hp[i],
        "y_prev_hp": y_prev,
        "y_drop": y_prev - own.hp[i],
        "y_mv_none": move is None,
        "y_mv_new": move is not None and last_move.get(incoming) != move["name"],
        "y_bst": y.base_stat_total if y else MISSING,
    }
