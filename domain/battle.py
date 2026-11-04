from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeVar

from domain.damage import STAGE_MULTIPLIER
from domain.moves import Move
from domain.pokedex import DEFAULT_LEVEL, PokemonGen2

NO_STATUS = "nostatus"
FAINTED = "fnt"
NO_EFFECT = "noeffect"
NO_ITEM = "noitem"
UNKNOWN_ITEM = None
STATS = ("atk", "def", "spa", "spd", "spe")
SCREEN_TURNS = 5
WEATHER_TURNS = 5

PARTIAL_TRAP = {"wrap", "bind", "clamp", "firespin", "whirlpool"}

EVENT_EFFECTS = {
    "focusband",
    "quickclaw",
    "mysteryberry",
    "magnitude",
    "beatup",
    "pursuit",
    "bide",
    "futuresight",
    "mimic",
    "spite",
    "typechange",
}

SIDE_EFFECTS = {"safeguard"}

MAX_TURNS = {
    "protect": 1,
    "endure": 1,
    "destinybond": 1,
    "confusion": 5,
    "encore": 6,
    "disable": 7,
    **{e: 5 for e in PARTIAL_TRAP},
    "perish": 4,
}

OPPONENT_BOUND = {"trapped", "attract", *PARTIAL_TRAP}

BATON_PASSABLE = {
    "substitute",
    "confusion",
    "leechseed",
    "trapped",
    "perish",
    "focusenergy",
    "curse",
    "lockon",
    "mindreader",
}

CLEARS_SELF = {"rapidspin": {"leechseed", *PARTIAL_TRAP}}
SCREENS = {"reflect", "lightscreen", "safeguard"}
WEATHER = {"raindance": "rain", "sunnyday": "sun", "sandstorm": "sand"}

CURE_BERRIES = {
    "mintberry": {"slp"},
    "przcureberry": {"par"},
    "psncureberry": {"psn", "tox"},
    "iceberry": {"brn"},
    "burntberry": {"frz"},
    "miracleberry": {"slp", "par", "psn", "tox", "brn", "frz"},
}
CURE_BERRIES_VOLATILE = {"bitterberry": {"confusion"}, "miracleberry": {"confusion"}}
TEAM_CURE = {"healbell"}

# The records repeat a Pokémon's last move on turns it did not act: when
# it switches in, while it stays frozen or asleep (except Sleep Talk and
# Snore) and, for the side still standing, on the turn after the other
# side's Pokémon fainted (only the replacement comes in).
IMMOBILE = {"frz": frozenset(), "slp": frozenset({"sleeptalk", "snore"})}
PHAZING_MOVES = {"roar", "whirlwind"}
SPIN_BLOCKERS = {"ghost"}  # Rapid Spin (Normal) does not touch Ghost types
PROTECTED = "protect"
PARALYSIS_SPEED = 4
MAX_STAGE = 6

T = TypeVar("T")
Timeline = Sequence[Mapping[str, Any]]
Species = Callable[[str], PokemonGen2]
Repair = Callable[[Timeline, Species], Mapping[int, str]]
"""Finds the P2 faints a timeline omits: {turn index: fainted Pokémon}."""


class PokemonState:
    """One individual Pokémon in a battle: its species plus its own level,
    current HP, status, item and the moves seen so far."""

    def __init__(
        self,
        pokemon: PokemonGen2,
        level: int = DEFAULT_LEVEL,
        item: str | None = NO_ITEM,
    ) -> None:
        if not 1 <= level <= 100:
            raise ValueError(f"{pokemon.name}: level must be in 1..100, got {level}")
        self.info = pokemon
        self.level = level
        self.hp_pct = 1.0
        self.status = NO_STATUS
        self.status_turns = 0
        self.asleep_by_rest = False

        self.item = item
        self.item_used = False
        self.item_stolen = False

        self.known_moves: dict[str, Move] = {}
        self.seen = False

    @property
    def name(self) -> str:
        return self.info.name

    @property
    def max_hp(self) -> int:
        return self.info.max_hp(self.level)

    @property
    def hp(self) -> float:
        return self.hp_pct * self.max_hp

    def stat(self, name: str) -> int:
        """Real value of ``"atk"``, ``"def"``, ``"spa"``, ``"spd"``, ``"spe"``
        at this Pokémon's level (boosts and status are not applied)."""
        return self.info.stat(name, self.level)

    @property
    def fainted(self) -> bool:
        return self.status == "fnt" or self.hp_pct == 0

    @property
    def statused(self) -> bool:
        return not self.fainted and self.status != NO_STATUS

    @property
    def has_item(self) -> bool:
        return self.item not in (NO_ITEM, UNKNOWN_ITEM) and not self.item_used

    def add_move(self, mv: Move) -> None:
        self.known_moves[mv.name] = mv

    def observe(self, hp_pct: float, status: str, move_name: str | None) -> str | None:
        cured = None
        if status == self.status and not (status == "slp" and move_name == "rest"):
            self.status_turns += 1
        else:
            if self.status not in (NO_STATUS, "fnt") and status == NO_STATUS:
                cured = self.status
                if (
                    move_name not in ("rest", *TEAM_CURE)
                    and self.item in CURE_BERRIES
                    and cured in CURE_BERRIES[self.item]
                    and not self.item_used
                ):
                    self.item_used = True
            self.status = status
            self.status_turns = 0 if status == NO_STATUS else 1
            self.asleep_by_rest = status == "slp" and move_name == "rest"
        self.hp_pct = hp_pct
        self.seen = True
        return cured

    def on_switch_out(self) -> None:
        if self.status == "tox":
            self.status = "psn"


class LeadState:
    def __init__(self, pokemon: PokemonState) -> None:
        self.pokemon = pokemon
        self.effects: dict[str, int] = {}
        self.boosts = {s: 0 for s in STATS}
        self.turns_in = 0
        self.last_move: str | None = None
        self.events: set[str] = set()

    def add_effect(self, effect: str) -> None:
        self.effects.setdefault(effect, 0)

    def remove_effects(self, effects) -> None:
        for e in effects:
            self.effects.pop(e, None)

    def reset_boosts(self) -> None:
        self.boosts = {s: 0 for s in STATS}

    def baton_pass_payload(self) -> tuple[dict, dict]:
        return (
            {e: n for e, n in self.effects.items() if e in BATON_PASSABLE},
            dict(self.boosts),
        )

    def tick(self) -> None:
        self.turns_in += 1
        for e in list(self.effects):
            self.effects[e] += 1
            if e in MAX_TURNS and self.effects[e] >= MAX_TURNS[e]:
                del self.effects[e]
        self.events.clear()


class PlayerState:
    def __init__(
        self, team: dict[str, PokemonState] | None = None, team_size: int = 6
    ) -> None:
        self.team = team if team is not None else {}
        self.team_size = team_size
        self.lead: LeadState | None = None
        self.spikes = False
        self.screens: dict[str, int] = {}
        self.n_switches = 0

    def add_pokemon(self, pokemon: PokemonState) -> None:
        if pokemon.name in self.team:
            return
        if len(self.team) >= self.team_size:
            raise ValueError("the team is full, cannot add new pokemons")
        self.team[pokemon.name] = pokemon

    def switch_in(self, name: str, baton_pass: bool = False) -> None:
        pokemon = self.team.get(name)
        if pokemon is None:
            raise ValueError(
                f"unknown name {name!r}: add the pokemon to the player team"
            )

        old = self.lead
        self.lead = LeadState(pokemon)
        if old is not None:
            self.n_switches += 1
            old.pokemon.on_switch_out()
            if baton_pass:
                effects, boosts = old.baton_pass_payload()
                self.lead.effects.update(effects)
                self.lead.boosts = boosts

    def tick(self) -> None:
        for s in list(self.screens):
            self.screens[s] -= 1
            if self.screens[s] <= 0:
                del self.screens[s]
        if self.lead:
            self.lead.tick()

    @property
    def active(self) -> LeadState:
        if self.lead is None:
            raise RuntimeError("no active pokemon: call switch_in first")
        return self.lead

    @property
    def n_fainted(self) -> int:
        return sum(p.fainted for p in self.team.values())

    @property
    def n_alive(self) -> int:
        return self.team_size - self.n_fainted

    @property
    def n_statused(self) -> int:
        return sum(p.statused for p in self.team.values())

    @property
    def n_unseen(self) -> int:
        return self.team_size - sum(p.seen for p in self.team.values())

    @property
    def hp_total(self) -> float:
        return sum(p.hp_pct for p in self.team.values() if p.seen) + self.n_unseen


class Battle:
    def __init__(
        self,
        p1: PlayerState,
        p2: PlayerState,
        species: Callable[[str], PokemonGen2] | None = None,
        trust_noeffect: bool = True,
    ) -> None:
        self.p1, self.p2 = p1, p2
        self._species = species
        self.turn = 0
        self.weather: str | None = None
        self.weather_turns = 0
        self.trust_noeffect = trust_noeffect
        self.moves: dict[str, Move | None] = {"p1": None, "p2": None}
        self.forced: dict[str, bool] = {"p1": False, "p2": False}
        self.switched: dict[str, bool] = {"p1": False, "p2": False}
        self.targets: dict[str, str | None] = {"p1": None, "p2": None}

    @classmethod
    def from_record(
        cls, record: dict, species: Callable[[str], PokemonGen2], **kw
    ) -> Battle:
        """Build the initial state; ``species`` maps a name to its species
        (for example ``Pokedex.get``). Levels come from the record."""
        p1 = PlayerState()
        for d in record["p1_team_details"]:
            p1.add_pokemon(
                PokemonState(
                    species(d["name"]),
                    level=d.get("level", DEFAULT_LEVEL),
                    item=d.get("item") or NO_ITEM,
                )
            )
        p1.team_size = len(p1.team)
        p2 = PlayerState()
        lead = record["p2_lead_details"]
        p2_lead = PokemonState(
            species(lead["name"]),
            level=lead.get("level", DEFAULT_LEVEL),
            item=UNKNOWN_ITEM,
        )
        p2_lead.seen = True  # revealed by the record even if it leaves at once
        p2.add_pokemon(p2_lead)
        p1_lead_name = record["p1_team_details"][0]["name"]
        p1.team[p1_lead_name].seen = True
        p1.switch_in(p1_lead_name)
        p2.switch_in(p2_lead.name)
        return cls(p1, p2, species, **kw)

    def side(self, name: str) -> PlayerState:
        return self.p1 if name == "p1" else self.p2

    def update(self, t: dict, p2_fainted: str | None = None) -> Battle:
        """Apply one turn of the record.

        ``p2_fainted`` names P2's active Pokémon when it fainted this turn
        without a ``fnt`` record (see ``knockouts``). Afterwards
        :attr:`moves` holds the moves really used this turn and
        :attr:`forced` which sides had to switch (faint or phazing),
        :attr:`switched` which changed Pokémon and :attr:`targets` the
        Pokémon each side's move was aimed at. Turn-based counters
        (effects, screens, weather) advance at the start of the next turn,
        so the state after ``update`` is the end of this turn as recorded.
        """
        if self.turn:
            self._end_of_turn()
        self.turn = t["turn"]
        sides = {"p1": (self.p1, self.p2), "p2": (self.p2, self.p1)}
        obs = {s: t[f"{s}_pokemon_state"] for s in sides}
        moves = {
            s: (
                Move.from_dict(t[f"{s}_move_details"])
                if t[f"{s}_move_details"]
                else None
            )
            for s in sides
        }
        before = {
            s: me.lead.pokemon if me.lead else None for s, (me, _) in sides.items()
        }
        was_fainted = {s: p is not None and p.fainted for s, p in before.items()}
        leaving = before["p2"]
        if leaving is not None and leaving.name == p2_fainted:
            leaving.hp_pct, leaving.status = 0.0, FAINTED

        switched = {}
        for s, (me, _) in sides.items():
            name = obs[s]["name"]
            if name not in me.team:
                if self._species is None:
                    raise ValueError(f"unknown pokemon {name!r} and no species given")
                me.add_pokemon(
                    PokemonState(
                        self._species(name), level=DEFAULT_LEVEL, item=UNKNOWN_ITEM
                    )
                )

            current = me.lead
            if current is None:
                switched[s] = True
                me.switch_in(name)
            elif current.pokemon.name != name:
                switched[s] = True
                me.switch_in(name, baton_pass=current.last_move == "batonpass")
            else:
                switched[s] = False

        for s in sides:
            other = "p2" if s == "p1" else "p1"
            replacement_turn = was_fainted[other] and not was_fainted[s]
            if (
                switched[s]
                or replacement_turn
                or _immobile(before[s], obs[s], moves[s])
            ):
                moves[s] = None
        self.moves = moves
        self.switched = switched
        self.forced = {}
        fainted_out = {}
        for s in sides:
            other = "p2" if s == "p1" else "p1"
            fainted_out[s] = switched[s] and before[s] is not None and before[s].fainted
            phazed = moves[other] is not None and moves[other].name in PHAZING_MOVES
            self.forced[s] = fainted_out[s] or (switched[s] and phazed)
        # a move hits the Pokémon in front; if that one fainted from it, the
        # replacement shown at the end of the turn was not the target
        self.targets = {}
        for s, (_, opp) in sides.items():
            other = "p2" if s == "p1" else "p1"
            gone = (
                before[other] if fainted_out[other] and not was_fainted[other] else None
            )
            self.targets[s] = gone.name if gone is not None else opp.active.pokemon.name

        for s, (me, _) in sides.items():
            other = "p2" if s == "p1" else "p1"
            if switched[other] and not switched[s]:
                me.active.remove_effects(OPPONENT_BOUND)

        order = self._action_order(moves)

        for s, (me, _) in sides.items():
            o, lead = obs[s], me.active
            mv = moves[s]
            mv_name = mv.name if mv is not None else None
            cured = lead.pokemon.observe(o["hp_pct"], o["status"], mv_name)
            if cured == "slp":
                lead.remove_effects({"nightmare"})
            lead.boosts = dict(o.get("boosts") or lead.boosts)

            self._observe_effects(me, o.get("effects") or [NO_EFFECT])
            if lead.pokemon.fainted:
                lead.effects.clear()

        for s in order:
            me, opp = sides[s]
            mv = moves[s]
            if mv is not None:
                if self._knocked_out_first(s, moves, order):
                    self.moves[s] = None
                else:
                    self._apply_move(mv, me, opp)
        return self

    def _end_of_turn(self) -> None:
        self.p1.tick()
        self.p2.tick()
        if self.weather:
            self.weather_turns -= 1
            if self.weather_turns <= 0:
                self.weather = None

    def _observe_effects(self, me: PlayerState, effects: Sequence[str]) -> None:
        """Read every recorded effect of the active Pokémon.

        The record lists the active's volatile effects at the end of the
        turn, so (with ``trust_noeffect``) an effect no longer listed, such
        as a broken Substitute, is removed.
        """
        lead = me.active
        volatile = set()
        for effect in effects:
            if effect in EVENT_EFFECTS:
                lead.events.add(effect)
            elif effect in SIDE_EFFECTS:
                me.screens.setdefault(effect, SCREEN_TURNS)
            elif effect != NO_EFFECT:
                lead.add_effect(effect)
                volatile.add(effect)
        if self.trust_noeffect:
            lead.remove_effects(set(lead.effects) - volatile)

    def _action_order(self, moves: Mapping[str, Move | None]) -> list[str]:
        """Sides in the order they acted: priority, then Speed."""

        def key(side: str) -> tuple[int, float]:
            move = moves[side]
            return (move.priority if move is not None else 0, _speed(self.side(side)))

        return sorted(("p1", "p2"), key=key, reverse=True)

    def _knocked_out_first(
        self, side: str, moves: Mapping[str, Move | None], order: Sequence[str]
    ) -> bool:
        """``side`` fainted this turn after the other side hit it first."""
        other = "p2" if side == "p1" else "p1"
        my_move, opp_move = moves[side], moves[other]
        if not self.side(side).active.pokemon.fainted or opp_move is None:
            return False
        if my_move is not None and my_move.name in ("explosion", "selfdestruct"):
            return False
        if self.side(other).active.pokemon.fainted and opp_move.name not in (
            "explosion",
            "selfdestruct",
        ):
            return False
        return not opp_move.is_status and order[0] == other

    def _apply_move(self, mv: Move, me: PlayerState, opp: PlayerState) -> None:
        """Effects of a move on what the record does not show.

        Only moves really used this turn get here. Boosts and the active
        Pokémon's status are read from the record (end of the turn), so
        Haze and Heal Bell do not overwrite them.
        """
        user, target = me.active, opp.active
        user.pokemon.add_move(mv)
        name = mv.name

        if name != "destinybond":
            user.remove_effects({"destinybond"})
        if name in CLEARS_SELF:
            user.remove_effects(CLEARS_SELF[name])
        if name == "rapidspin" and _spin_hits(target):
            me.spikes = False
        if name == "spikes":
            opp.spikes = True
        if name in SCREENS:
            me.screens.setdefault(name, SCREEN_TURNS)
        if name in WEATHER:
            self.weather, self.weather_turns = WEATHER[name], WEATHER_TURNS
        if name in TEAM_CURE:
            for p in me.team.values():
                if not p.fainted and p is not user.pokemon:
                    p.status, p.status_turns = NO_STATUS, 0
        if name == "thief" and not user.pokemon.has_item and target.pokemon.has_item:
            target.pokemon.item_stolen = True
            user.pokemon.item, target.pokemon.item = target.pokemon.item, NO_ITEM

        user.last_move = name


def _speed(player: PlayerState) -> float:
    """Speed of the active Pokémon with stat stages and paralysis."""
    lead = player.active
    stage = max(-MAX_STAGE, min(MAX_STAGE, lead.boosts.get("spe", 0)))
    value = lead.pokemon.stat("spe") * STAGE_MULTIPLIER[stage]
    if lead.pokemon.status == "par":
        value /= PARALYSIS_SPEED
    return value


def _spin_hits(target: LeadState) -> bool:
    """Rapid Spin fails on a Ghost type or into Protect."""
    if SPIN_BLOCKERS.intersection(target.pokemon.info.types):
        return False
    return PROTECTED not in target.effects


def _immobile(
    pokemon: PokemonState | None, observed: Mapping[str, Any], move: Move | None
) -> bool:
    """The Pokémon stayed frozen or asleep, so it could not use ``move``."""
    if pokemon is None or move is None or pokemon.name != observed["name"]:
        return False
    allowed = IMMOBILE.get(pokemon.status)
    return (
        allowed is not None
        and observed["status"] == pokemon.status
        and move.name not in allowed
    )


# ----------------------------------------------------------------------
# replay context: what every feature receives
# ----------------------------------------------------------------------
SIDES = ("p1", "p2")
LABEL_KEY = "player_won"


@dataclass(frozen=True)
class SideSnapshot:
    """One player's situation at the end of a turn (copied, immutable)."""

    active: str
    hp_pct: float
    status: str
    boosts: dict[str, int]
    effects: tuple[str, ...]
    move: Move | None
    target: str | None
    switched: bool
    voluntary_switch: bool
    n_alive: int
    n_fainted: int
    n_statused: int
    n_unseen: int
    hp_total: float
    spikes: bool
    screens: tuple[str, ...]

    @classmethod
    def capture(
        cls,
        player: PlayerState,
        move: Move | None,
        previous: SideSnapshot | None,
        forced: bool = False,
        switched: bool | None = None,
        target: str | None = None,
    ) -> SideSnapshot:
        """``move`` is the move used this turn and ``target`` the Pokémon it
        was aimed at; ``forced`` a switch made because the previous Pokémon
        fainted or was phazed. ``switched`` defaults to a change of active
        Pokémon since ``previous``."""
        lead = player.active
        name = lead.pokemon.name
        if switched is None:
            switched = previous is not None and previous.active != name
        return cls(
            active=name,
            hp_pct=lead.pokemon.hp_pct,
            status=lead.pokemon.status,
            boosts=dict(lead.boosts),
            effects=tuple(sorted(lead.effects)),
            move=move,
            target=target,
            switched=switched,
            voluntary_switch=switched and not forced,
            n_alive=player.n_alive,
            n_fainted=player.n_fainted,
            n_statused=player.n_statused,
            n_unseen=player.n_unseen,
            hp_total=player.hp_total,
            spikes=player.spikes,
            screens=tuple(sorted(player.screens)),
        )


@dataclass(frozen=True)
class TurnSnapshot:
    """Both players at the end of one turn."""

    turn: int
    p1: SideSnapshot
    p2: SideSnapshot

    def side(self, name: str) -> SideSnapshot:
        return self.p1 if name == "p1" else self.p2


@dataclass(frozen=True)
class BattleContext:
    """A replayed battle: final state, raw record and per-turn history.

    ``battle`` is the final state after the last turn; ``history`` has
    one snapshot per turn, in order. The outcome is deliberately absent:
    ``record`` is a copy without ``player_won``, so no feature can read
    the label (labels live only in ``data.Dataset``).
    """

    battle_id: Any
    record: dict[str, Any]
    battle: Battle
    history: tuple[TurnSnapshot, ...]
    species: Callable[[str], PokemonGen2] = field(repr=False, compare=False)
    repair: Repair | None = field(default=None, repr=False, compare=False)
    _truncated: dict[int, BattleContext] = field(
        default_factory=dict, repr=False, compare=False
    )
    _memo: dict[Hashable, Any] = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def from_record(
        cls,
        record: dict[str, Any],
        species: Callable[[str], PokemonGen2],
        repair: Repair | None = None,
    ) -> BattleContext:
        """Replay ``record`` turn by turn, capturing a snapshot per turn.

        ``species`` maps a name to its species (for example
        ``Pokedex.get``). ``repair`` (e.g. ``knockout_detector.KnockoutDetector``)
        finds the P2 faints the record omits; they are applied turn by
        turn, so the history and the final state both see them, and again
        on truncated copies (:meth:`upto`). Errors from the replay
        propagate unchanged.
        """
        battle = Battle.from_record(record, species)
        timeline = record["battle_timeline"]
        faints = repair(timeline, species) if repair is not None else {}
        history: list[TurnSnapshot] = []
        previous: TurnSnapshot | None = None
        for i, turn in enumerate(timeline):
            battle.update(turn, faints.get(i))
            snapshot = TurnSnapshot(
                turn=turn["turn"],
                p1=SideSnapshot.capture(
                    battle.p1,
                    battle.moves["p1"],
                    previous.p1 if previous else None,
                    battle.forced["p1"],
                    battle.switched["p1"],
                    battle.targets["p1"],
                ),
                p2=SideSnapshot.capture(
                    battle.p2,
                    battle.moves["p2"],
                    previous.p2 if previous else None,
                    battle.forced["p2"],
                    battle.switched["p2"],
                    battle.targets["p2"],
                ),
            )
            history.append(snapshot)
            previous = snapshot
        if not history:
            raise ValueError(f"battle {record['battle_id']}: empty timeline")
        return cls(
            battle_id=record["battle_id"],
            record={k: v for k, v in record.items() if k != LABEL_KEY},
            battle=battle,
            history=tuple(history),
            species=species,
            repair=repair,
        )

    def upto(self, turn: int | None) -> BattleContext:
        """The battle as if the timeline stopped after its first ``turn`` turns.

        ``None`` (or a turn past the end) returns the battle itself. The
        truncated battle is replayed from the record, so its final state,
        history and counters only use those turns. The last truncation
        is cached, so features sampled at the same turn share one replay.
        """
        if turn is None or turn >= self.n_turns:
            return self
        if turn < 1:
            raise ValueError(f"turn must be >= 1, got {turn}")
        cached = self._truncated.get(turn)
        if cached is None:
            timeline = self.record["battle_timeline"][:turn]
            record = {**self.record, "battle_timeline": timeline}
            cached = BattleContext.from_record(record, self.species, self.repair)
            self._truncated.clear()
            self._truncated[turn] = cached
        return cached

    def cached(self, key: Hashable, compute: Callable[[], T]) -> T:
        """``compute()`` once per replay, then the remembered value.

        For values derived only from this battle (stateless features):
        cross-validation asks for the same battle in every fold.
        """
        if key not in self._memo:
            self._memo[key] = compute()
        return self._memo[key]

    @property
    def last(self) -> TurnSnapshot:
        return self.history[-1]

    @property
    def n_turns(self) -> int:
        return len(self.history)

    def side(self, name: str) -> PlayerState:
        """Final state of ``"p1"`` or ``"p2"``."""
        return self.battle.side(name)

    def window(self, last_n: int) -> tuple[TurnSnapshot, ...]:
        """The last ``last_n`` turns (all of them if there are fewer)."""
        return self.history[-last_n:]

    def series(self, side: str, attribute: str) -> list[Any]:
        """One attribute of one side across all turns, e.g. hp_total."""
        return [getattr(t.side(side), attribute) for t in self.history]
