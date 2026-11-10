"""What a battle record shows at each timeline entry, side by side, without simulating anything.

:class:`domain.battle.BattleReplay` replays the turns into states: it ages effects and screens,
tracks items, applies the faints of a repair and decides which moves were really used.
:class:`ObservedTimeline` only copies what the record shows, entry by entry: the active Pokémon of
each side, its HP, status, effects (``noeffect`` dropped), boosts and move field, plus a few
readings derived from those columns alone:

* **events**: the move field repeats the last move of the active Pokémon, so a fresh use shows up
  as a change of that field (a Pokémon coming back with the move it last used counts once);
* **revealed moves** of each Pokémon (P2's lead counts as revealed even if it leaves at once and
  the timeline never shows it again);
* first and last entry of each Pokémon;
* **Spikes** on each side: laid by an event of the opponent, cleared by an own Rapid Spin event;
* :meth:`ObservedTimeline.used_move`, the move actually used at an entry.

Levels and items are the record's; a Pokémon whose level the record does not show (P2 after its
lead) is assumed to be at level 100, and P2's items are never shown.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

Move = Mapping[str, Any]
Record = Mapping[str, Any]

SIDES = ("p1", "p2")
OPPONENT = {"p1": "p2", "p2": "p1"}
NO_EFFECT = "noeffect"
FAINTED = "fnt"
CANNOT_ACT = ("slp", "frz")
SPIKES = "spikes"
RAPID_SPIN = "rapidspin"
UNKNOWN_LEVEL = 100


class SideTimeline:
    """One side's columns of the timeline, with its events and revealed moves."""

    def __init__(self, timeline: Sequence[Record], side: str) -> None:
        self.names: list[str] = []
        self.hp: list[float] = []
        self.status: list[str] = []
        self.effects: list[frozenset[str]] = []
        self.boosts: list[dict[str, int]] = []
        self.moves: list[Move | None] = []
        for turn in timeline:
            state = turn[f"{side}_pokemon_state"]
            self.names.append(state["name"])
            self.hp.append(state["hp_pct"])
            self.status.append(state["status"])
            self.effects.append(frozenset(e for e in state["effects"] if e != NO_EFFECT))
            self.boosts.append(state["boosts"])
            self.moves.append(turn[f"{side}_move_details"])
        self.n = len(self.names)
        self.events = self._events()
        self.revealed: dict[str, dict[str, Move]] = {}
        for name, move in zip(self.names, self.moves, strict=True):
            moves = self.revealed.setdefault(name, {})
            if move is not None:
                moves[move["name"]] = move
        self.first_index: dict[str, int] = {}
        self.last_index: dict[str, int] = {}
        for i, name in enumerate(self.names):
            self.first_index.setdefault(name, i)
            self.last_index[name] = i

    def _events(self) -> list[Move | None]:
        """The move field where it changed (a fresh use), else None."""
        events: list[Move | None] = [None] * self.n
        last: dict[str, str] = {}
        for i in range(self.n):
            name, move = self.names[i], self.moves[i]
            if move is None:
                continue
            if i > 0 and self.names[i - 1] == name:
                previous = self.moves[i - 1]
                reference = previous["name"] if previous else None
            else:
                reference = last.get(name)
            if reference != move["name"]:
                events[i] = move
            last[name] = move["name"]
        return events


class ObservedTimeline:
    """Both sides of a record entry by entry, Spikes, levels and items (see the module)."""

    def __init__(self, record: Record) -> None:
        team = record["p1_team_details"]
        lead = record["p2_lead_details"]
        self.levels: dict[str, dict[str, int]] = {
            "p1": {p["name"]: p["level"] for p in team},
            "p2": {lead["name"]: lead["level"]},
        }
        self.items: dict[str, str] = {p["name"]: p["item"] for p in team}
        timeline: Sequence[Record] = record["battle_timeline"]
        self.n = len(timeline)
        self.sides = {s: SideTimeline(timeline, s) for s in SIDES}
        self.sides["p2"].revealed.setdefault(lead["name"], {})
        self.spikes = {s: self._spikes(s) for s in SIDES}

    def level(self, side: str, name: str) -> int:
        """Level of ``name`` on ``side`` (``UNKNOWN_LEVEL`` when the record does not show it)."""
        return self.levels[side].get(name, UNKNOWN_LEVEL)

    def item(self, side: str, name: str) -> str | None:
        """Item of a P1 Pokémon; P2's items are never shown."""
        return self.items.get(name) if side == "p1" else None

    def used_move(self, side: str, i: int) -> Move | None:
        """Move used at entry ``i`` by the active Pokémon of ``side``.

        The move field also counts when the same move is used again, but not on a switch-in,
        while the Pokémon cannot act (asleep or frozen at both entries), or right after a P1
        faint (only the replacement comes in).
        """
        own = self.sides[side]
        if i == 0:
            return own.moves[0]
        if self.sides["p1"].status[i - 1] == FAINTED or own.names[i] != own.names[i - 1]:
            return None
        if own.status[i - 1] in CANNOT_ACT and own.status[i] in CANNOT_ACT:
            return None
        return own.moves[i]

    def _spikes(self, side: str) -> list[bool]:
        """Whether ``side`` has Spikes after each entry."""
        own, opp = self.sides[side], self.sides[OPPONENT[side]]
        on = False
        result: list[bool] = []
        for i in range(self.n):
            event = opp.events[i]
            if event is not None and event["name"] == SPIKES:
                on = True
            event = own.events[i]
            if event is not None and event["name"] == RAPID_SPIN:
                on = False
            result.append(on)
        return result
