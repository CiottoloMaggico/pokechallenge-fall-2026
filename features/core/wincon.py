"""Win conditions still available to each player at the sampled turn.

Counted over the Pokémon still alive, from the moves revealed so far
(P1's moves are not in the records either), as P1's count minus P2's,
over 6 (positive when P1 has more of them):

* ``wc_healers``: Pokémon with a recovery move (Recover, Soft-Boiled,
  Milk Drink, Moonlight, Morning Sun, Synthesis, Rest);
* ``wc_setup``: Pokémon with a boosting move (Swords Dance, Belly Drum,
  Curse used by a non-Ghost, Agility, Amnesia, Growth, ...);
* ``wc_phazers``: Pokémon with Roar or Whirlwind;
* ``wc_sleepers``: Pokémon with a sleep move (Spore, Sleep Powder,
  Lovely Kiss, Hypnosis, Sing).

``wc_spikes_adv``: +1 if Spikes are on P2's side and P2 has shown no
alive Rapid Spinner, -1 for the same on P1's side, 0 otherwise.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from domain.battle import BattleContext, PlayerState, PokemonState

TEAM_SIZE = 6
RAPID_SPIN = "rapidspin"
CURSE = "curse"
HEALING = frozenset(
    {
        "recover",
        "softboiled",
        "milkdrink",
        "moonlight",
        "morningsun",
        "synthesis",
        "rest",
    }
)
SETUP = frozenset(
    {
        "swordsdance",
        "bellydrum",
        "agility",
        "amnesia",
        "growth",
        "meditate",
        "sharpen",
        "barrier",
        "acidarmor",
        "doubleteam",
        "minimize",
    }
)
PHAZING = frozenset({"roar", "whirlwind"})
SLEEP = frozenset({"spore", "sleeppowder", "lovelykiss", "hypnosis", "sing"})

Counter = Callable[[PokemonState], float]


def _alive(player: PlayerState) -> list[PokemonState]:
    return [p for p in player.team.values() if not p.fainted]


def _knows(moves: frozenset[str]) -> Counter:
    return lambda pokemon: bool(moves & set(pokemon.known_moves))


def _sets_up(pokemon: PokemonState) -> float:
    known = set(pokemon.known_moves)
    if SETUP & known:
        return 1.0
    if CURSE not in known:
        return 0.0
    # Curse only boosts a non-Ghost, and the types of this species are unknown
    if not pokemon.info.has_stats:
        return math.nan
    return float("ghost" not in pokemon.info.types)


def _difference(counter: Counter) -> Callable[[BattleContext], float]:
    def difference(context: BattleContext) -> float:
        p1, p2 = (
            sum(counter(p) for p in _alive(context.side(s))) for s in ("p1", "p2")
        )
        return (p1 - p2) / TEAM_SIZE

    return difference


healers = _difference(_knows(HEALING))
setup_users = _difference(_sets_up)
phazers = _difference(_knows(PHAZING))
sleepers = _difference(_knows(SLEEP))


def spikes_advantage(context: BattleContext) -> float:
    """``wc_spikes_adv`` (see the module docstring)."""
    return float(_spikes_stick(context.side("p2"))) - float(
        _spikes_stick(context.side("p1"))
    )


def _spikes_stick(player: PlayerState) -> bool:
    spinner = any(RAPID_SPIN in p.known_moves for p in _alive(player))
    return player.spikes and not spinner
