"""The Pokédex comes only from records; species without stats are never invented."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from domain.__main__ import main
from domain.battle import BattleContext
from domain.damage import Combatant, DamageCalculator
from domain.errors import PokedexError, UnknownSpeciesError
from domain.moves import Move
from domain.pokedex import (
    Missing,
    Pokedex,
    PokedexBuilder,
    PokemonGen2,
    Strict,
    species_without_stats,
)
from domain.typechart import TypeChart

ROOT = Path(__file__).resolve().parents[2]
TACKLE = Move("tackle", "normal", "physical", 40, 1.0, 0)


def details(
    name: str, types: tuple[str, str] = ("normal", "notype"), hp: int = 50
) -> dict[str, Any]:
    return {
        "name": name, "level": 100, "types": list(types), "base_hp": hp, "base_atk": 50,
        "base_def": 50, "base_spa": 50, "base_spd": 50, "base_spe": 50,
    }  # fmt: skip


def state(name: str) -> dict[str, Any]:
    return {
        "name": name,
        "hp_pct": 1.0,
        "status": "nostatus",
        "effects": ["noeffect"],
        "boosts": {},
    }


def record(
    team: tuple[str, ...] = ("a",), lead: str = "b", p2: tuple[str, ...] = ("b",)
) -> dict[str, Any]:
    timeline = [
        {
            "turn": i + 1,
            "p1_pokemon_state": state(team[0]),
            "p1_move_details": None,
            "p2_pokemon_state": state(name),
            "p2_move_details": None,
        }
        for i, name in enumerate(p2)
    ]
    return {
        "battle_id": 7,
        "p1_team_details": [details(name) for name in team],
        "p2_lead_details": details(lead),
        "battle_timeline": timeline,
    }


def test_builder_reads_team_and_lead_and_drops_placeholder_types() -> None:
    pokedex = PokedexBuilder().add_records([record()]).build()
    assert sorted(s.name for s in pokedex) == ["a", "b"]
    assert pokedex.get("a").types == ("normal",)


def test_builder_ignores_species_only_in_the_timeline() -> None:
    pokedex = PokedexBuilder().add_records([record(p2=("b", "hidden"))]).build()
    assert pokedex.find("hidden") is None
    with pytest.raises(UnknownSpeciesError):
        pokedex.get("hidden")


def test_builder_refuses_contradicting_stats() -> None:
    other = record()
    other["p2_lead_details"]["base_hp"] = 51
    with pytest.raises(PokedexError):
        PokedexBuilder().add_records([record(), other])


def test_json_round_trip_keeps_species_and_source_hash(tmp_path: Path) -> None:
    pokedex = PokedexBuilder().add_records([record()]).build("abc")
    pokedex.to_json(tmp_path / "cache" / "species.json")
    loaded = Pokedex.from_json(tmp_path / "cache" / "species.json")
    assert list(loaded) == sorted(pokedex, key=lambda s: s.name)
    assert loaded.source_sha256 == "abc"


def test_from_json_says_how_to_build_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(PokedexError, match="build-pokedex"):
        Pokedex.from_json(tmp_path / "species.json")


def test_for_battle_adds_the_stats_of_the_battle_only() -> None:
    pokedex = PokedexBuilder().add_records([record()]).build()
    battle = pokedex.for_battle(record(team=("a", "new")))
    assert battle.find("new") is not None
    assert pokedex.find("new") is None


def test_for_battle_without_new_species_is_the_same_object() -> None:
    pokedex = PokedexBuilder().add_records([record()]).build()
    assert pokedex.for_battle(record()) is pokedex


def test_for_battle_refuses_contradicting_stats() -> None:
    pokedex = PokedexBuilder().add_records([record()]).build()
    other = record()
    other["p2_lead_details"]["base_hp"] = 51
    with pytest.raises(PokedexError):
        pokedex.for_battle(other)


def test_policies() -> None:
    battle = record(p2=("b", "hidden"))
    pokedex = PokedexBuilder().add_records([battle]).build()
    assert pokedex.lookup(battle, Missing())("hidden") is None
    with pytest.raises(UnknownSpeciesError, match="battle 7"):
        pokedex.lookup(battle, Strict())("hidden")


def test_species_without_stats_counts_battles() -> None:
    known = PokedexBuilder().add_records([record()]).build()
    battles = [record(p2=("b", "hidden")), record(p2=("b", "hidden")), record()]
    assert species_without_stats(battles, known) == {"hidden": 2}


def test_placeholder_stats_are_nan() -> None:
    unknown = PokemonGen2.without_stats("hidden")
    assert not unknown.has_stats
    assert unknown.types == ()
    assert math.isnan(unknown.max_hp(100))
    assert math.isnan(unknown.stat("spe", 100))
    assert math.isnan(unknown.base_stat_total)


@pytest.mark.parametrize("attacker_known", [True, False])
def test_damage_is_not_computed_without_stats(attacker_known: bool) -> None:
    known = PokemonGen2("known", ("normal",), 50, 50, 50, 50, 50, 50)
    unknown = PokemonGen2.without_stats("hidden")
    attacker, defender = (known, unknown) if attacker_known else (unknown, known)
    damage = DamageCalculator(TypeChart.from_csv(ROOT / "assets" / "type_chart_gen2.csv"))
    assert damage.roll(Combatant(attacker), Combatant(defender), TACKLE) is None
    assert (
        damage.expected(Combatant(attacker), Combatant(defender), TACKLE, with_accuracy=True)
        is None
    )
    assert damage.hits_to_ko(Combatant(attacker), Combatant(defender), TACKLE) is None


def test_combatant_stat_stays_nan_through_the_clamp() -> None:
    assert math.isnan(Combatant(PokemonGen2.without_stats("hidden")).stat("spe"))


def test_replay_keeps_a_battle_with_a_species_without_stats() -> None:
    battle = record(p2=("b", "hidden"))
    pokedex = PokedexBuilder().add_records([battle]).build()
    context = BattleContext.from_record(battle, pokedex.lookup(battle, Missing()))
    hidden = context.side("p2").team["hidden"]
    assert context.last.p2.active == "hidden"
    assert hidden.hp_pct == 1.0
    assert math.isnan(hidden.max_hp)
    assert math.isnan(context.new_pokemon("other").stat("spd"))


def test_replay_with_strict_refuses_it() -> None:
    battle = record(p2=("b", "hidden"))
    pokedex = PokedexBuilder().add_records([battle]).build()
    with pytest.raises(UnknownSpeciesError):
        BattleContext.from_record(battle, pokedex.lookup(battle, Strict()))


def test_nan_speed_leaves_p1_first() -> None:
    battle = record(p2=("b", "hidden"))
    pokedex = PokedexBuilder().add_records([battle]).build()
    replayed = BattleContext.from_record(battle, pokedex.lookup(battle, Missing())).battle
    assert replayed._action_order({"p1": None, "p2": None}) == ["p1", "p2"]


def test_build_pokedex_command(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    train = tmp_path / "train.jsonl"
    train.write_text(json.dumps({**record(p2=("b", "natu")), "player_won": 1}) + "\n")
    out = tmp_path / "species.json"
    assert main(["build-pokedex", str(train), "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "2 species" in printed
    assert "no stats in any record: natu (in 1 battle(s))" in printed
    assert len(Pokedex.from_json(out).source_sha256 or "") == 64
