"""Registries of feature sets and models, and the ``@t<N>`` suffix."""

from __future__ import annotations

import pytest

import models
from features import sets as feature_sets
from registry import ENSEMBLES, MODELS, Registry, build_feature_set, parse_name, set_name

COMPACT_V5_COLUMNS = [
    "material_healthy",
    "spikes_on_field",
    "bst_sum",
    "eff_substitute",
    "snorlax_diff",
    "p_switch_diff",
    "switch_excess_all",
    "hidden_bst",
    "te_pair_mean",
    "sim_seen_left",
    "p2_hidden_ko_unsure",
    "tr_p1",
    "tr_p2",
    "phpw_spd",
    "hpw_spd_t20",
    "smat_hp",
    "b_stat_diff",
    "p2_fainted",
]


def test_a_name_is_declared_once() -> None:
    registry = Registry[int]("thing")
    registry.add("a", 1)
    with pytest.raises(ValueError, match="already registered"):
        registry.add("a", 2)


def test_an_unknown_name_lists_the_valid_ones() -> None:
    registry = Registry[int]("thing")
    registry.add("a", 1)
    with pytest.raises(KeyError, match=r"unknown thing 'b'; choose from \['a'\]"):
        registry["b"]


@pytest.mark.parametrize(("name", "parsed"), [("core", ("core", None)), ("core@t10", ("core", 10))])
def test_turn_suffix(name: str, parsed: tuple[str, int | None]) -> None:
    assert parse_name(name) == parsed
    assert set_name(*parsed) == name


@pytest.mark.parametrize("name", ["core@t", "core@t0", "core@tx"])
def test_invalid_turn_suffix(name: str) -> None:
    with pytest.raises(ValueError, match="invalid turn"):
        parse_name(name)


def test_a_set_is_built_at_the_requested_turn() -> None:
    assert build_feature_set("core@t10").name == "core@t10"
    assert build_feature_set("core", 10).name == "core@t10"
    with pytest.raises(ValueError, match="conflicts"):
        build_feature_set("core@t10", 5)


def test_a_fixed_set_refuses_a_turn() -> None:
    with pytest.raises(ValueError, match="fixed at the last turn"):
        build_feature_set("hp30@t10")


def test_revalidated_sets_keep_their_columns_and_order() -> None:
    assert feature_sets.build("compact_v5").declared_columns == COMPACT_V5_COLUMNS


@pytest.mark.parametrize(
    "base", ["compact", "compact_v2", "compact_v3", "compact_v4", "compact_v5"]
)
def test_originals_keep_the_dropped_features(base: str) -> None:
    kept = feature_sets.build(base)
    original = feature_sets.build(f"{base}_orig")
    assert original.name == f"{base}_orig"
    assert set(kept.feature_ids) < set(original.feature_ids)


def test_an_ensemble_needs_the_libraries_of_its_members() -> None:
    assert ENSEMBLES["ensemble_from_lr"] == ("gb_lr", "catboost_lr", "lightgbm_lr", "xgboost_lr")
    assert set(MODELS["ensemble_from_lr"].requires) == {"catboost", "lightgbm", "xgboost"}
    assert MODELS["logreg"].requires == ()


def test_models_without_a_search_space_are_not_tunable() -> None:
    assert "logreg" in models.search_spaces()
    assert "ensemble_from_lr" not in models.search_spaces()


def test_an_unknown_model_lists_the_valid_ones() -> None:
    with pytest.raises(KeyError, match="unknown model 'nope'"):
        models.build("nope")


def test_an_ensemble_with_a_tuned_member_is_named_tuned() -> None:
    built = models.build_tuned("ensemble_from_lr", {"gb_lr": {"learning_rate": 0.07}})
    assert built.name == "ensemble_from_lr-tuned"
