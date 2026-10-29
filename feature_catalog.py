"""Concrete features and the named feature sets built from them.

Each feature set has a builder registered in :data:`FEATURE_SETS`, so
the main program and saved runs can rebuild a set from its name::

    features = build("core")          # sampled at the last recorded turn
    features = build("core@t10")      # the same features at turn 10
    table = features.to_frame(dataset.contexts)

The ``@t<N>`` suffix samples every feature of the set on the battle
truncated after turn N (see ``turn_features``); column names stay the
same, so results at different turns can be compared and plotted.

Sets:

* ``hp30``      HP difference at the last recorded turn (fixed turn)
* ``material``  HP, alive, statused, healthy, sleep, revealed (8 columns)
* ``hazards``   Spikes placed / spun / on field, phazing (4 columns)
* ``switch``    next-switch probability of both players (2 columns)
* ``core``      material + hazards + switch
* ``compact``   the 14 features kept by the incremental selection
                (see the feature-selection report)
* ``compact_plus`` compact + team encoding and battle simulation (17)
* ``compact_ko`` the 19 features selected on the knockout-repaired replay
  (``domain.knockouts``); use it with repaired datasets (``main.py``
  always repairs)
* ``compact_v2`` compact_ko re-selected on the corrected replay (stale
  moves, per-turn knockouts): with ``p2_hidden_ko_soft``, ``ghost_ko_50``,
  ``snorlax_diff`` and ``switch_excess_all`` (19)
* ``compact_v3`` compact_v2 with ``p2_hidden_ko_det`` (the learned
  detector's expected knockouts) in place of ``p2_hidden_ko_soft`` (19)
* ``selection_pool`` compact_v3 plus every other candidate (37 columns),
  the starting pool of ``selection.py``
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from functools import cache
from pathlib import Path

from domain.battle import BattleContext
from domain.typechart import TypeChart
from feature_set import Feature, FeatureSet, FeatureValue
from features.encoding_features import OpponentSpeciesEncoding, TeamEncoding
from features.field_features import boost_sum, substitute_up, trapped
from features.hazard_features import (
    hazard_features,
    phazing_uses,
    spikes_on_field,
    spikes_placed,
    spikes_spun,
)
from features.knockout_features import (
    HiddenKnockoutProbability,
    expected_hidden_knockouts,
    ghost_knockouts,
    unsure_hidden_knockouts,
)
from features.matchup_features import active_duel
from features.material_features import (
    alive_share,
    healthy_share,
    hp_healed,
    hp_share,
    material_features,
    other_sleep_share,
    p1_mean_hp_alive,
    p2_revealed_share,
    rest_sleep_share,
    statused_share,
)
from features.move_features import move_features
from features.simulation_features import BattleSimulation
from features.switch_features import (
    NextSwitchProbability,
    SwitchPressure,
    switch_count_feature,
)
from features.team_features import HiddenTeamStrength, hpw_def, hpw_spd, snorlax_diff
from features.turn_features import AtTurn, share_difference, side_difference

TURN_SEPARATOR = "@t"
TYPE_CHART_PATH = Path(__file__).resolve().parent / "assets" / "type_chart_gen2.csv"

Builder = Callable[[int | None], FeatureSet[BattleContext]]


def hp_diff_t30(context: BattleContext) -> FeatureValue:
    """P1 total HP minus P2 total HP at the last recorded turn.

    The timelines are cut at turn 30, so for most battles this is the
    situation at turn 30; shorter battles use their final turn.
    """
    last = context.last
    return "hp_diff_t30", last.p1.hp_total - last.p2.hp_total


def hp30(turn: int | None = None) -> FeatureSet[BattleContext]:
    """Only the HP difference at the last recorded turn."""
    if turn is not None:
        raise ValueError("hp30 is fixed at the last turn: use 'material' instead")
    features = FeatureSet[BattleContext]("hp30")
    features.register(hp_diff_t30)
    return features


def material(turn: int | None = None) -> FeatureSet[BattleContext]:
    return _feature_set("material", turn, material_features(turn))


def hazards(turn: int | None = None) -> FeatureSet[BattleContext]:
    return _feature_set("hazards", turn, hazard_features(turn))


def switch(turn: int | None = None) -> FeatureSet[BattleContext]:
    return _feature_set("switch", turn, [NextSwitchProbability(turn)])


def core(turn: int | None = None) -> FeatureSet[BattleContext]:
    features = [
        *material_features(turn),
        *hazard_features(turn),
        NextSwitchProbability(turn),
    ]
    return _feature_set("core", turn, features)


def compact(turn: int | None = None) -> FeatureSet[BattleContext]:
    """The 14 features kept by the incremental selection."""
    features: list[Feature[BattleContext]] = [
        AtTurn("material_hp", side_difference(hp_share), turn),
        AtTurn("material_statused", side_difference(statused_share, bad=True), turn),
        AtTurn("p1_mean_hp_alive", p1_mean_hp_alive, turn),
        AtTurn(
            "material_asleep_rest", side_difference(rest_sleep_share, bad=True), turn
        ),
        AtTurn("spikes_on_field", spikes_on_field, turn),
        AtTurn("bst_sum", boost_sum, turn),
        AtTurn("hpw_spd", hpw_spd, turn),
        AtTurn("eff_substitute", substitute_up, turn),
        AtTurn("eff_trapped", trapped, turn),
        *move_features(type_chart(), turn),
        SwitchPressure(turn),
        HiddenTeamStrength(turn),
        OpponentSpeciesEncoding(turn),
    ]
    return _feature_set("compact", turn, features)


def compact_plus(turn: int | None = None) -> FeatureSet[BattleContext]:
    """``compact`` plus ``te_p1_team``, ``te_pair_mean``, ``sim_seen_left``."""
    extra = _feature_set(
        "extra",
        turn,
        [
            TeamEncoding(turn),
            BattleSimulation(type_chart(), turn, outputs=("sim_seen_left",)),
        ],
    )
    return compact(turn).merge(extra, name=set_name("compact_plus", turn))


def compact_ko(turn: int | None = None) -> FeatureSet[BattleContext]:
    """The 18 features on the knockout-repaired replay."""
    chart = type_chart()
    features: list[Feature[BattleContext]] = [
        AtTurn("material_hp", side_difference(hp_share), turn),
        AtTurn("material_healthy", side_difference(healthy_share), turn),
        AtTurn("spikes_on_field", spikes_on_field, turn),
        AtTurn("bst_sum", boost_sum, turn),
        AtTurn("hpw_spd", hpw_spd, turn),
        AtTurn("hpw_def", hpw_def, turn),
        AtTurn("eff_substitute", substitute_up, turn),
        AtTurn("eff_trapped", trapped, turn),
        AtTurn("hp_healed", hp_healed, turn),
        AtTurn("act_duel", active_duel(chart), turn),
        AtTurn("material_alive", side_difference(alive_share), turn),
        *move_features(chart, turn),
        SwitchPressure(turn),
        switch_count_feature(turn),
        HiddenTeamStrength(turn),
        TeamEncoding(turn),
        BattleSimulation(chart, turn, outputs=("sim_seen_left",)),
    ]
    return _feature_set("compact_ko", turn, features)


def compact_v2(turn: int | None = None) -> FeatureSet[BattleContext]:
    """compact_ko re-selected on the corrected replay (19 features)."""
    chart = type_chart()
    features: list[Feature[BattleContext]] = [
        AtTurn("material_hp", side_difference(hp_share), turn),
        AtTurn("material_healthy", side_difference(healthy_share), turn),
        AtTurn("spikes_on_field", spikes_on_field, turn),
        AtTurn("bst_sum", boost_sum, turn),
        AtTurn("hpw_spd", hpw_spd, turn),
        AtTurn("eff_substitute", substitute_up, turn),
        AtTurn("eff_trapped", trapped, turn),
        AtTurn("hp_healed", hp_healed, turn),
        AtTurn("act_duel", active_duel(chart), turn),
        AtTurn("ghost_ko_50", ghost_knockouts, turn),
        AtTurn("snorlax_diff", snorlax_diff, turn),
        *move_features(chart, turn),
        SwitchPressure(turn, outputs=("p_switch_diff", "switch_excess_all")),
        HiddenTeamStrength(turn),
        TeamEncoding(turn),
        BattleSimulation(chart, turn, outputs=("sim_seen_left",)),
        HiddenKnockoutProbability(chart, turn),
    ]
    return _feature_set("compact_v2", turn, features)


def compact_v3(turn: int | None = None) -> FeatureSet[BattleContext]:
    """compact_v2 with the learned detector's expected knockouts (19)."""
    return _feature_set("compact_v3", turn, _compact_v3_features(turn))


def selection_pool(turn: int | None = None) -> FeatureSet[BattleContext]:
    """compact_v3 and the other candidate features, for ``selection.py``.

    Features with several outputs (switch pressure, simulation) compute
    all of them here; ``selection.py`` picks columns.
    """
    chart = type_chart()
    core = [
        f
        for f in _compact_v3_features(turn)
        if not isinstance(f, SwitchPressure | BattleSimulation)
    ]
    extra: list[Feature[BattleContext]] = [
        SwitchPressure(turn, outputs=SwitchPressure.OUTPUTS),
        BattleSimulation(chart, turn),
        AtTurn("p2_hidden_ko_unsure", unsure_hidden_knockouts, turn),
        HiddenKnockoutProbability(chart, turn),
        AtTurn("hpw_def", hpw_def, turn),
        AtTurn("material_alive", side_difference(alive_share), turn),
        AtTurn("material_statused", side_difference(statused_share, bad=True), turn),
        AtTurn("p1_mean_hp_alive", p1_mean_hp_alive, turn),
        AtTurn(
            "material_asleep_rest", side_difference(rest_sleep_share, bad=True), turn
        ),
        AtTurn(
            "material_asleep_other", side_difference(other_sleep_share, bad=True), turn
        ),
        AtTurn("p2_revealed", p2_revealed_share, turn),
        AtTurn("spikes_placed", share_difference(spikes_placed), turn),
        AtTurn("spikes_spun", share_difference(spikes_spun), turn),
        AtTurn("phazing", share_difference(phazing_uses), turn),
        switch_count_feature(turn),
        NextSwitchProbability(turn),
        OpponentSpeciesEncoding(turn),
    ]
    return _feature_set("selection_pool", turn, [*core, *extra])


def _compact_v3_features(turn: int | None) -> list[Feature[BattleContext]]:
    chart = type_chart()
    return [
        AtTurn("material_hp", side_difference(hp_share), turn),
        AtTurn("material_healthy", side_difference(healthy_share), turn),
        AtTurn("spikes_on_field", spikes_on_field, turn),
        AtTurn("bst_sum", boost_sum, turn),
        AtTurn("hpw_spd", hpw_spd, turn),
        AtTurn("eff_substitute", substitute_up, turn),
        AtTurn("eff_trapped", trapped, turn),
        AtTurn("hp_healed", hp_healed, turn),
        AtTurn("act_duel", active_duel(chart), turn),
        AtTurn("ghost_ko_50", ghost_knockouts, turn),
        AtTurn("snorlax_diff", snorlax_diff, turn),
        AtTurn("p2_hidden_ko_det", expected_hidden_knockouts, turn),
        *move_features(chart, turn),
        SwitchPressure(turn, outputs=("p_switch_diff", "switch_excess_all")),
        HiddenTeamStrength(turn),
        TeamEncoding(turn),
        BattleSimulation(chart, turn, outputs=("sim_seen_left",)),
    ]


@cache
def type_chart() -> TypeChart:
    """The Gen 2 type chart, loaded once."""
    return TypeChart.from_csv(TYPE_CHART_PATH)


FEATURE_SETS: dict[str, Builder] = {
    "hp30": hp30,
    "material": material,
    "hazards": hazards,
    "switch": switch,
    "core": core,
    "compact": compact,
    "compact_plus": compact_plus,
    "compact_ko": compact_ko,
    "compact_v2": compact_v2,
    "compact_v3": compact_v3,
    "selection_pool": selection_pool,
}


def set_name(base: str, turn: int | None) -> str:
    """``base`` or ``base@t<turn>``."""
    return base if turn is None else f"{base}{TURN_SEPARATOR}{turn}"


def parse_name(name: str) -> tuple[str, int | None]:
    """Split ``"core@t10"`` into ``("core", 10)``; no suffix gives None."""
    base, separator, turn = name.partition(TURN_SEPARATOR)
    if not separator:
        return name, None
    if not turn.isdigit() or int(turn) < 1:
        raise ValueError(f"invalid turn in feature set name {name!r}")
    return base, int(turn)


def build(name: str, turn: int | None = None) -> FeatureSet[BattleContext]:
    """Build a feature set from its name (``"core"``, ``"core@t10"``).

    ``turn`` can be given separately instead of the ``@t`` suffix.
    """
    base, parsed = parse_name(name)
    if parsed is not None and turn is not None and parsed != turn:
        raise ValueError(f"{name!r} conflicts with turn={turn}")
    try:
        builder = FEATURE_SETS[base]
    except KeyError:
        choices = sorted(FEATURE_SETS)
        raise KeyError(f"unknown feature set {base!r}; choose from {choices}") from None
    features = builder(parsed if parsed is not None else turn)
    expected = set_name(base, parsed if parsed is not None else turn)
    if features.name != expected:
        raise ValueError(f"builder {base!r} returned feature set {features.name!r}")
    return features


def _feature_set(
    base: str, turn: int | None, features: Iterable[Feature[BattleContext]]
) -> FeatureSet[BattleContext]:
    feature_set = FeatureSet[BattleContext](set_name(base, turn))
    feature_set.register_all(features)
    return feature_set
