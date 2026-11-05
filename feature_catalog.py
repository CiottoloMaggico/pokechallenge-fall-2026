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
  (``domain.transitions``); use it with repaired datasets (``main.py``
  always repairs)
* ``compact_v2`` compact_ko re-selected on the corrected replay (stale
  moves, per-turn knockouts): with ``p2_hidden_ko_soft``, ``ghost_ko_50``,
  ``snorlax_diff`` and ``switch_excess_all`` (19)
* ``compact_v3`` compact_v2 with ``p2_hidden_ko_det`` (the learned
  detector's expected knockouts) in place of ``p2_hidden_ko_soft`` (19)
* ``compact_v4`` compact_v3 + ``p2_hidden_ko_unsure``, the only change
  kept by ``selection.py`` on the detector-repaired replay (20)
* ``compact_v5`` compact_v4 re-selected with logistic regression after
  adding the team-pool features (``features.pool_features``): team
  ratings ``tr_p1`` / ``tr_p2`` in place of ``te_p1_team``, ``phpw_spd``
  in place of ``hpw_spd``, plus ``smat_hp``, ``hpw_spd_t20``,
  ``b_stat_diff``, ``alive_gain_10_30``, ``p2_fainted`` (26)
* ``selection_pool`` compact_v3 plus every other candidate (37 columns),
  the starting pool of ``selection.py``

Revalidation on 6 seeds x 5 folds (exp/reval_analyze.py, corrected repeated
k-fold t-test): every compact set above is now the original minus the
columns whose removal did not raise the log loss (one-sided corrected
p >= 0.20, removed one at a time, least harmful first); the originals stay
available as ``<name>_orig``. Removed: ``mv_res`` everywhere, ``act_duel``
and ``ghost_ko_50`` from v2..v5, the knockout columns ``p2_hidden_ko_soft``
/ ``p2_hidden_ko_det`` (the repair applied to every set already carries the
detector), ``hidden_bst`` and ``te_p2_mean`` from compact, ``material_hp``
(``smat_hp`` replaces it), ``hp_healed``, ``alive_gain_10_30`` and
``eff_trapped`` from v5. Columns: compact 11, v2 15, v3 15 (= v2), v4 16,
v5 18; accuracy and log loss equal to the originals within noise.
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
from features.pool_features import (
    PairEncoding,
    PoolBulk,
    TeamRating,
    alive_gain_10_30,
    p2_fainted,
    soft_material_hp,
    status_move_share_diff,
)
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


def compact_v4(turn: int | None = None) -> FeatureSet[BattleContext]:
    """compact_v3 plus the knockouts the repair left unmarked (20)."""
    features = [
        *_compact_v3_features(turn),
        AtTurn("p2_hidden_ko_unsure", unsure_hidden_knockouts, turn),
    ]
    return _feature_set("compact_v4", turn, features)


def compact_v5(turn: int | None = None) -> FeatureSet[BattleContext]:
    """compact_v4 re-selected with logistic regression on the pool features (26)."""
    chart = type_chart()
    hpw_turn = 20 if turn is None else min(20, turn)
    features: list[Feature[BattleContext]] = [
        AtTurn("material_hp", side_difference(hp_share), turn),
        AtTurn("material_healthy", side_difference(healthy_share), turn),
        AtTurn("spikes_on_field", spikes_on_field, turn),
        AtTurn("bst_sum", boost_sum, turn),
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
        PairEncoding(turn),
        BattleSimulation(chart, turn, outputs=("sim_seen_left",)),
        AtTurn("p2_hidden_ko_unsure", unsure_hidden_knockouts, turn),
        TeamRating(turn),
        PoolBulk(turn),
        AtTurn("hpw_spd_t20", hpw_spd, hpw_turn),
        AtTurn("smat_hp", soft_material_hp, turn),
        AtTurn("b_stat_diff", status_move_share_diff, turn),
        AtTurn("alive_gain_10_30", alive_gain_10_30, turn),
        AtTurn("p2_fainted", p2_fainted, turn),
    ]
    return _feature_set("compact_v5", turn, features)


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


# columns dropped by the multi-seed revalidation (see the module docstring)
REVALIDATION_DROPS: dict[str, tuple[str, ...]] = {
    "compact": ("hidden_team_strength", "opponent_species_encoding", "mv_res"),
    "compact_v2": ("hidden_knockout_probability", "ghost_ko_50", "act_duel", "mv_res"),
    "compact_v3": ("p2_hidden_ko_det", "ghost_ko_50", "act_duel", "mv_res"),
    "compact_v4": ("p2_hidden_ko_det", "ghost_ko_50", "act_duel", "mv_res"),
    "compact_v5": (
        "p2_hidden_ko_det", "material_hp", "mv_res", "hp_healed", "ghost_ko_50",
        "act_duel", "alive_gain_10_30", "eff_trapped",
    ),
}


def _revalidated(base: str, original: Builder) -> Builder:
    """``original`` without its ``REVALIDATION_DROPS`` (same set name)."""
    drops = REVALIDATION_DROPS[base]

    def build(turn: int | None = None) -> FeatureSet[BattleContext]:
        full = original(turn)
        missing = set(drops) - set(full.feature_ids)
        if missing:
            raise KeyError(f"{base}: no features {sorted(missing)} to drop")
        return full.select(
            [f for f in full.feature_ids if f not in drops], name=set_name(base, turn)
        )

    build.__doc__ = f"{base} after the multi-seed revalidation (drops {', '.join(drops)})."
    return build


def _renamed(base: str, original: Builder) -> Builder:
    """``original`` under the name ``<base>_orig``."""

    def build(turn: int | None = None) -> FeatureSet[BattleContext]:
        full = original(turn)
        return full.select(list(full.feature_ids), name=set_name(f"{base}_orig", turn))

    build.__doc__ = f"{base} as it was before the multi-seed revalidation."
    return build


def compact_v5_v11(turn: int | None = None) -> FeatureSet[BattleContext]:
    """compact_v5 + ``v11_proba`` (``features/v11_bridge.py``), for the
    ``blend_v11`` model only: v11 is trained inside each fold, the column is
    NaN on training battles. Needs the v11 repository (``V11_REPO``)."""
    from features.v11_bridge import V11Probability

    if turn is not None:
        raise ValueError("compact_v5_v11 is only defined at the last turn")
    base = FEATURE_SETS["compact_v5"](None)
    extra = _feature_set("v11", None, [V11Probability()])
    return base.merge(extra, name="compact_v5_v11")


FEATURE_SETS: dict[str, Builder] = {
    "hp30": hp30,
    "material": material,
    "hazards": hazards,
    "switch": switch,
    "core": core,
    "compact": _revalidated("compact", compact),
    "compact_plus": compact_plus,
    "compact_ko": compact_ko,
    "compact_v2": _revalidated("compact_v2", compact_v2),
    "compact_v3": _revalidated("compact_v3", compact_v3),
    "compact_v4": _revalidated("compact_v4", compact_v4),
    "compact_v5": _revalidated("compact_v5", compact_v5),
    "compact_orig": _renamed("compact", compact),
    "compact_v2_orig": _renamed("compact_v2", compact_v2),
    "compact_v3_orig": _renamed("compact_v3", compact_v3),
    "compact_v4_orig": _renamed("compact_v4", compact_v4),
    "compact_v5_orig": _renamed("compact_v5", compact_v5),
    "compact_v5_v11": compact_v5_v11,
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
