# PokeChallenge 2026 – final refactoring

Goal: the same pipeline (replay, knockout repair, features, models, gym, CLI)
reorganised in layers with explicit interfaces, so that every part can be
tested alone and replaced without touching the others. Behaviour and results
must not change, with one deliberate exception: the Pokédex loses its manual
species file (§3.3), so the baseline is re-measured once after that step.
Otherwise the refactoring is done when `evaluate --seeds 33 66 133` on
`compact_v5` reproduces the current numbers (ensemble_from_lr 72.02 % /
0.5401, logreg 71.86 % / 0.5425) to the fourth decimal.

## 1. Layers and dependency rule

```
cli (main.py) ──► application (gym, detector, tuning, selection, statistics, storage)
                        │
          ┌─────────────┼──────────────┐
          ▼             ▼              ▼
       addons ──►   features        models      (addons may use both)
          │             │       (no domain import)
          ▼             ▼
                 domain  (pure logic + TimelineRepair protocol)
```

- `domain` imports only the standard library and numpy. It defines the protocol
  `TimelineRepair` but never imports an implementation: addons are
  passed in (`BattleReplay(repair=...)`).
- `addons` implement the domain protocol. An addon may be deterministic code
  (only `domain`) or load a trained artifact (`features`, `models`,
  `artifacts.py`); it never imports `gym`, `tuning` or the CLI.
- `features` imports `domain` and the feature interfaces; never `models`, `gym`.
- `models` imports nothing from `domain` or `features`: a model sees a
  `DataFrame` and labels, nothing else.
- `application` depends on the interfaces (protocols) of features and models,
  not on concrete classes; concrete classes are looked up in registries.
- `cli` only parses arguments, builds a `Settings` object and calls one use case.

Each rule is checked by a test (`tests/test_architecture.py`, see §8).

## 2. File tree

```
pokechallenge/
├── pyproject.toml              # ruff (pep8 + isort), pyright, pytest, optional extras
├── main.py                     # CLI only
├── assets/
│   └── type_chart_gen2.csv     # competition file, the only asset
├── domain/
│   ├── __init__.py             # public API of the package
│   ├── records.py              # BattleRecord / TurnRecord typed dicts, validation
│   ├── errors.py               # DomainError hierarchy
│   ├── pokedex.py              # Species, Pokedex, PokedexBuilder
│   ├── typechart.py            # TypeChart
│   ├── moves.py                # Move
│   ├── damage.py               # Combatant, Field, DamageCalculator
│   ├── battle.py               # BattleReplay, BattleContext, TurnState
│   ├── repair.py               # TimelineRepair protocol + NoRepair
│   ├── transitions.py          # Transition, TransitionExtractor (switch description)
│   ├── transition_mechanics.py # Gen 2 field and end-of-turn rules of the transitions
│   ├── knockouts.py            # detector column lists, deleted in step 7
│   ├── team_knowledge.py       # team and move archive read from replayed battles
│   └── __main__.py             # `python -m domain …`: build-pokedex, replay, check
├── addons/
│   ├── __init__.py             # registry: "none", "rule", "model" -> factory
│   ├── knockouts_rule.py       # deterministic repair (hand-written rule)
│   └── knockouts_model.py      # repair that loads a detector artifact (§3.8)
├── features/
│   ├── core/                   # one module per family of concrete features
│   │   ├── material.py  hazards.py  switching.py  moves.py  team.py
│   │   ├── knockouts.py  simulation.py  encoding.py  pool.py  turn.py
│   │   └── field.py            # today's root field_features.py
│   ├── compact.py              # one file per feature set (builder + docstring)
│   ├── compact_v2.py … compact_v5.py
│   └── transitions/            # feature sets of the knockout detector task
│       ├── ko_raw50.py         # the 50 original columns
│       └── ko_domain30.py      # the 30 domain-compressed columns (current)
├── models/
│   ├── core/
│   │   ├── estimator.py        # EstimatorModel adapter
│   │   ├── ensemble.py         # EnsembleModel
│   │   ├── from_logreg.py      # BoostFromLogreg
│   │   ├── logit_blend.py      # LogitBlend (model + probability column)
│   │   └── preprocessing.py    # DropEmptyColumns, require(), MissingDependencyError
│   ├── logreg.py  hgb.py  lightgbm.py  xgboost.py  catboost.py
│   ├── gb_lr.py  catboost_lr.py  lightgbm_lr.py  xgboost_lr.py
│   └── ensembles.py            # ensemble, ensemble_from_lr, …
├── interfaces.py               # Feature, FeatureSource protocols (Model, Splitter, Probe: steps 5-6)
├── feature_set.py              # FeatureSet, at the root while the v11 bridge imports it (§4.2)
├── registry.py                 # generic Registry[T] + decorators
├── artifacts.py                # TrainedRun: save / load a fitted feature set + model
├── detector.py                 # detector dataset (P1 transitions), probes, commands
├── gym.py                      # Gym, GymResult (rows, labels, groups: any table)
├── statistics.py               # corrected_ttest, summary tables
├── storage.py                  # ResultsLayout: every path under results/
├── settings.py                 # Settings dataclass (paths, seeds, folds, jobs)
├── tuning/
│   ├── tuner.py                # Tuner (Optuna), objective, pruning
│   └── params.py               # ParamsFile load / save
├── selection.py                # Selector, backward / forward selection
├── results/                    # ONLY outputs (git-ignored)
│   ├── runs/<set>__<model>/    # run.json, feature_state.pkl, model.pkl
│   ├── addons/knockouts/<version>/  # detector artifact: run files + info.json
│   ├── tuning/                 # optuna.db, <set>__<model>.json
│   ├── evaluations/<name>/     # folds.csv, predictions.csv, summary.csv, meta.json
│   ├── cache/                  # fold tables, detector matrices
│   └── submissions/
├── input/                      # train.jsonl, test.jsonl (git-ignored)
└── tests/
    ├── fixtures/               # 20 hand-picked battles (P1 faint, double KO, …)
    ├── domain/  features/  models/  application/
    └── test_architecture.py
```

`tuning/` holds code; its outputs (study database, best parameters) go to
`results/tuning/`, so `results/` stays the only place written by the program.
`exp/` and `tools/` are not part of the package: experiment scripts move to a
git-ignored `notebooks/` or are deleted once their result is in the report.

## 3. Domain

The domain answers one question: given a battle record, what was the state of
the game at every turn. It knows nothing about labels, folds or models.

### 3.1 Records (`domain/records.py`)

```python
class PokemonDetails(TypedDict):
    name: str
    level: int
    types: list[str]
    base_hp: int; base_atk: int; base_def: int; base_spa: int; base_spd: int; base_spe: int
    item: NotRequired[str | None]   # present in some records

class PokemonState(TypedDict):
    name: str
    hp_pct: float
    status: str            # "nostatus", "par", "slp", "frz", "psn", "tox", "brn", "fnt"
    effects: list[str]     # "noeffect" when empty
    boosts: dict[str, int]

class TurnRecord(TypedDict):
    turn: int
    p1_pokemon_state: PokemonState
    p1_move_details: MoveDetails | None
    p2_pokemon_state: PokemonState
    p2_move_details: MoveDetails | None

class BattleRecord(TypedDict):
    battle_id: int
    p1_team_details: list[PokemonDetails]
    p2_lead_details: PokemonDetails
    battle_timeline: list[TurnRecord]
    player_won: NotRequired[int]    # 0 / 1 in the files; absent in test

def read_records(path: Path) -> Iterator[BattleRecord]: ...
def validate(record: Mapping[str, Any]) -> BattleRecord: ...   # raises InvalidRecordError
def fingerprint(record: BattleRecord) -> str: ...               # sha1 without id and label
```

- `fingerprint` is the single definition of "same battle" (duplicate groups,
  test/train overlap, external caches). Today it exists twice (`data.py` duplicate
  groups and the v11 bridge).
- `validate` checks keys and value ranges once at load time, so the rest of
  the code can trust the types (pyright strict on `domain/`).

### 3.2 Errors (`domain/errors.py`)

```python
class DomainError(Exception): ...
class InvalidRecordError(DomainError): ...        # malformed record, with battle_id and path
class UnknownSpeciesError(DomainError): ...       # species not in the Pokédex
class ReplayError(DomainError): ...               # impossible timeline (e.g. two actives)
```

Data artifacts are not errors: they are handled by policies (§3.3, §3.6). An
error is raised only when the program cannot continue; the CLI turns
`DomainError` into a one-line message and exit code 2.

### 3.3 Pokédex (`domain/pokedex.py`)

```python
@dataclass(frozen=True)
class PokemonGen2:
    name: str
    types: tuple[str, ...]          # "notype" removed
    base_hp: float; base_atk: float; base_def: float
    base_spa: float; base_spd: float; base_spe: float   # NaN for without_stats
    @classmethod
    def without_stats(cls, name: str) -> PokemonGen2: ...
    @property
    def has_stats(self) -> bool: ...
    @property
    def base_stat_total(self) -> float: ...

SpeciesLookup = Callable[[str], PokemonGen2 | None]

class Pokedex:
    @classmethod
    def from_json(cls, path: str | Path) -> Pokedex: ...
    def to_json(self, path: str | Path) -> None: ...
    def find(self, name: str) -> PokemonGen2 | None: ...
    def get(self, name: str) -> PokemonGen2: ...      # raises UnknownSpeciesError
    def for_battle(self, record: Mapping[str, Any]) -> Pokedex: ...
    def lookup(self, record: Mapping[str, Any], unknown: UnknownSpeciesPolicy) -> SpeciesLookup: ...
    def __len__(self) -> int: ...

class PokedexBuilder:
    def add_records(self, records: Iterable[Mapping[str, Any]]) -> PokedexBuilder: ...
    def build(self, source_sha256: str | None = None) -> Pokedex: ...   # conflict -> PokedexError

class UnknownSpeciesPolicy(Protocol):
    def resolve(self, name: str, record: Mapping[str, Any], pokedex: Pokedex) -> PokemonGen2 | None: ...
```

**Data source rule.** The Pokédex is built only from battle records in the
competition format (`train.jsonl`), passed explicitly; the stats come from
`p1_team_details` and `p2_lead_details`, the only places with base stats. No
external data: no hand-written species file, no Pokédex download, no stats
from memory. By default the source is the training file only; the test file
is never used to build it. The type chart (`assets/type_chart_gen2.csv`,
supplied by the competition) is the only other input file.

- Today `build_pokedex.py` completes the extracted species with a manual JSON
  (species revealed only after P2's lead, with no stats in any record): that
  file goes away. Those species are handled by the policy instead.
- `Pokedex.for_battle(record)` adds the species whose stats are in the battle
  itself (P1 team, P2 lead): reading the input battle, not fitting on it, so
  it is allowed for validation and test battles.
- `Pokedex.lookup(record, policy)` is the `SpeciesLookup` of one battle: its
  `for_battle` species, then the policy.
- Policies: `Strict` (raise `UnknownSpeciesError`; default of `BattleReplay`, for
  callers that must refuse such a battle), `Missing` (default of `Dataset`: return
  `None`; features that need stats give NaN, damage is not computed). Neither
  `check` command uses `Strict`: they list these species instead of stopping at
  the first (§3.9). Today's seeded random replacement is dropped: it invents data.
- Removing the manual file and the random replacement can move the results:
  measure it right after this step (§9) with the corrected test and accept the
  new numbers as the regression baseline.
- `python -m domain build-pokedex input/train.jsonl --out results/cache/species.json`
  replaces `build_pokedex.py`; the JSON stores the hash of the source file.

### 3.4 Type chart, moves, damage (`typechart.py`, `moves.py`, `damage.py`)

```python
class TypeChart:
    @classmethod
    def from_csv(cls, path: Path) -> TypeChart: ...
    def effectiveness(self, move_type: str, defender: Sequence[str]) -> float: ...

@dataclass(frozen=True)
class Move:
    name: str; type: str; category: Literal["physical", "special", "status"]
    base_power: int; accuracy: float; priority: int
    @classmethod
    def from_details(cls, details: MoveDetails) -> Move: ...

@dataclass(frozen=True)
class Field:
    weather: Literal["rain", "sun", "sand"] | None = None
    screens: frozenset[str] = frozenset()        # "reflect", "lightscreen" of the defender

@dataclass(frozen=True)
class Combatant:
    species: Species
    boosts: Mapping[str, int] = {}
    status: str = "nostatus"
    def stat(self, name: str) -> float: ...      # level 100, max DV / stat exp, stages, par / brn
    @property
    def max_hp(self) -> int: ...

@dataclass(frozen=True)
class DamageRoll:
    min: float; max: float; mean: float

class DamageCalculator:
    def __init__(self, chart: TypeChart) -> None: ...
    def roll(self, attacker: Combatant, defender: Combatant, move: Move,
             field: Field = Field(), crit: bool = False) -> DamageRoll | None: ...
    def expected(self, attacker: Combatant, defender: Combatant, move: Move,
                 field: Field = Field(), *, with_accuracy: bool) -> float | None: ...
```

- Same formulas as today's `calc_damage` / `expected_damage`, wrapped in one
  object that owns the chart (no chart argument repeated everywhere).
- `with_accuracy` is keyword-only without default: today `include_accuracy`
  defaults to False but the detector passes True, so every call site states it.
- The sketches above are simplified: existing fields (`Combatant.level`,
  `hp_fraction`, `item`, `DamageRange` and its helpers) stay; when the code and
  this document disagree, the code's behaviour wins and the document is fixed.
- Tests: a table of hand-checked Gen 2 damage cases (STAB, immunity, crit,
  Reflect, sand, burn) with expected min / max.

### 3.5 Replay (`domain/battle.py`)

```python
@dataclass(frozen=True)
class SideState:
    active: str; hp_pct: float; status: str          # of the active Pokémon
    boosts: dict[str, int]; effects: tuple[str, ...]
    move: Move | None; target: str | None            # fresh moves only (stale repeats removed)
    switched: bool; voluntary_switch: bool
    n_alive: int; n_fainted: int; n_statused: int; n_unseen: int; hp_total: float
    spikes: bool; screens: tuple[str, ...]

@dataclass(frozen=True)
class TurnState:
    turn: int; p1: SideState; p2: SideState
    def side(self, name: str) -> SideState: ...

@dataclass(frozen=True)
class BattleContext:
    battle_id: Any
    record: dict[str, Any]                 # copy without player_won
    battle: Battle                         # final state (team details, weather)
    history: tuple[TurnState, ...]
    species: SpeciesLookup                 # Callable[[str], PokemonGen2 | None], for this battle
    repair: TimelineRepair = NO_REPAIR
    def at_turn(self, turn: int | None) -> BattleContext: ...   # truncated replay; None: itself

class BattleReplay:
    def __init__(self, pokedex: Pokedex, repair: TimelineRepair = NO_REPAIR,
                 unknown: UnknownSpeciesPolicy = STRICT) -> None: ...
    def replay(self, record: Mapping[str, Any]) -> BattleContext: ...
```

- Replaces `BattleContext.from_record`. `Dataset.from_jsonl` stays and goes
  through `BattleReplay`, with the `Missing` policy by default. The replay is
  deterministic: same record, same repair state, same context.
- `player_won` is removed first, so neither the repair nor a feature can read
  it.
- The repair is applied here, as today (`main._repair`), and again on each
  truncated copy made by `at_turn`: every feature sees repaired battles.
- An impossible timeline raises `ReplayError` (today: an empty one).
  `Dataset` turns replay failures into `DataError` with the battle id, except
  `UnknownSpeciesError`.

### 3.6 Repair protocol (`domain/repair.py`)

A repair fixes a data artifact of the timeline (today: hidden P2 faints). It
may be deterministic code or a model loaded from disk; the protocol does not
care.

```python
class TimelineRepair(Protocol):
    def probabilities(self, record: Mapping[str, Any]) -> Mapping[int, float]: ...
    def repairs(self, record: Mapping[str, Any]) -> Mapping[int, str]: ...
    def describe(self) -> dict[str, Any]: ...   # JSON: name, version, params (saved in runs)

class NoRepair:                                 # identity, for tests and baselines
    ...

NO_REPAIR = NoRepair()
```

- No `fit` in the protocol: deterministic repairs need none, learned ones are
  trained by `python main.py detector train` (§3.8) and only loaded here.
- A deterministic repair returns probabilities 0 / 1.
- `probabilities`: turn index → P(the P2 Pokémon that left at that row
  fainted). Used by the repair and by features (`p2_hidden_ko_unsure`, `smat_hp`).
- `repairs`: turn index → name of the Pokémon to mark as fainted
  (`probability >= threshold` in the detector). The threshold is not part of the
  protocol: features that need it compare `probabilities` with `repairs`.
- A repair never sees the outcome (`player_won`): it is label-free by contract,
  which is why a detector may be trained once on all of train.
- Results are cached inside the implementation. Today's detector
  (`DetectorRepair` in `knockout_detector.py`, the adapter until step 7) uses a
  key of the timeline (each row's actives, HP, status and move names), not
  `fingerprint`; step 7 checks that key.
- From step 7, the registry `addons/__init__.py` maps names to factories and the
  CLI selects one with `--repair none | rule | model=<artifact>` (§3.8).
- From step 4a, `describe()` is saved in every run of the main model; `predict`
  refuses a run whose repair differs from the one requested (step 7).

### 3.7 Transitions (`domain/transitions.py`)

```python
@dataclass(frozen=True)
class Transition:
    index: int          # row where the replacement appears
    leaver: str
    fainted: bool       # recorded faint (P1 always, P2 only on double KOs)
    final: bool         # the leaver never comes back
    features: Mapping[str, float]

class TransitionExtractor:
    def __init__(self, pokedex: SpeciesLookup, damage: DamageCalculator) -> None: ...
    def extract(self, record: Mapping[str, Any], side: Literal["p1", "p2"]) -> list[Transition]: ...
```

- The description of a switch moved out of `domain/knockouts.py`; the Gen 2
  field and end-of-turn rules are in `domain/transition_mechanics.py`. P1 faints
  skip the `fnt` row so they look like hidden P2 faints (the alignment variant
  was rejected and is not kept).
- `SpeciesLookup` is a callable, so an extractor is bound to one lookup: fitting
  uses `pokedex.find`, scoring a battle uses `pokedex.lookup(record, Missing())`,
  which adds the stats the battle itself carries.
- Columns: the full set (`ALL_COLUMNS`, 53) is computed; detectors choose a
  subset. Tests: one fixture per case (P1 faint, double KO, voluntary switch,
  phazing, Explosion).

### 3.8 Knockout detector: trained outside, loaded as an addon

The detector is a model like the main one, so it is developed with the same
machinery (feature sets, model registry, multi-seed folds, corrected test,
runs) on a different task, exported as a versioned artifact, and loaded at run
time by a thin addon. Changing or comparing detectors becomes `evaluate` on
the detector task, and its versions are folders under `results/addons/`.

**Dataset** (`detector.py`): the rows are P1's final switches (from
`TransitionExtractor`), the label is the recorded faint, the group is the
battle. The gym already takes any rows / labels / groups, so no new
abstraction is needed: `detector.py` only builds this table and adds the
detector checks, run on P2's candidate switches with the model fitted on all
P1 rows: P2 repairs against the symmetry estimate (recorded P1 faints −
recorded P2 faints, times the detectable share; ~10,100 today), adversarial
AUC P1 vs P2 switches, agreement with a previous artifact.

**Feature sets** (`features/transitions/`): `Feature[Transition]` objects over
the columns of `TransitionExtractor`, with the same interfaces as the battle
features (state, `get_state`, contract tests). `ko_raw50` and `ko_domain30`
reproduce the current matrices; domain codes (`x_status`, `y_code`) are
declared categorical in the feature set metadata, so the model builder can
pass them to HGB.

**Models**: the same registry; e.g. `@model("hgb_ko")` with today's detector
parameters, and any other registered model can be tried.

**Commands**

```
python main.py detector evaluate --features ko_domain30 ko_raw50 --models hgb_ko \
    --n-seeds 3 --reference ko_raw50/hgb_ko          # same tables as evaluate + probes
python main.py detector train --features ko_domain30 --models hgb_ko \
    --threshold 0.5 --version v3                      # -> results/addons/knockouts/v3/
python main.py evaluate --features compact_v5 --models ensemble_from_lr \
    --repair model=results/addons/knockouts/v3 --n-seeds 3
```

**Artifact** (`results/addons/knockouts/<version>/`): the `TrainedRun` files
(`run.json`, `feature_state.pkl`, `model.pkl`) plus `info.json`: version,
feature set, model and parameters, threshold, hash of the training file, the
`detector evaluate` summary and the probes. Versions are never overwritten.

**Addons** (`addons/`)

```python
class ModelKnockoutRepair(TimelineRepair):          # addons/knockouts_model.py
    def __init__(self, artifact: Path, pokedex: Pokedex, damage: DamageCalculator) -> None: ...
    # loads the TrainedRun once, scores the P2 candidate transitions of a record
    # with an extractor for that battle's lookup (§3.7), caches (key: §3.6);
    # describe() reads info.json

class RuleKnockoutRepair(TimelineRepair):           # addons/knockouts_rule.py
    def __init__(self, pokedex: Pokedex, damage: DamageCalculator, margin: float = 0.0) -> None: ...
    # deterministic: faint when the expected damage covers the leaver's HP and the
    # replacement entered unhurt (the compact_v2 rule); probabilities are 0 / 1
```

- Today's `knockout_detector.py` splits into the task, the feature sets, the
  model spec and `ModelKnockoutRepair`; `domain/knockouts.py` holds only the
  detector's column lists until step 7, which deletes them.
- Main-task CV with a detector trained on all of train is allowed (label-free);
  `--refit-detector-per-fold` retrains the artifact inside each outer fold, to
  check it (no difference today: 0.5420 vs 0.5419).
- The detector work must be reproducible from scratch:
  `detector train` with the same version settings must rebuild an artifact
  with identical P2 repairs.

v11 (the FDS repository) is out of scope: it will be ported natively later,
as features in `features/core/` and models in `models/`, not as a bridge to an
external codebase. The refactoring only keeps the extension points it will
need (§10). Until then the current bridge (`features/v11_bridge.py`,
`compact_v5_v11`, `tools/v11_*`) stays on the old branch, untouched.

### 3.9 Domain CLI (`domain/__main__.py`)

```
python -m domain build-pokedex input/train.jsonl --out results/cache/species.json
python -m domain replay input/train.jsonl --battle 42 [--pokedex results/cache/species.json]
python -m domain check input/train.jsonl input/test.jsonl   # unknown species, invalid rows
```

- `domain` never imports addons, so `python -m domain replay` shows the battle
  without repairs and has no `--repair` option; a repaired replay goes through
  `python main.py replay --repair rule | model=<artifact>` (step 7).
- `replay` loads the Pokédex written by `build-pokedex`, so replaying a test
  battle never builds one from the test file; `check` builds it from its first
  file only.
- `check` lists every species without stats and exits 0, since the replay
  handles them with `Missing`; it does not use `Strict`, which would stop at the
  first one. An invalid row stops it with a one-line message and exit code 2.

## 4. Features

### 4.1 Feature (`interfaces.py`)

```python
class Feature[BattleT](ABC):
    feature_id: str
    requires_fit: bool = False    # learns a state from training battles
    uses_labels: bool = False     # that state uses the outcomes
    inner_oof: bool = True        # False: never outputs in-sample values (NaN)

    @property
    def output_names(self) -> tuple[str, ...] | None: ...   # None: a FunctionFeature's column
    def fit(self, battles: Sequence[BattleT], labels: Sequence[int] | None = None) -> Self: ...
    def prepare(self, battles: Sequence[BattleT]) -> None: ...   # optional batch hook
    @abstractmethod
    def compute(self, battle: BattleT) -> Mapping[str, float]: ...
    def reset(self) -> None: ...
    def get_state(self) -> dict[str, Any]: ...
    def set_state(self, state: Mapping[str, Any]) -> None: ...
```

- `unlabeled` battles are handled by `FeatureSet.fit`, which passes them to
  label-free features as part of `battles`.
- Contract: `compute` returns exactly `output_names`, all floats (NaN allowed);
  it never sees labels. A label-aware feature must give out-of-fold values on
  its training battles (`FeatureSet.fit_transform` handles it) or NaN
  (`inner_oof = False`).
- Helpers: `FunctionFeature` (wraps `def f(context) -> float`), `AtTurn`
  (evaluates a feature on `context.at_turn(n)`), `side_difference`,
  `share_difference`.
- Tests: a generic contract test parametrised over every registered feature and
  the features only the experiments use (names, types, determinism, same output
  under two `PYTHONHASHSEED` values, `get_state` / `set_state` round trip, no
  label access in `compute`).

### 4.2 FeatureSet (`feature_set.py`)

The application depends on the protocols `FeatureSource` and
`FittableFeatureSource` of `interfaces.py`; `FeatureSet` is the concrete class.
It stays in the root `feature_set.py` because `features/v11_bridge.py` imports
`Feature` from there and must not change; moving it is decided with the bridge.

```python
class FeatureSet(Generic[BattleT]):
    name: str
    def __init__(self, name: str, features: Iterable[Feature[BattleT]] = (),
                 on_error: ErrorPolicy = ErrorPolicy.RAISE) -> None: ...
    def select(self, feature_ids: Iterable[str], name: str | None = None) -> FeatureSet[BattleT]: ...
    def without(self, feature_ids: Iterable[str], name: str | None = None) -> FeatureSet[BattleT]: ...
    def merge(self, other: FeatureSet[BattleT], name: str | None = None) -> FeatureSet[BattleT]: ...
    def clone(self) -> FeatureSet[BattleT]: ...
    @property
    def columns(self) -> tuple[str, ...]: ...
    @property
    def is_stateful(self) -> bool: ...
    def fit(self, battles, labels=None, unlabeled=()) -> Self: ...
    def fit_transform(self, battles, labels, unlabeled=(), groups=None,
                      inner_folds: int = 5, seed: int = 0) -> pd.DataFrame: ...
    def transform(self, battles: Sequence[BattleT]) -> pd.DataFrame: ...
    def get_state(self) -> dict[str, dict[str, Any]]: ...
    def set_state(self, state: Mapping[str, Mapping[str, Any]]) -> None: ...
```

- Same semantics as today (inner OOF grouped by duplicates, `prepare` before
  extraction). `without` replaced the `_revalidated` helper.
- The sketch is simplified: `on_conflict` and the other existing arguments stay.

### 4.3 Feature set files (`features/compact_v5.py`, …)

```python
@feature_set("compact_v5")
def compact_v5(turn: int | None = None) -> FeatureSet[BattleContext]:
    """18 columns after the multi-seed revalidation (originals: compact_v5_orig)."""
    return compact_v5_orig(turn).without(REVALIDATION_DROPS, name=set_name("compact_v5", turn))
```

A set fixed at the last turn (`hp30`, `compact_v5_v11`) refuses a turn in its own
builder, so the decorator needs no `turns` argument.

- One file per set with its history in the docstring (what was added, what
  was dropped, CV on seeds 42-47). `feature_catalog.py` disappears; the
  registry (`registry.py`) collects the decorated builders; `@t<N>` parsing
  stays in the registry.

## 5. Models

### 5.1 Model (`interfaces.py`)

```python
class Model(ABC):
    name: str
    def fit(self, x: pd.DataFrame, y: np.ndarray) -> Self: ...          # checks binary labels
    def predict_proba(self, x: pd.DataFrame) -> np.ndarray: ...         # 1-D P(P1 wins)
    def predict(self, x: pd.DataFrame, threshold: float = 0.5) -> np.ndarray: ...
    @abstractmethod
    def clone(self) -> Model: ...                                       # unfitted, same config
    @abstractmethod
    def get_params(self) -> dict[str, Any]: ...                         # JSON-serialisable
    @abstractmethod
    def _fit(self, x: pd.DataFrame, y: np.ndarray) -> None: ...
    @abstractmethod
    def _predict_proba(self, x: pd.DataFrame) -> np.ndarray: ...
```

Core implementations (`models/core/`): `EstimatorModel` (any scikit-learn
style estimator, column subset, column check at predict), `EnsembleModel`
(probability or logit average), `BoostFromLogreg` (LR log-odds as starting
margin for sklearn / CatBoost / LightGBM / XGBoost), `LogitBlend` (catalog
model + probability column).

### 5.2 Model specification and registry (`models/<name>.py`)

```python
@dataclass(frozen=True)
class ModelSpec:
    name: str
    build: Callable[..., Model]                       # keyword arguments override defaults
    search_space: Callable[[Trial], dict[str, Any]] | None = None
    requires: tuple[str, ...] = ()                    # optional modules (lightgbm, …)

@model("gb_lr", space=gb_lr_space)
def gb_lr(**params: Any) -> Model: ...

@ensemble("ensemble_from_lr", members=("gb_lr", "catboost_lr", "lightgbm_lr", "xgboost_lr"))
def ensemble_from_lr(**params: Any) -> Model: ...    # requires = those of its members
```

- Replaces `BUILDERS`, `SEARCH_SPACES`, `REQUIRED_MODULE`, `ENSEMBLES` of
  `model_catalog.py` with one declaration per model. `build(name, **params)`,
  `build_tuned(name, params_files)` and `available()` stay as registry functions.
- Since step 4a `registry.py` holds `ModelSpec`, `@model` and `@ensemble`, and
  `model_catalog.py` registers through them; step 5 moves the declarations.

## 6. Application

### 6.1 Splitter and fold plan

```python
class Splitter(Protocol):
    def split(self, x, y, groups) -> Iterator[tuple[np.ndarray, np.ndarray]]: ...

@dataclass(frozen=True)
class FoldPlan:
    n_folds: int = 5
    seeds: tuple[int, ...] = (42,)
    def splitters(self) -> Iterator[tuple[int, Splitter]]: ...   # StratifiedGroupKFold per seed
```

Every split is checked for leakage (no shared row, no shared duplicate group).

### 6.2 Gym (`gym.py`)

```python
class Gym:
    def __init__(self, models: Sequence[Model], feature_sets: Sequence[FeatureSet],
                 plan: FoldPlan, metrics: Mapping[str, Metric] = DEFAULT_METRICS,
                 probes: Mapping[str, Probe] = {}, jobs: int = 1) -> None: ...
    def evaluate(self, data: LabelledBattles, unlabeled: Sequence[BattleContext] = ()) -> GymResult: ...
    def train(self, data: LabelledBattles, unlabeled=()) -> list[TrainedRun]: ...
    def fold_tables(self, data, feature_set, plan) -> dict[int, list[FoldData]]: ...  # cached
```

- The gym works on any rows / labels / groups: `evaluate`, `train`, tuning and
  selection are the same for battles (main model) and switches (detector).
- `LabelledBattles` = items + labels + ids + groups, produced by the task
  (today `Dataset`).
- `jobs`: seeds in parallel with fork; disabled automatically when a model or
  feature uses OpenMP libraries (LightGBM, XGBoost, CatBoost), which hang in
  forked workers.

### 6.3 GymResult and statistics (`gym.py`, `statistics.py`)

```python
@dataclass
class GymResult:
    predictions: pd.DataFrame      # feature_set, model, seed, fold, id, y_true, proba
    folds: pd.DataFrame            # one row per (feature_set, model, seed, fold) + metrics
    meta: dict[str, Any]           # seeds, splitters, models' params, versions
    def summary(self) -> pd.DataFrame: ...          # mean, fold std, seed std
    def by_seed(self, metric: str = "accuracy") -> pd.DataFrame: ...
    def paired(self, reference: tuple[str, str], metric: str) -> pd.DataFrame: ...
    @classmethod
    def combine(cls, results: Mapping[int, GymResult]) -> GymResult: ...

def corrected_ttest(diffs: np.ndarray, test_train_ratio: float) -> tuple[float, float]: ...
```

Every comparison printed by the program uses `corrected_ttest`
(Nadeau-Bengio); the naive t-test is removed.

### 6.4 Runs and storage (`artifacts.py`, `storage.py`)

```python
class TrainedRun:
    source: FeatureSource; model: Model; columns: tuple[str, ...]
    repair: Mapping[str, Any] | None      # describe() of the repair; None in older runs
    def predict_proba(self, battles: Sequence[BattleContext]) -> np.ndarray: ...
    def save(self, layout: ResultsLayout) -> Path: ...
    @classmethod
    def load(cls, path: Path) -> TrainedRun: ...

@dataclass(frozen=True)
class ResultsLayout:
    root: Path = Path("results")
    def run(self, feature_set: str, model: str) -> Path: ...
    def evaluation(self, name: str) -> Path: ...
    def tuning(self) -> Path: ...
    def cache(self, *parts: str) -> Path: ...
    def submission(self, name: str) -> Path: ...
```

A run stores the feature set name, the columns, the feature state, the model,
and the repair state and Pokédex hash, so `predict` rebuilds exactly the same
pipeline (today the detector is refitted at predict time). `save` and `load` take
a directory until `ResultsLayout` exists (step 6).

### 6.5 Tuning (`tuning/`) and selection (`selection.py`)

```python
class Tuner:
    def __init__(self, spec: ModelSpec, folds: Sequence[FoldData], metric: str = "log_loss") -> None: ...
    def run(self, trials: int, timeout: float | None, storage: str, study: str) -> optuna.Study: ...

class ParamsFile:     # {"model", "feature_set", "params", "metric", "value", "seeds", "saved"}
    @staticmethod
    def load(path: Path) -> tuple[str, dict[str, Any]]: ...
    @staticmethod
    def save(path: Path, **info: Any) -> None: ...

class Selector:
    def __init__(self, folds: Sequence[FoldData], model: str) -> None: ...
    def score(self, columns: Sequence[str]) -> Scores: ...
    def backward(self, start: Sequence[str], p_keep: float = 0.20) -> SelectionReport: ...
    def forward(self, start: Sequence[str], pool: Sequence[str], p_add: float = 0.05) -> SelectionReport: ...
```

- Tuning on folds with seeds different from evaluation (2027 now); the final
  check of tuned parameters on unseen seeds is part of the command output.
- `backward` is the procedure of the revalidation (one column at a time,
  corrected one-sided p); `forward` the old `selection.py`.

### 6.6 Settings and CLI (`settings.py`, `main.py`)

```python
@dataclass(frozen=True)
class Settings:
    train: Path; test: Path | None; pokedex: Path
    repair: str = "model=results/addons/knockouts/v3"   # or "none", "rule"
    plan: FoldPlan = FoldPlan()
    jobs: int = 1; results: ResultsLayout = ResultsLayout()
```

Commands (unchanged names and options): `check`, `evaluate` (`--n-seeds`,
`--seeds`, `--random-seeds`, `--jobs`, `--reference`, `--repair`), `train`,
`predict`, `tune`, `select`, `replay`, and the detector group `detector evaluate`,
`detector train`, `detector tune` (same options, rows = P1 switches).

## 7. Rules that must survive the refactoring

- Only train fits anything; test is only predicted (Pokédex, detector,
  features, models). Unlabeled test battles are never passed to `fit` by default.
- Duplicates (same `fingerprint`) stay in the same fold; every split is checked.
- Label-aware features: out of fold on training rows (grouped inner folds) or NaN.
- The repair is applied to every battle before any feature; the default
  detector artifact uses the 30 domain columns, no alignment.
- The Pokédex comes only from competition-format records (train by default);
  no external species data; unknown species give NaN, never invented stats.
- Repairs never see `player_won`; every run records `describe()` of its repair.
- All comparisons with the corrected repeated k-fold t-test; multi-seed by default
  for decisions (`--n-seeds 3` at least).
- No fork with OpenMP libraries after they have been used in the parent.

## 8. Quality gates

- `ruff check` and `ruff format --check` (pep8, isort, pyupgrade, bugbear);
  line length 100.
- `pyright` strict on `domain/`, `interfaces.py`, `statistics.py`; basic elsewhere.
- `pytest`: unit tests per module, contract tests for every registered feature
  and model, an end-to-end test on `tests/fixtures/` (evaluate → train → predict
  on 200 battles, under a minute).
- `tests/test_architecture.py`: imports are scanned (ast) to enforce §1
  (`domain` imports only stdlib / numpy and never `addons`, `features`, `models`,
  `gym`; `models` never imports `domain`; `addons` never import `gym`, `tuning`
  or the CLI).
- Regression: `evaluate --features compact_v5 --models logreg ensemble_from_lr
  --seeds 33 66 133` must match the numbers in the goal above (after step 2,
  the re-measured baseline); `detector evaluate` on `ko_domain30` must match
  the out-of-fold numbers (log loss 0.0988, AUC 0.9918, 10,130 P2 repairs:
  re-measure after step 2 (10,109 repairs)).
- Comments: English, only where the code is not self-explanatory; no separator
  comments; every public class and function has a docstring.

## 9. Order of work

The steps, their prompts and acceptance checks are in `REFACTORING_STEPS.md`;
the rules for the agent are in `CLAUDE.md`. In short: tooling and golden
reference → domain values → Pokédex from train only (planned result change,
re-measured) → replay and repair protocol → interfaces, registry, features → models →
gym, CLI → knockout detector trained with the gym and loaded as a versioned addon → tuning,
selection, cleanup → quality gates. Every step must pass
`tools/golden.py check` against the reference before the next one starts.

## 10. Extension points kept for a native v11 (and other external models)

Not implemented as v11 code now, but the interfaces must allow it:

- `Feature.prepare(battles)`: batch hook, for features that compute a whole
  pipeline at once (simulations, forecasts on truncated timelines).
- `Feature.inner_oof = False`: features that never output in-sample values
  (NaN on their own training battles), for stacked predictions.
- `LogitBlend(base, column, w)`: blends any registered model with any
  probability column, not v11 specifically; its weight is chosen by a generic
  `blend-weight` command (nested CV, corrected t-test) to add when needed.
- v11's LightGBM detector becomes another model / feature set on the
  detector table, compared with `detector evaluate` and exported as a new
  artifact version; no new addon code is needed.
- Feature sets with hundreds of columns must stay cheap: `FeatureSet.transform`
  must not assume one Python call per battle per column (batch `prepare`), and
  fold tables are cached by `Gym.fold_tables`.
