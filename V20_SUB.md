# Task: rebuild v11 on the repository's architecture (same numbers, repository constructs)

Read `AGENTS.md` first, then this file in full. This task supersedes `V11_SUB20.md` (done: the
current port was verified bit for bit against the original; see "Starting point").

## Goal

Today v11 is a **foreign body** in the repository: one monolithic `Feature` (`V11Features`) runs
a copy of the original `build()` on the raw records, with its own Pokédex, type chart, timeline
parser and hidden-faint detector (fitted inside the feature), and a private model wrapper.
Rebuild it on the layered architecture of `AGENTS.md`:

- **domain**: v11 uses the repository's `domain` (Pokédex, type chart, records, `BattleContext`,
  `TimelineRepair`) plus **three new generic modules** (observed timeline, expected damage,
  switch descriptions). The domain stays agnostic: no model, feature set or version in it.
- **folds**: a non-stratified `FoldPlan` option, used **only** by the v11 detector.
- **addons**: v11's hidden-faint detector is a repair addon (`--repair v11=<artifact>`), built
  with the gym like `v3`.
- **features**: one ordinary `Feature` per v11 group in the `v11` `FeatureSet`, on
  `BattleContext`, working with **any** repair.
- **models**: the members are `EstimatorModel`s of static scikit-learn pipelines; the blend is a
  small logit-average `Model`.

**Hard requirement: v11 replicates the original as faithfully as possible.** Same logic,
formulas, constants, operation and iteration order, seeds, hyper-parameters, blend weights.
With `--repair v11=<artifact>`, `train` / `predict` and the original's own protocol must stay
**bitwise identical** (NaN equal) to the reference dumps below, in this repository's `.env`.

## Decisions taken by the user (do not reopen)

1. **Domain additions are generic.** New modules only; no change to the logic of existing domain
   files. Name them by concept, never by version, model or feature set (no `v10`, `v11`, `ko_`
   in `domain/`); numeric choices are documented as assumptions of an estimate, with defaults.
   Pyright strict and tests in `tests/domain/`.
2. **Non-stratified folds exist only for v11.** `folds.FoldPlan` gets an option
   (`stratified: bool = True`); with `False` it is a shuffled `KFold` over the **groups** it
   receives (in order of first appearance, each row in the fold of its group). It is used
   exclusively by the v11 detector artifact: do not expose it in the CLI, `evaluate`, `tune` or
   `select` (the fold-table cache key would not tell it apart).
3. **One repair at a time, independent of the predictive models.** v11's features read
   `BattleContext.repair` and work with every repair (`none`, `rule`, `model=…` such as `v3`,
   `v11=…`). The faithful reference is `--repair v11=<artifact>`; v11 with `v3` is a legitimate
   variant (to be compared with the corrected t-test). Other feature sets may also run with the
   v11 repair.
4. **v11 is faithful to the original** (the bitwise requirement above), hence:
   - the v11 detector's cross-fitted tables use the original's folds: `KFold(5, shuffle=True,
     random_state=42)` over the **battles** (groups = battle index, not fingerprint). This is a
     **declared exception to rule 2** limited to these tables (15 of the 17 duplicate groups of
     the training file fall in different detector folds; the detector never reads
     `player_won`);
   - the inner cross-validations of `StackedLogit` / `TeamEffect` (not grouped by duplicates)
     and of the forecast stay as in the original: declared as v11's logic;
   - the detector (and the Pokédex) are fitted **once on the training file**, like `v3`, not
     inside each fold of `evaluate`. `train` / `predict` and the original's protocol are
     unaffected; the `evaluate` numbers are expected close to, not equal to, those of the old
     per-fold port.
5. **The detector artifact is built by the application with the gym**: `train_v11` in
   `detector.py`, `detector train-v11` in `main.py`. The gym and `TrainedRun` are used as they
   are.
6. **`--test` is refused with v11 in `train` / `evaluate`.** v11's label-free features must fit
   on the training battles only; `usecases.train` / `usecases.evaluate` stop with a clear message
   when a requested feature set is `v11` and `settings.test` is set (a few lines, no new
   abstraction).
7. **The v11 models are generic by contract, not tied to the `v11` set.** The steps that need
   particular columns take their column roles as explicit parameters (`TeamEffect` the team-key
   columns, default prefix `k_`; the bag `StackedLogit` its prefixes, default `bs_ bh_ bm_`;
   the full `StackedLogit` excludes the key columns) and are **skipped when the table has no
   such columns**, so on any feature set a member is at least "stacked logit + booster". No
   model names or imports the `v11` set. Moving the bag logit and the team effect into the
   feature layer (as `features/core/pool.py` does with `TeamRating`) would make the members
   fully generic but changes the numbers (other inner folds; the team effect depends on each
   member's own base logit): not done here, mention it in the report as possible future work.
8. **A versioned v11 regression check**: the step-1 harness becomes `tools/v11_check.py`
   (`export` / `check`, like `tools/golden.py`, which stays unchanged), comparing with the dumps
   in `results/v11_alignment/dump/`. `AGENTS.md` lists it among the commands, to run on every
   change to the domain modules v11 uses, to `folds.py`, to the v11 addon, features or models.

## Hard constraints

- Do not change the logic of existing files of `domain/`, nor `interfaces.py`, `feature_set.py`,
  `registry.py`, `gym.py`, `data.py`, `artifacts.py`, `stats_tables.py`, `tools/golden.py`,
  `results/golden/`, `requirements.txt`, any other feature set or model, or the tests of
  anything that is not v11 / the new code.
- Allowed changes:
  - new `domain/timeline.py`, `domain/expected_damage.py`, `domain/switches.py` and their tests;
  - `folds.py`: the `stratified` option of `FoldPlan` (default unchanged) and its test;
  - `features/core/v11/*`, `features/v11.py`, new `features/transitions/ko_v11.py` (not
    registered in `TRANSITION_SETS`);
  - new `addons/knockouts_v11.py`, the spec parsing in `addons/__init__.py`;
  - new `models/ko_v11.py`, `models/v11.py`, `models/core/v11_*.py`;
  - `detector.py` (`train_v11` and its probe), `main.py` (`detector train-v11`);
  - `usecases.py`: only the `--test` guard of decision 6;
  - new `tools/v11_check.py` (decision 8);
  - the v11 tests, new `tests/addons/test_knockouts_v11.py`, one entry in
    `tests/features/test_feature_contract.py`; `AGENTS.md`, `README.md`.
- Never edit the original repository in `~/Downloads/...`; never re-export the golden reference;
  never delete `results/v11_alignment/`.
- At most two heavy processes at a time; poll long jobs with sleeps ≤ 5 min and report.
- One change per commit, after golden check, `pytest`, `ruff check . && ruff format --check .`,
  `pyright`. Messages `feat(domain): …`, `feat(folds): …`, `feat(addons): …`,
  `refactor(v11): …`, `docs: …`. No push, no history rewrite.
- If something cannot be expressed with the repository's constructs without changing values,
  stop and ask instead of changing the logic.

## Starting point (what exists and is already verified)

- Current port: `features/core/v11/` (mechanics, replay, faint, build, feature, groups),
  `features/v11.py`, `models/core/v11_{member,stacking,team}.py`, `models/v11.py`. It is
  bitwise identical to the original `blend_v11` (replay, detector, `X_train` / `X_test`,
  members, blend, the original's 5×3 CV out-of-fold predictions).
- **Reference dumps** (git-ignored), produced by the original in `.env`:
  - `results/v11_alignment/dump/original/p2_faint.pkl`: detector probabilities
    (`{"train": [dict per battle], "test": [...]}`, keys = timeline index of P2's switches);
  - `results/v11_alignment/dump/original/features_v10.pkl`: `X_train` (10,000 rows), `X_test`
    (5,000), `y`, `groups`; the v11 columns are the 819 of `columns_lr_spline_v11.json`
    (same order for the four members);
  - `results/v11_alignment/dump/original/member_<pipeline>.npy`, `blend_v11.npy`: test
    probabilities;
  - `results/v11_alignment/dump/cv/original_<pipeline>.npy`, `original_blend_v11.npy`:
    out-of-fold predictions of the original's protocol (`cv.make_folds`, seeds 42–44);
  - scripts to reuse: `results/v11_alignment/{compare.py, cv_step3.py, port_dump.py}` and the
    copy of the original in `results/v11_alignment/original/` (importable as `src` with
    `FDS_DATA_DIR=results/v11_alignment/data`).
- Facts already checked:
  - the repository Pokédex (`results/cache/species.json`) has the same 239 species as v11's
    Pokédex with identical base stats, types (`notype` dropped) and base stat total (ints);
    `Pokedex.lookup(record, Missing())` = v11's `Pokedex.for_battle(record).get` (None for a
    species without stats);
  - the type chart file is the same; `TypeChart.multiplier` raises `KeyError` on a type outside
    `domain.pokedex.TYPES`; the records contain the move type `threequestionmarks` (Curse) and
    the defender type `notype`, which v11 treats as neutral (1.0) / skipped;
  - `features/core/v11/state.p2_track` takes the max of the repair's probability and the
    recorded `fnt` status, so probabilities given only for candidate switches (as `v3`) work.

## Target design, layer by layer

### Domain: three new generic modules (pure, stdlib + numpy, pyright strict)

Moved from the current port, made independent of v11 by name and documentation only (same code
paths, same values):

| module | content | from |
|---|---|---|
| `domain/timeline.py` | `ObservedTimeline`: what a record shows at each entry, per side: active names, HP, status, effects (`noeffect` dropped), boosts, move field; **fresh move events** (changes of the move field); moves revealed per Pokémon (P2's lead counts as revealed, from `p2_lead_details`); first / last index; Spikes on each side (laid by the opponent, cleared by Rapid Spin); `used_move(side, i)` (the move actually used at an entry) | `features/core/v11/replay.py` (`Side`, `Battle._spikes`), `simulation.used_move` |
| `domain/expected_damage.py` | a quick, roll-free damage estimate: `stat_value` (maximal DVs / stat experience, as documented), `stage_mult`, `expected_damage` (share of max HP, accuracy-weighted, mean roll factor 0.925, STAB, burn, Thick Club / Light Ball of the attacker, Explosion halving defence, fixed-damage moves), `best_damage`, `effective_speed`, `effectiveness` over a `TypeChart` that skips `notype` and treats a type outside `TYPES` as neutral. Species as `PokemonGen2`, moves as competition-format move records (mappings) | `features/core/v11/mechanics.py` |
| `domain/switches.py` | `SwitchExtractor`: every change of active Pokémon of one side described for a hidden-faint detector, the "basic" description (no physical-evidence columns, unlike `domain/transitions.py`): leaver, incoming Pokémon, opponent, the estimate of `expected_damage`; P1's recorded faints described from the row before, so they look like hidden P2 faints; `recorded`, `final`, `label`; candidates = P2 rows not recorded and final | `features/core/v11/replay.py` (`transitions`, `_transition_features`) |

- The docstrings explain how `switches.py` differs from `transitions.py` (basic description,
  simpler damage estimate) and how `timeline.py` differs from `BattleReplay` (records what is
  observed, simulates nothing).
- What is **not** domain: the merge of the timeline with the repair's probabilities and the
  tracks (`p2_track`, …), move classes used by features (`HEAL`, `SETUP`, …), simulations,
  bags, team models: they stay in `features/core/v11/`.

### Folds: `FoldPlan(stratified=False)` (only for v11)

- `stratified=True` (default) is today's `StratifiedGroupKFold`, unchanged.
- `stratified=False`: a splitter that runs `KFold(n_folds, shuffle=True, random_state=seed)` on
  the unique groups in order of first appearance and maps every row to the fold of its group;
  train / valid row indices ascending. With groups = battle index it reproduces the original
  detector folds exactly (verify).
- Used only by `detector.train_v11`. Test it in `tests/application/`.

### Addon: `addons/knockouts_v11.py` (`V11KnockoutRepair`, implements `TimelineRepair`)

- Uses the infrastructure of `v3`: the protocol, the repair specs (`check_spec`, `load_repair`
  accept `v11=<artifact folder>`), the artifact layout (`results/addons/knockouts_v11/<version>/`,
  `info.json`, never overwritten, a clear error when missing), `TrainedRun.load`, the run's
  recorded repair.
- `probabilities(record)` = the original's `battle.p2_faint`: for every P2 row `i` of
  `SwitchExtractor`, 1.0 if `recorded`, P(faint) if candidate, else 0.0; same keys and order.
  - **training battles of the artifact**: the cross-fitted value of the tables, key
    `(battle_id, fingerprint(record))` (training duplicates have distinct ids and different
    folds; test ids can repeat training ids);
  - **truncated training battles** (`len(timeline) == cut`, `cut` in `CUTS = (12, 14, …, 24)`):
    the cross-fitted value of that cut's detector, key `(battle_id, fingerprint(record))`;
  - **any other record**: the full detector (`TrainedRun.predict_proba` on the record's P2
    candidates, one batch per record; check bitwise equal to the original's batch scoring).
- `repairs(record)`: P2 rows with probability ≥ 0.5 (only feeds the generic domain replay).
- `describe()`: `{"name": "knockouts_v11", "version": ..., "params": {...}}` from `info.json`.
- Imports: `domain` (switches, records, pokedex), `features.transitions.ko_v11`, `artifacts`, as
  `knockouts_model.py` does.

### Application: the artifact, with the gym (`detector.train_v11`, `detector train-v11`)

- `features/transitions/ko_v11.py`: a `FeatureSet` (name `ko_v11`) over `SwitchExtractor` rows
  with one `Feature` returning the feature values in the original's column order (every key but
  `b, i, x, recorded, final, label`), declared in `output_names`; booleans become floats as the
  original's `astype(float)`. Not registered in `TRANSITION_SETS` (`detector train` / `evaluate`
  feed domain `Transition`s).
- `models/ko_v11.py`: `@model("lgbm_ko_v11", requires=("lightgbm",))` →
  `EstimatorModel(name, LGBMClassifier(**PARAMS))`, the original's `PARAMS` (`SEED = 42`,
  `N_JOBS = 4`, deterministic, force_row_wise); no `DropEmptyColumns`.
- `train_v11(settings, version)` writes `settings.results.addon("knockouts_v11", version)`:
  - task: `LabelledBattles` of P1's rows with `final & x_hp.notna()` (label = recorded faint,
    id = row, group = battle index), battles in file order, rows in order;
  - cross-fitted tables: `Gym([lgbm_ko_v11], [ko_v11], plan=FoldPlan(5, (42,),
    stratified=False), probes={"p2": probe})`; the probe receives each fold's fitted model and
    returns `(battle, i, p)` for the P2 candidates of the battles of that fold (recorded P2 rows
    1.0, other P2 rows 0.0). The gym's out-of-fold AUC on P1 rows is the original's
    `detector_auc`;
  - the same for each `cut` on the training battles truncated to `cut` entries;
  - full model: `gym.train(task)` → `TrainedRun` saved with `run.save(target)`;
  - optional diagnostics like `v3` (`gym.evaluate` with the default plan and `--seeds`): only
    written to `info.json`;
  - `tables.pkl` (full length and per cut) and `info.json` (version, params, columns, cuts,
    `train_sha256`, `pokedex_sha256`, `detector_auc`, `detector_pos_rate`,
    `p2_hidden_per_battle`, optional gym summary, timestamp).
- `main.py`: `python main.py detector train-v11 --version v1` (arguments, then the call).

### Features: one `Feature` per group, on `BattleContext`

- `features/core/v11/battle.py` (replaces `replay.py`): the per-battle view the group functions
  read, built from a `BattleContext`: `ObservedTimeline(context.record)`, species
  `context.species`, levels and items from the record, `p2_faint =
  dict(context.repair.probabilities(context.record))`. Memoized with `context.cached(("v11",
  ...), ...)` (derived only from that battle and its repair).
- The group functions keep their logic; species accesses become `PokemonGen2` attributes
  (`d["hp"]` → `d.base_hp`, `d["types"]` → `d.types`, `d["bst"]` → `d.base_stat_total`, same
  ints) and damage / speed / effectiveness calls go to `domain.expected_damage`. Move classes and
  simulation constants stay in `features/core/v11/`.
- Group features, registered **in this order** (the original matrix order, read by the four
  members):

  | feature | columns | fitted state (training battles only) |
  |---|---|---|
  | `V11Static` | `s_` | – |
  | `V11State` | `f_`, `t_` | – |
  | `V11Active` | `a_` | – |
  | `V11Events` | `e_` | – |
  | `V11Matchup` | `m_` | – |
  | `V11WinConditions` | `w_` | – |
  | `V11Sleep` | `x_` | – |
  | `V11Bags` | `bs_ bh_ bm_` | `Vocab` |
  | `V11TeamKeys` | `k_` | – |
  | `V11Duel` | `d_` | `MoveTable` |
  | `V11Behaviour` | `v_` | `SwitchRates` |
  | `V11Simulation` | `z_` | `MoveTable`, `TeamModel` |
  | `V11Condition` | `c_` | `TeamModel` |
  | `V11Forecast` | `q_*` (6 columns) | regressors + its own `MoveTable`, `TeamModel`, `SwitchRates` |

  `cut` is not an output (no member reads it); the forecast uses it internally.
- `V11Forecast` (the original `forecast.py` + `truncated_features`):
  - `fit(battles)`: `cut = float(len(first record's timeline))`; for each `c` in `CUTS` the
    input columns (non-bag groups `s_ … c_` in matrix order, then `cut`) of
    `battle.at_turn(c)` (replayed with the same repair, so the v11 repair gives that cut's
    table) with helpers fitted on the full battles; targets from `paths`; out-of-fold `q_*` of
    the fitted battles (`KFold(5, shuffle=True, random_state=42)` over their order, indexed by
    battle id) cached by `(battle_id, fingerprint)`; the six regressors on all truncated rows;
  - `prepare(battles)`: one batch `predict` per regressor for the other battles;
  - `compute(battle)`: the cached out-of-fold row, else the batch row.
- `features/v11.py`: `make_set("v11", None, [...the 14 features...])`, `requires=("lightgbm",)`,
  `@t` refused.
- Delete `features/core/v11/build.py`, `feature.py`, `faint.py`, `replay.py`, `mechanics.py`
  (moved to the domain), v11's `Pokedex` / `type_chart`, the `uses_labels=True` workaround:
  every v11 feature is label-free.
- Never pass `--test` to `train` / `evaluate` with v11 (label-free features would learn from the
  test battles too).

### Models

- Generic by contract (decision 7): no step names the `v11` set; column roles are parameters
  with defaults; a step whose columns are absent is skipped (identity), so the pipeline is built
  once, before seeing the table, and fits any table.
- `models/core/v11_stacking.py`: `StackedLogit` resolves its columns at fit time from a selector
  (all but the key columns, or the bag prefixes, in `X.columns` order) and passes the table
  through unchanged when the selection is empty; spline keys filtered by the resolved columns,
  as today. `spline_lr` keeps only the keys present (none → no spline block).
- `models/core/v11_team.py`: `TeamEffect` takes the key-column prefix (default `k_`) and is the
  identity (also for `drop_base`: then it drops nothing it did not add) when the key columns are
  missing; the base column is a parameter, as today.
- `models/v11.py`: each member is `EstimatorModel(name, Pipeline([...]))` with today's steps,
  order and parameters (`stack_full` → `stack0` → `team` → booster; `team_base` →
  `team(drop_base)` → `spline_lr`), `@model("v11_*")`. Delete `V11Member` /
  `models/core/v11_member.py`. On the `v11` table every step is active and the values are
  today's (check bitwise); on `compact_v5` (fixtures) the members fit and predict (test).
- The blend: `LogitBlend(Model)`: `expit(Σ w_i logit(clip(p_i, 1e-6, 1 - 1e-6)))`, weights as
  given (no renormalisation; `EnsembleModel` averages probabilities and divides by the weight
  sum). `@model("v11_blend")`, not `@ensemble` (`build_tuned` would turn it into a probability
  average).

## Steps

1. **Harness = `tools/v11_check.py`** (versioned, decision 8; `check` exits non-zero on any
   difference and prints rows × columns and max |diff| per stage). It compares bitwise against
   the dumps (stages that need code not yet written are skipped until they exist):
   - detector: `V11KnockoutRepair.probabilities` of every train battle (tables) and test battle
     vs `p2_faint.pkl`; the cut tables vs the original `fit_detector` on
     `parse(train, dex, cut=c)`;
   - folds: `FoldPlan(5, (42,), stratified=False)` with groups = battle index vs the original
     `KFold` assignment;
   - domain: `ObservedTimeline` / `SwitchExtractor` vs the original `Side` / `transitions` on all
     15,000 battles;
   - features: `build("v11").fit_transform(train contexts)` vs `X_train[cols]`,
     `.to_frame(test contexts)` vs `X_test[cols]` (contexts from `Dataset.from_jsonl(...,
     repair=<v11 repair>)`; first check that the Dataset keeps the records in file order with
     the same content minus the label);
   - members and blend on the dumped `X_train[cols]` vs the dumped test probabilities;
   - the original's CV protocol (`cv_step3.py port`) vs `dump/cv/original_*.npy`.
2. **Domain modules** (`timeline.py`, `expected_damage.py`, `switches.py`, tests); the current
   port switched to them; harness: domain and features stages at 0. Commit
   `feat(domain): observed timeline, expected damage estimate, switch descriptions`.
3. **Folds option** (`FoldPlan(stratified=False)`, test; harness folds stage). Commit
   `feat(folds): non-stratified group k-fold plan (v11 detector only)`.
4. **Detector addon + artifact** (`ko_v11` set, `lgbm_ko_v11`, `knockouts_v11.py`,
   `addons/__init__.py`, `train_v11`, `main.py`); `python main.py detector train-v11 --version
   v1`; harness detector stage at 0. Commit `feat(addons): v11 hidden-faint detector as a
   repair addon`. **Stop and show the harness results.**
5. **Features on `BattleContext`** (the 14 features, `battle.py`, `features/v11.py`, delete the
   monolith); harness features stage (0 differing cells on 10,000 × 819 and 5,000 × 819). Commit
   `refactor(v11): one feature per group on BattleContext and the repair`.
6. **Models** (`EstimatorModel` members, `LogitBlend`); harness members, blend and CV stages at
   0. Commit `refactor(v11): members as EstimatorModel pipelines, logit blend`.
7. **End to end**: `train` (with `--repair v11=results/addons/knockouts_v11/v1`) and `predict`
   twice (same files); probabilities vs `dump/original/blend_v11.npy` (max |diff| 0); label
   differences vs `results/v11_ref/sub20_blend_v11.csv` (expected 97, as the original in
   `.env`). Also run `train` + `predict` with `--repair model=results/addons/knockouts/v3` once,
   to show v11 works with another repair (numbers differ, by design). **Stop and show.**
8. **Tests and docs.** Tests on `tests/fixtures/` only (tiny v11 artifact built in `tmp_path`):
   domain modules, folds option, addon (table hit, truncated hit, fallback, recorded → 1.0,
   keys), v11 features (columns and order, forecast out of fold, works with `none` / `rule`
   repairs, determinism), members and blend (fit / predict / clone / save-load); the v11 entry
   of the hash-seed contract test; the `--test` guard (decision 6); the members on a non-v11
   table (decision 7). `AGENTS.md` and `README.md` already describe the target:
   check them against what you built, fix mismatches, remove the "In progress" note and the
   README's "pending" line. Commit `docs: v11 on the repository's architecture`.
9. **Evaluation** (stop before starting it):
   `python main.py evaluate --features v11 --models v11_lr_spline v11_xgb v11_lgbm v11_cat v11_blend logreg hgb lightgbm xgboost catboost --seeds 33 66 133 --reference v11/v11_blend --repair v11=results/addons/knockouts_v11/v1 --out results/evaluations/models_v11`
   (several hours), then the learning curve / plots and the README rows as in `V11_SUB20.md`
   step 5 (merged with `results/logs/compare.py`, corrected t-test); commit `docs(readme): …`.

## Pitfalls that break bitwise identity

- **Battle order**: every KFold (detector, cut detectors, forecast) runs over the battles in file
  order; `FeatureSet.fit` receives them in `Dataset` order (= file order).
- **Keys**: always `(battle_id, fingerprint)`; duplicates share the fingerprint, not the id.
- **Detector dict**: same keys and order as the original `p2_faint` (all P2 rows, not only
  candidates).
- **Species**: `PokemonGen2` base stats are ints, as v11's dicts; `None` for a species without
  stats (`Missing`), never `without_stats` (NaN stats would change the branches).
- **Types**: `compute` returns floats (`FeatureSet` rejects a non-`Real` such as `numpy.bool_`).
- **Column order**: the 14 features in the table order; inside a group, the dict order of the
  group function; bag columns in `Vocab`'s sorted order.
- **Forecast**: inputs = non-bag groups in matrix order, then `cut`; truncated rows with helpers
  fitted on the full battles and the detector of that cut; `cut` from the first training record.
- **Model table index**: unique (`RangeIndex`), so `StackedLogit` / `TeamEffect` use
  `StratifiedKFold`, as the original.
- **LightGBM**: batch scoring (`prepare`, one batch per record in the addon), checked bitwise.
- **Threads**: `N_JOBS = 4`; no `fork` after LightGBM / XGBoost / CatBoost (rule 6).
- **Strict typing in the domain** must not change a computation (no int → float conversions,
  same operation order).

## Report

- File map per layer: moved, deleted, added.
- For each harness stage: rows × columns compared and max |diff| (expected 0).
- End-to-end probabilities vs the dump; label differences vs `sub20_blend_v11.csv`; the run with
  the `v3` repair (metrics are not expected equal).
- The metrics of step 9 (repository protocol, detector fitted once: close to, not equal to, the
  numbers below) next to the original's protocol in this `.env`:

  | pipeline | accuracy | log loss | ROC AUC | Brier |
  |---|---|---|---|---|
  | lr_spline_v11 | 0.71663 | 0.54681 | 0.79272 | 0.18547 |
  | xgb_v11 | 0.71963 | 0.54337 | 0.79647 | 0.18397 |
  | lgbm_v11 | 0.72013 | 0.54286 | 0.79673 | 0.18379 |
  | cat_v11 | 0.72043 | 0.54174 | 0.79819 | 0.18325 |
  | blend_v11 | 0.72027 (nested 0.7204) | 0.54096 | 0.79830 | 0.18305 |

- Anything that could not be expressed with the repository's constructs without changing values,
  and the accepted limits: the forecast's own cross-fitting cache (not `FeatureSet`'s inner
  folds), no `@t` / `--turns` for v11, the v11 detector not comparable in `detector evaluate`,
  the addon's training-file tables; the possible future "fully generic members" variant
  (decision 7).
- End with `git log --oneline -10` and `git status`.
