# Task: rebuild v11 on the repository's architecture (same numbers, repository constructs)

Read `AGENTS.md` first, then this file in full. This task supersedes `V11_SUB20.md` (done: the
current port was verified bit for bit against the original; see "Starting point").

## Goal

Today v11 is a **foreign body** in the repository: one monolithic `Feature` (`V11Features`) runs
a copy of the original `build()` on the raw records, with its own Pokédex, its own type chart, its
own hidden-faint detector fitted inside the feature, and a private model wrapper. Rebuild it so
that it follows the layered architecture of `AGENTS.md`:

- **domain**: v11 uses the repository's `domain` as it is (Pokédex, type chart, records,
  `BattleContext`, the `TimelineRepair` protocol). **No file of `domain/` changes.**
- **addons**: v11's hidden-faint detector becomes a repair addon (`--repair v11=<artifact>`),
  like the repository's own detector; its artifact is built by an application command.
- **features**: the v11 columns come from ordinary `Feature`s, one per group, registered in the
  `v11` `FeatureSet`, computed from `BattleContext` (record, species, repair).
- **models**: the members are `EstimatorModel`s of static scikit-learn pipelines; the blend is a
  small logit-average `Model`.

**Hard requirement: the numbers do not move.** Same logic, same formulas, constants, operation
and iteration order, seeds, hyper-parameters, blend weights. Every stage must stay **bitwise
identical** (NaN equal) to the reference dumps listed below, in this repository's `.env`.

## Decisions already taken by the user (do not reopen)

1. **The artifact of the v11 detector is built by the application**: one new function in
   `detector.py` (`train_v11`) and one new subcommand in `main.py` (`detector train-v11`). This
   is the only application change.
2. **The detector is fitted once on the training file (addon), not inside every fold.** This
   is the accepted, deliberate difference from the current port, the same compromise the
   repository makes with its own `v3` detector:
   - `train` / `predict` and the original's protocol (features once on all training battles)
     stay **bitwise identical** to the original and to the dumps;
   - in `evaluate` the training rows of a fold get the values cross-fitted on the whole
     training file and the validation rows their cross-fitted value (the current port refits
     the detector and the Pokédex inside each fold), so the `evaluate` numbers of the new v11
     are **not** expected to match the current port bit for bit. The detector never reads
     `player_won` (it learns from P1's recorded faints), so there is no label leakage; the
     difference is transductive and small. Write it in `AGENTS.md` and in the README section
     of v11.
   - The Pokédex is the repository's (`results/cache/species.json`, built on the whole training
     file), not one rebuilt per fold: same consequence, same treatment.

## Hard constraints

- Do not modify `domain/` (no new file there either), `interfaces.py`, `feature_set.py`,
  `registry.py`, `gym.py`, `folds.py`, `data.py`, `artifacts.py`, `stats_tables.py`,
  `tools/golden.py`, `results/golden/`, `requirements.txt`, any other feature set or model, or the
  tests of anything that is not v11 (exception: the one-line additions listed in step 6).
- Allowed changes:
  - `features/core/v11/*`, `features/v11.py`;
  - new addon `addons/knockouts_v11.py` and the spec parsing in `addons/__init__.py`;
  - new model file `models/ko_v11.py` (the detector's classifier), `models/v11.py`,
    `models/core/v11_*.py`;
  - application: one new function in `detector.py` and one new subcommand in `main.py`
    (`detector train-v11`), nothing else;
  - the v11 tests (`tests/features/test_v11_*.py`, `tests/models/test_v11_models.py`, new
    `tests/addons/test_knockouts_v11.py`), `AGENTS.md`, `README.md`.
- Never edit the original repository in `~/Downloads/...`; never re-export the golden reference.
- At most two heavy processes at a time; poll long jobs with sleeps ≤ 5 min and report.
- One change per commit, after golden check, `pytest`, `ruff check . && ruff format --check .`,
  `pyright`. Messages `refactor(v11): …` (value-preserving), `feat(addons): …`, `docs: …`.
  No push, no history rewrite.

## Starting point (what exists and is already verified)

- Current port: `features/core/v11/` (mechanics, replay, faint, build, feature, groups),
  `features/v11.py`, `models/core/v11_{member,stacking,team}.py`, `models/v11.py`. It is
  bitwise identical to the original `blend_v11` (replay, detector, `X_train` / `X_test`,
  members, blend, the original's 5×3 CV out-of-fold predictions).
- **Reference dumps** (git-ignored, do not delete), produced by the original in `.env`:
  - `results/v11_alignment/dump/original/p2_faint.pkl`: detector probabilities
    (`{"train": [dict per battle], "test": [...]}`, keys = timeline index of P2's switches);
  - `results/v11_alignment/dump/original/features_v10.pkl`: `X_train` (10,000 rows), `X_test`
    (5,000), `y`, `groups`; the v11 columns are the 819 of `columns_lr_spline_v11.json`
    (same order for the four members);
  - `results/v11_alignment/dump/original/member_<pipeline>.npy`, `blend_v11.npy`: test
    probabilities;
  - `results/v11_alignment/dump/cv/original_<pipeline>.npy`, `original_blend_v11.npy`:
    out-of-fold predictions of the original's protocol (`cv.make_folds`, seeds 42–44);
  - scripts to reuse: `results/v11_alignment/{compare.py, cv_step3.py, port_dump.py}`.
- Facts already checked:
  - the repository Pokédex (`results/cache/species.json`, `domain.pokedex.Pokedex`) has the
    same 239 species as v11's Pokédex with identical base stats, types (``notype`` dropped)
    and base stat total (int); `Pokedex.lookup(record, Missing())` = v11's
    `Pokedex.for_battle(record).get` (None for a species without stats);
  - the type chart file is the same; `domain.typechart.TypeChart.multiplier` raises
    `KeyError` on a type outside `domain.pokedex.TYPES`, and the records contain the move type
    `threequestionmarks` (Curse) and the defender type `notype`, which v11 treats as neutral
    (1.0) / skipped.

## Target design, layer by layer

### Domain (unchanged, only used)

| v11 needs | repository construct |
|---|---|
| raw record (timeline, team details, levels, items) | `BattleContext.record` (label-free copy) |
| species stats (`dex.get(name)` → dict or None) | `BattleContext.species(name)` → `PokemonGen2 \| None` |
| type multipliers | `features.common.type_chart()` (`domain.typechart.TypeChart`) |
| `b.p2_faint` (P2 switch index → P(faint)) | `BattleContext.repair.probabilities(context.record)` |
| truncated timelines (forecast cuts) | `BattleContext.at_turn(cut)` (replays `timeline[:cut]` with the same repair) |
| content key of a battle | `domain.records.fingerprint` |

### Addon: `addons/knockouts_v11.py` (`V11KnockoutRepair`, implements `TimelineRepair`)

- `probabilities(record)` returns exactly the original's `battle.p2_faint`: for every P2
  transition row `i` (`Battle.transitions("p2")`), 1.0 if `recorded`, the detector's P(faint)
  if `not recorded and final`, else 0.0; same keys, same insertion order.
  - **Training battles of the artifact** get the **cross-fitted** value stored in the artifact
    (the original's `fit_detector` on the training file: `KFold(5, shuffle=True,
    random_state=42)` over the battles in file order). Key: `(battle_id, fingerprint(record))`
    (training duplicates have distinct ids and can fall in different folds; test ids can repeat
    training ids).
  - **Truncated training battles** (`len(timeline) == cut` for `cut` in
    `forecast.CUTS = (12, 14, ..., 24)`) get the cross-fitted value of the detector refitted
    on the truncated training battles of that cut (the original's
    `truncated_features → fit_detector(battles_cut, [])`). Key: `(battle_id,
    fingerprint(truncated record))`.
  - **Any other record** (validation / test battles) is scored with the detector fitted on all
    P1 rows of the training file (the original's `full` model).
- `repairs(record)`: the P2 indices with probability ≥ 0.5 (the v11 "hard" threshold); v11
  features never read it, it only feeds the generic domain replay.
- `describe()`: `{"name": "knockouts_v11", "version": ..., "params": {...}}` from `info.json`.
- `addons/__init__.py`: `check_spec` / `load_repair` accept `v11=<artifact folder>`.
- The rows, the columns (`META`, `astype(float)`), the candidate selection and the LightGBM
  parameters are the original's `faint.py`, unchanged. The per-battle transition rows come from
  `features/core/v11/transitions.py` (see features); addons may import features.

### Application: the v11 detector artifact

**Use the detector infrastructure that already exists; do not build a parallel one.** The v11
detector plugs into exactly the same pieces as the `v3` detector:
- the `domain.repair.TimelineRepair` protocol (`probabilities`, `repairs`, `describe`), applied
  by `BattleReplay` and kept in `BattleContext.repair`;
- the repair specs of `addons/__init__.py` (`check_spec`, `load_repair`), next to `model=`;
- the artifact layout and conventions of `addons/knockouts_model.py` / `detector.train`:
  `settings.results.addon(<kind>, <version>)`, `INFO_FILE` (`info.json` with version, params,
  `train_sha256`, `pokedex_sha256`, saved time), versions never overwritten, a clear
  `ArtifactError`-style message when the folder is missing;
- a classifier registered in `models/` (`@model`, `EstimatorModel`) and saved / loaded with the
  repository's `Model.save` / `Model.load` (or `artifacts.TrainedRun` if it fits without
  changing values);
- the run's recorded repair (`describe()` stored with the run, `predict` refuses another one).
Only the **fitting method** is v11's own (the original `fit_detector`: one KFold with seed 42
over the training battles, the cross-fitted tables, the seven truncated detectors, no gym
evaluation and no checks): it is part of v11's logic and must not be replaced by
`detector train`'s gym / multi-seed CV.

- `detector.py`: `train_v11(settings, version)` writes `results/addons/knockouts_v11/<version>/`
  (`settings.results.addon("knockouts_v11", version)`, never overwritten):
  - `model.pkl`: the full detector (`models/ko_v11.py`, fitted on every P1 row of the training
    file with `final & x_hp.notna()`);
  - `tables.pkl`: the cross-fitted probabilities of the training battles (full length) and of
    each cut, keyed as above;
  - `info.json`: version, params, columns, cuts, `train_sha256`, `pokedex_sha256`, the
    original's diagnostics (`detector_auc`, `detector_pos_rate`, `p2_hidden_per_battle`),
    timestamp.
  The cross-fitting loop is a move of the original `fit_detector` (same KFold, same order).
- `main.py`: `python main.py detector train-v11 --version v1` (parses arguments, calls the
  function; nothing else).
- `models/ko_v11.py`: `@model("lgbm_ko_v11", requires=("lightgbm",))` →
  `EstimatorModel(name, LGBMClassifier(**PARAMS))` with the original's `PARAMS` (`SEED = 42`,
  `N_JOBS = 4`, deterministic, force_row_wise). No `DropEmptyColumns` (it could drop columns).

### Features: one `Feature` per group, on `BattleContext`

- `features/core/v11/mechanics.py`: keep v11's formulas (`stat_value`, `stage_mult`,
  `damage_fraction`, `best_damage`, `stab_effectiveness`, `effective_speed`, move classes) but
  **delete** v11's `Pokedex` and `type_chart()`:
  - `entry(species: PokemonGen2 | None) -> dict | None`: `{"hp", "atk", "def", "spa", "spd",
    "spe", "types", "bst"}` from `base_*`, `types`, `base_stat_total` (ints, same values);
  - `effectiveness(move_type, def_types)`: product of `type_chart().multiplier(a, d)` over the
    defender's types, skipping `notype`, and 1.0 for any pair with a type outside
    `domain.pokedex.TYPES` (equivalent to the original's `chart.get(pair, 1.0)`, since the chart
    file only holds known types).
- `features/core/v11/replay.py`: `Side` unchanged; `Battle.from_context(context)` builds the view
  from `context.record` (timeline, team, lead, levels, items), `dex` = a mapping-like adapter
  `name -> entry(context.species(name))` (only `.get(name)` and `name in dex` are used: check
  with `rg "dex"`), and `p2_faint = dict(context.repair.probabilities(context.record))`.
  Memoize per context with `context.cached(("v11", "battle"), ...)` and the tracks with
  `("v11", "tracks")` (values derived only from that battle and its repair).
- `features/core/v11/transitions.py`: `Battle.transitions(side)` and the row features, moved out
  of `replay.py` (used by the addon and the application).
- Group features (each a `Feature[BattleContext]`, `output_names` known after fit for bags,
  otherwise declared), registered **in this order** (the original matrix order, which the four
  members read):

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

  The group functions (`static_features`, `state_features`, …) stay as they are; the features
  only wrap them. `cut` is **not** an output (no member reads it); the forecast uses it
  internally.
- `V11Forecast` (the original `forecast.py` + `truncated_features`):
  - `fit(battles)`: `cut = float(len(first record's timeline))`; for each `c` in `CUTS`, the
    input columns (`inputs(...)`: groups `s_ … c_` without bags / keys, then `cut`) of
    `battle.at_turn(c)` with helpers fitted on the full battles; the targets from `paths`; the
    out-of-fold `q_*` of the fitted battles (`KFold(5, shuffle=True, random_state=42)` over
    their order, indexed by battle id) cached by `(battle_id, fingerprint)`; the six regressors
    fitted on all truncated rows.
  - `prepare(battles)`: score in one batch the battles that are neither fitted nor cached
    (one `predict` per regressor, as the original's test matrix).
  - `compute(battle)`: the cached out-of-fold row for a fitted battle, else the batch row.
- `features/v11.py`: `make_set("v11", None, [...the 14 features...])`, `requires=("lightgbm",)`,
  `@t` still refused.
- **Delete** `features/core/v11/build.py`, `feature.py`, `faint.py`, v11's `Pokedex` /
  `type_chart`, the `uses_labels=True` workaround: every v11 feature is label-free
  (`requires_fit` only where the table says so).
- Rule: do not pass `--test` (unlabeled battles) to `train` / `evaluate` for v11: label-free
  features would then learn from the test battles too (repository semantics), and the numbers
  would move. The documented commands never do.

### Models

- `models/core/v11_stacking.py`: `StackedLogit` gets the columns at fit time from a selector
  (e.g. `columns=None, exclude_prefixes=("k_",)` / `prefixes=("bs_", "bh_", "bm_")`, resolved
  on `X.columns` in their order), so that a pipeline can be built before seeing the table. The
  spline keys of the base are filtered by the resolved columns, as today. Everything else
  unchanged.
- `models/v11.py`: each member is `EstimatorModel(name, Pipeline([...]))` with the same steps and
  order as today (`stack_full` → `stack0` → `team` → booster; `team_base` → `team(drop_base)` →
  `spline_lr`), the same parameters, `@model("v11_*")`. Delete `V11Member` and
  `models/core/v11_member.py` (`make` moves to `models/v11.py` or stays as a helper there).
- The blend: a `LogitBlend(Model)` (in `models/v11.py` or `models/core/v11_blend.py`):
  `expit(Σ w_i logit(clip(p_i, 1e-6, 1 - 1e-6)))` with the weights **as given** (no
  renormalisation: `EnsembleModel` averages probabilities and divides by the weight sum, which
  changes the last bits). Registered with `@model("v11_blend")`, not `@ensemble`
  (`build_tuned` rebuilds an ensemble as a probability average).

## Steps

1. **Baseline.** Note the current outputs of the port (already equal to the dumps). Create a
   scratch harness `results/v11_alignment/verify_v20.py` that, with the new code, writes and
   compares bitwise against the dumps:
   - detector: `V11KnockoutRepair.probabilities` of every train (from the table) and test
     battle vs `p2_faint.pkl` (also the truncated tables against a run of the original
     `fit_detector` on `parse(train, dex, cut=c)` for each cut);
   - features: `features.sets.build("v11").fit_transform(train contexts)` vs `X_train[cols]`
     and `.to_frame(test contexts)` vs `X_test[cols]` (same column order, NaN equal), with the
     contexts from `Dataset.from_jsonl(..., repair=<v11 repair>)`; check first that the
     Dataset keeps the 10,000 / 5,000 records in file order with the same content minus the
     label;
   - members and blend fitted on the dumped `X_train[cols]` vs the dumped test probabilities;
   - the original's CV protocol (`cv_step3.py port`) vs `dump/cv/original_*.npy`.
2. **Detector addon + artifact** (`features/core/v11/transitions.py`, `models/ko_v11.py`,
   `addons/knockouts_v11.py`, `addons/__init__.py`, `detector.train_v11`, `main.py`). Build
   `python main.py detector train-v11 --version v1`; check the detector stage of the harness
   (max |diff| 0). Commit `feat(addons): v11 hidden-faint detector as a repair addon`.
3. **Features on `BattleContext`**: `mechanics.py` on the domain Pokédex / chart,
   `Battle.from_context`, the 14 group features, `features/v11.py`; delete the monolith. Check
   the feature stage (0 differing cells on 10,000 × 819 and 5,000 × 819). Commit
   `refactor(v11): one feature per group on BattleContext, domain Pokédex and repair`.
4. **Models on `EstimatorModel`** and `LogitBlend`. Check members and blend (max |diff| 0) and
   the CV protocol stage. Commit `refactor(v11): members as EstimatorModel pipelines, logit blend`.
5. **End to end** with the repository commands (v11 repair everywhere):
   `python main.py train --features v11 --models v11_blend --repair v11=results/addons/knockouts_v11/v1 --out results/runs/v11`
   then `predict` twice (same files); compare the probabilities with
   `results/v11_alignment/dump/original/blend_v11.npy` (expected max |diff| 0) and count the
   label differences with `results/v11_ref/sub20_blend_v11.csv` (expected 97, as the original
   in `.env`).
6. **Tests and docs.**
   - Tests on `tests/fixtures/battles.jsonl` only (no `input/`, no `results/`): the addon
     (table hit, truncated hit, full-model fallback, recorded → 1.0, keys), the v11 features
     (columns and order, training rows out of fold for the forecast, determinism), the members
     and the blend (fit / predict / clone / save-load). Build a tiny v11 artifact inside the
     test (`tmp_path`).
   - Add the v11 set to the hash-seed contract test (`tests/features/test_feature_contract.py`,
     one entry, with the tiny artifact) and `knockouts_v11.py` to the files checked in
     `tests/test_architecture.py` if the test lists addons by name.
   - `AGENTS.md` and `README.md` were already updated by the user to the target state (v11
     artifact, `--repair v11=...` commands, rules 4 and 5, protocol note, "In progress:
     `V20_SUB.md`"). Check that they match what you built (commands, timings, paths), fix any
     mismatch, and remove the "In progress" note from `AGENTS.md` and the "pending" line from the
     README's v11 section (the rows come in step 7).
   - Commit `docs: v11 on the repository's architecture`.
7. **Evaluation** (the one stopped earlier), now with the v11 repair:
   `python main.py evaluate --features v11 --models v11_lr_spline v11_xgb v11_lgbm v11_cat v11_blend logreg hgb lightgbm xgboost catboost --seeds 33 66 133 --reference v11/v11_blend --repair v11=results/addons/knockouts_v11/v1 --out results/evaluations/models_v11`
   (~1.6 h per seed per pass; several hours). Then the learning curve / plots and the README rows
   as described in `V11_SUB20.md` step 5 (merged with `results/logs/compare.py`, corrected
   t-test), commit `docs(readme): …`.

## Pitfalls that break bitwise identity (check each one)

- **Battle order**: every KFold (detector, truncated detectors, forecast) runs over the battles
  in file order; `FeatureSet.fit` receives them in `Dataset` order (= file order).
- **Keys**: training duplicates share the fingerprint but not the id, and get different
  cross-fitted values; test ids may equal training ids. Always key by `(battle_id, fingerprint)`.
- **Detector dict**: same keys and order as the original `p2_faint` (all P2 transition rows, not
  only candidates); the tracks read it by index.
- **Types**: return Python floats / numpy floats from `compute` (`FeatureSet` rejects a
  non-`Real` such as `numpy.bool_`); `float(x)` is value-preserving.
- **Column order**: the 14 features in the table order; inside a group, the dict order of the
  group function; bag columns in `Vocab`'s sorted order.
- **Forecast inputs**: the original's `inputs(X_train.columns)` = non-bag groups in matrix order,
  then `cut`; the truncated rows are computed with helpers fitted on the **full** training
  battles and the detector of **that cut**.
- **`cut`**: `float(len(first training record's timeline))`, the same for every battle.
- **Index of the model table**: `FeatureSet` returns a `RangeIndex` (unique), so `StackedLogit` /
  `TeamEffect` use `StratifiedKFold`, as the original (unique battle ids). Keep it unique.
- **LightGBM batch vs row**: score new battles in batches (`prepare`); check bitwise that
  per-battle detector scoring in the addon equals the original's batch scoring.
- **Threads**: `N_JOBS = 4` everywhere, as today. No `fork` after LightGBM / XGBoost / CatBoost
  (rule 6).
- **Evaluation protocol** (decision 2, accepted): with the addon the detector is fitted once
  on the training file, as the repository's `v3`; do not try to reproduce the per-fold refit
  of the current port in `evaluate`. `train` / `predict` and the original's protocol must
  match the dumps exactly. State the difference in `AGENTS.md` and the README.

## Report

- File map: what moved where, what was deleted, what was added (per layer).
- For each stage of the harness: rows × columns compared and max |diff| (expected 0).
- The end-to-end probabilities vs the dump, the label differences vs `sub20_blend_v11.csv`.
- The metrics of step 7 (repository protocol, detector fitted once: expected close to, not
  equal to, the numbers below) next to the step-3 numbers of the original's protocol:

  | pipeline | accuracy | log loss | ROC AUC | Brier |
  |---|---|---|---|---|
  | lr_spline_v11 | 0.71663 | 0.54681 | 0.79272 | 0.18547 |
  | xgb_v11 | 0.71963 | 0.54337 | 0.79647 | 0.18397 |
  | lgbm_v11 | 0.72013 | 0.54286 | 0.79673 | 0.18379 |
  | cat_v11 | 0.72043 | 0.54174 | 0.79819 | 0.18325 |
  | blend_v11 | 0.72027 (nested 0.7204) | 0.54096 | 0.79830 | 0.18305 |

- Anything that could not be expressed with the repository's constructs without changing values,
  and how it was handled.
- End with `git log --oneline -8` and `git status`.
