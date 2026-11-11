# v11 on the repository's architecture: done, and the next task (`v11_fs`, the living line)

Read `AGENTS.md` first, then this file in full. The file has two parts:

1. **Next task** (below, done): `v11_fs`, the living line of v11: the label-aware steps of
   v11's members (bag logit, team effect) as features, generic models, the faithful v11 frozen;
   measure how much the numbers move against it.
2. **Done: the rebuild of v11** (the rest of the file, from "Task: rebuild v11"): kept as the
   record of what was built and verified; its decisions still hold for the faithful v11.

## Status of the rebuild (done, commits `dd6a102` … `73b9035`)

- Domain: `domain/timeline.py`, `domain/expected_damage.py`, `domain/switches.py`. Addon:
  `addons/knockouts_v11.py` (`--repair v11=results/addons/knockouts_v11/v1`, artifact built by
  `python main.py detector train-v11 --version v1 --seeds 42 1 2`). Features: one `Feature` per
  group (`features/core/v11/groups.py`, `forecast.py`, view in `battle.py`), set `v11`.
  Models: `models/v11.py` (`EstimatorModel` pipelines, `LogitBlend`), `models/ko_v11.py`.
- `python tools/v11_check.py check`: every bitwise stage at max |diff| 0 (domain, full detector,
  features 10,000 × 819 and 5,000 × 819 with the injected repair, members and blend, the
  original's CV protocol); noise stages: detector tables mean |Δp| 0.0124 (0.94 % across 0.5,
  1.0453 vs 1.0442 hidden faints per battle), end to end 110 labels apart from the dump
  (re-seeding the original's own detector `KFold` alone moves 104).
- Repository protocol (`results/evaluations/models_v11`, seeds 33 66 133, v11 repair):
  `v11_blend` accuracy 0.7192, log loss 0.5409, ROC AUC 0.7985, Brier 0.1830 (README, "v11").
- Known blemish: commit `e3ac7d0` deletes `models/core/v11_member.py` one commit early (fixed by
  `ec1b8f6`); history is not rewritten without the user's request.

## Next task: `v11_fs`, the living line of v11 (label-aware features, generic models)

**Status: done** (commits `a057863` … this file's last commit). Step 3: `v11_fs / stack_blend`
0.7167 / 0.5427 vs `v11 / v11_blend` 0.7175 / 0.5417 on seed 33 (Δ log loss +0.0011, p 0.59).
Step 4 (`results/evaluations/models_v11_fs`, two-stage code of Part A): every generic model within
noise of its twin (`stack_blend` 0.7175 / 0.5416 / 0.7980 / 0.1833 vs `v11_blend` 0.7192 /
0.5409 / 0.7985 / 0.1830, Δ accuracy −0.0016 p 0.42, Δ log loss +0.0006 p 0.56; all p ≥ 0.26),
so step 5 was not run. Step 6: 217 test labels apart from `submission_v11_blend.csv`, 227 from
the dump. README, "v11".

**Later alignment** (commits `be92611`, `968b19a`; it supersedes decisions 3 and 5 below where
they differ): `V11TeamEffect` also outputs its base logit `stack_logit_full`, `V11Splines` and
the generic `StackedLogit` are gone, the models are `boost_xgboost`, `boost_lightgbm`,
`boost_catboost`, `boost_blend` (`models/boosted.py`) and `logreg_l2`. Out-of-fold labels apart
from `v11_blend`: 4.1 % → 2.5 % (`results/evaluations/models_v11_fs_closeness`; README, "v11").
**Later still** (commit `884cf0e`): those generic models and `LogitBlend` are deleted; `v11_fs`
is read by the standard models with tuned parameters (`results/tuning/v11_fs__<model>.json`) and
blended by `ensemble_boost` (README, "Reproducing the submissions").

These decisions of the user supersede the earlier proposal of this section and the amendments of
commit `240499c` (the `learners.py` approach, the `v11fs_*` names): do not reopen them.

### Goal

Port v11 cleanly onto a **new living line**, on which every future change is made, and measure
how much the numbers move against the faithful v11:

- the two steps that today learn from the labels inside the v11 members (the bag logit `stack0`
  and the team effect `TeamEffect`) become label-aware features of the feature layer;
- the models become generic: no column name or prefix, no tie to a feature set;
- the faithful v11 (feature set `v11`, models `v11_*`, the v11 addon, `tools/v11_check.py`) is a
  **frozen** historical reference: never modified again, not even to move code.

### Decisions

1. **Faithful v11 frozen.** Never touch `features/v11.py`, `features/core/v11/*`,
   `models/v11.py`, `models/core/v11_*.py`, `models/ko_v11.py`, `addons/knockouts_v11.py`,
   `tools/v11_check.py`. `python tools/v11_check.py check --stages models cv` gives max |diff| 0
   after every commit.
2. **Earlier uncommitted work** (the `learners.py` approach): the changes to
   `models/core/v11_team.py`, `models/core/v11_stacking.py` and `tests/test_architecture.py` are
   reverted, `learners.py` is deleted; what is useful of the draft label-aware features, the
   `v11_fs` set, the `usecases.py` guard and the `features/sets.py` line is moved into the
   structure below.
3. **Naming.** "v11" stays only in the feature set's name: `v11_fs` (`features/v11_fs.py`),
   package `features/core/v11_fs/`. The models do not mention v11:
   - `stack_xgboost`, `stack_lightgbm`, `stack_catboost`: a generic `StackedLogit` (every column,
     `keep=True`, C=0.003, `StratifiedKFold(5, shuffle=True, random_state=42)` as today) →
     booster, with the hyper-parameters, seeds and threads of `v11_xgb`, `v11_lgbm`, `v11_cat`;
   - `logreg_l2`: `SimpleImputer(median) → StandardScaler → LogisticRegression(C=0.0007,
     max_iter=5000)`;
   - `stack_blend`: weighted mean of the four logits with the weights of `blend_v11` (keys per
     member: `logreg_l2`, `stack_xgboost`, `stack_lightgbm`, `stack_catboost`), no
     renormalisation.
   Code in `models/stacked.py` (one import line in `models/__init__.py`); a new `LogitBlend` in
   `models/core/logit_blend.py` (the faithful v11 keeps its own); a new generic `StackedLogit`
   in `models/core/stacking.py` (the logic of v11's, without selectors or splines): no import
   from `models/core/v11_*.py`. Twins: `logreg_l2` ↔ `v11_lr_spline`, `stack_xgboost` ↔
   `v11_xgb`, `stack_lightgbm` ↔ `v11_lgbm`, `stack_catboost` ↔ `v11_cat`, `stack_blend` ↔
   `v11_blend`.
4. **The living line owns all its code**: `features/core/v11_fs/` starts as a copy of `battle`,
   `state`, `static`, `active`, `events`, `matchup`, `sleep`, `bags`, `team`, `team_model`,
   `simulation`, `simulation2`, `switching`, `condition`, `forecast`, `groups`, `moves` of
   `features/core/v11/` (internal imports rewritten to `v11_fs`; no import from
   `features/core/v11/`). It uses the domain modules (timeline, expected damage, switches).
5. **Feature set `v11_fs`**, in this order: the label-free groups (Static, State, Active,
   Events, Matchup, WinConditions, Sleep, Duel, Behaviour, Simulation, Condition, Forecast, from
   the `v11_fs` package), then
   - `V11Splines` (label-free): the spline expansion of the 25 keys of `LR_SPLINE`
     (`SimpleImputer(median)` + `SplineTransformer(n_knots=4, degree=3)` fitted on the training
     battles), columns `spl_<key>_<k>`;
   - `V11BagLogit` (label-aware, `stack_logit_0`): the bags computed internally (`Vocab` as
     `V11Bags`), `SimpleImputer(median) → StandardScaler → LogisticRegression(C=0.01)` fitted on
     the training battles; output its decision function;
   - `V11TeamEffect` (label-aware, `tm, tm_n1, tm_n2, tm_pair, tm_npair, tm_soft, tm_nsoft,
     tm_raw1, tm_raw2`): keys computed internally (`team_keys`), the fitting and encoding logic of
     `TeamEffect` ported here (a=1, p2, link, pair, soft, raw as today), against an internal base
     logit: the regression (C=0.003) of every non-key column of `v11`, bags included (as the
     boosters' original `stack_full`), cross-fitted with `StratifiedKFold(5, shuffle=True,
     random_state=42)` on the feature's training battles.
   No raw `bs_ bh_ bm_ k_` column is output. Every model sees the same `v11_fs` table (part of
   what is measured; reported). The package lives in the feature layer: the logistic pipeline is
   built there with scikit-learn, no import from `models/`.
6. **Folds and leakage** (rules 2 and 3): `V11BagLogit` and `V11TeamEffect` have
   `uses_labels = True`, `inner_oof = True`; training rows out of fold through the grouped inner
   folds of `FeatureSet.fit_transform`; never fitted on unlabelled battles; no read of
   `player_won`.
7. **Timing**: the label-free inputs of the two label-aware features (groups and forecast) are
   fitted once and the inner-fold copies reuse them, **only** if the requested battles are a
   subset of those the cache was built on (the set of `(battle_id, fingerprint)` keys is
   stored); otherwise they are refitted.
8. **Repair** `--repair v11=results/addons/knockouts_v11/v1` for every run. `train` and
   `evaluate` refuse `--test` with `v11_fs` too (the guard of `usecases.py` covers the base names
   `v11` and `v11_fs`).

### Allowed files

New: `features/core/v11_fs/*`, `features/v11_fs.py`, `models/stacked.py`,
`models/core/logit_blend.py`, `models/core/stacking.py`, `tests/features/test_v11_fs.py`,
`tests/models/test_stacked.py`. Changes: one import line in `features/sets.py` and in
`models/__init__.py`, the guard in `usecases.py`, one entry in
`tests/features/test_feature_contract.py`, `README.md`, `AGENTS.md`, this file. Nothing else.
Never touch the original repository in `~/Downloads`; never delete `results/v11_alignment/`.

### Steps

0. This file and `AGENTS.md` (the faithful v11 frozen; future changes on `v11_fs` and the
   generic models). Commit `docs: …`.
1. Package `features/core/v11_fs/`, `V11Splines`, `V11BagLogit`, `V11TeamEffect`, set `v11_fs`.
   Tests on the fixtures: initial copy (with the same repair, the label-free group columns of
   `v11_fs` equal those of `v11` bit for bit); port (on a synthetic table, the team effect and
   bag logit logic of `v11_fs`, same base and folds, equal the frozen `TeamEffect` /
   `StackedLogit` of `models/core/`); out-of-fold training rows (a training battle's value is
   that of a copy fitted without its inner fold; labels flipped on other battles change it, on
   itself do not); save / load; `none` / `rule` repairs; the cache (a feature cloned from a
   fitted one and refitted on other battles does not reuse it); the hash-seed contract entry.
   Commit `feat(v11_fs): label-aware bag logit and team effect, splines, own feature package`.
2. Generic models; tests (fit / predict / clone / save-load, any table without v11 columns, no
   column name in the parameters). Commit `feat(models): generic stacked members and logit
   blend`.
3. Quick check (~1 h): `python main.py evaluate --features v11 v11_fs --models v11_blend
   stack_blend --seeds 33 --repair v11=results/addons/knockouts_v11/v1 --out
   results/evaluations/v11_fs_quick` (the mixed combinations ignored); if |Δ log loss| of
   `v11_fs / stack_blend` against `v11 / v11_blend` exceeds 0.01, look for a bug first.
4. Measurement, with the two-stage code of "Part A" below: one process per seed (at most two at
   a time, no `--jobs`), `python main.py evaluate --features v11_fs --models logreg_l2
   stack_xgboost stack_lightgbm stack_catboost stack_blend --seeds <seed> --repair
   v11=results/addons/knockouts_v11/v1 --out results/evaluations/models_v11_fs_s<seed>` for 33,
   66, 133 (the quick check's seed 33 does not count), merged into
   `results/evaluations/models_v11_fs` (`GymResult.load` + `GymResult.combine`); merged with
   `models_v11` (`results/logs/compare.py`, same validation battles), every model against its twin
   with the corrected t-test (same 15 folds).
5. Decomposition (only if step 4 moves beyond noise): `grouped_inner` (default True) on
   `V11BagLogit` / `V11TeamEffect`; False uses the original's ungrouped `StratifiedKFold(5,
   shuffle=True, random_state=42)` (`inner_oof = False`, cross-fit inside `_fit`). One seed.
6. End to end: `train` + `predict` of `stack_blend` on `v11_fs` with the v11 repair; label
   differences against `results/submissions/submission_v11_blend.csv` and the dump.
7. Docs: `README.md` rows (v11 section), `AGENTS.md` (commands, the exception to rule 2 only for
   the faithful v11). Commit `docs(readme): …`.
8. Report.

### Part A (done): v11_fs as a two-stage feature set (same values)

Done before step 4, to cut the evaluation time without changing a value.

- `staged_feature_set.py` (features layer): `StagedFeatureSet(name, base, stage)`, a
  `FeatureSet` of battle features then `TableFeature`s on its table (`reads`, `consumes`,
  `fit(table, labels)`, `transform(table)`); same leakage rules and inner folds as `FeatureSet`;
  satisfies `FittableFeatureSource`, so the gym, folds, `TrainedRun`, `train` and `predict` are
  unchanged. Commit `44f2b17`.
- `v11_fs` = first stage: the 14 groups of `features/core/v11_fs` in v11's exact matrix order
  (bags and team keys included); second stage: `V11Splines`, `V11BagLogit` (consumes `bs_ bh_
  bm_`), `V11TeamEffect` (consumes `k_`). The cache of label-free inputs is gone. Commit
  `14db11a`. The stateless groups are remembered on the context (`stateless.py`, commit
  `c241b35`).
- Values: identical bit for bit to the previous `v11_fs` on the fixtures and on fold 0 of seed 33
  (8,000 + 2,000 battles), every column, except `stack_logit_0` on the training rows (at most
  3.3e-15: the inner-fold copies now score their rows in one batch instead of one battle at a
  time, so BLAS sums in another order; accepted by the user). The evaluation of `stack_blend` on
  seed 33 equals `results/evaluations/v11_fs_quick/v11_fs` (metrics within 2.2e-16,
  probabilities within 1.7e-15, 0 labels apart).
- Time: features of one fold 1,626 s → 666 s (same load); `evaluate` of `stack_blend` on one
  seed 8,479 s → 4,007 s.

### Hard constraints

Stop and ask only if the golden check or `v11_check --stages models cv` fails and cannot be fixed
within the allowed files; a decision turns out impossible; an error can only be fixed outside the
allowed files; data or artifacts are missing and cannot be rebuilt with the documented commands;
the environment fails. Anything else: fix it and continue. At most two heavy processes; sleeps
≤ 5 min. Every commit after the golden check, `pytest`, ruff (check and format), `pyright`,
`v11_check --stages models cv`; explicit paths, no push, no history rewrite.

### Report

What was reverted and reused; the file map; the label-aware features (columns, folds, base logit,
cache) and the model pipelines; `v11_check` after each commit and the copy / port tests; the
tables of steps 3 and 4 (accuracy, log loss, ROC AUC, Brier of each generic model next to its
twin, Δ, corrected p, fold and seed std); step 5 if run; step 6 label differences; what moved and
why; `git log --oneline -10` and `git status`.

---

# Task: rebuild v11 on the repository's architecture (original logic, repository constructs) — done

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
- **addons**: v11's hidden-faint detector is a repair addon (`--repair v11=<artifact>`), built
  with the gym and the repository's standard `FoldPlan`, like `v3`.
- **features**: one ordinary `Feature` per v11 group in the `v11` `FeatureSet`, on
  `BattleContext`, working with **any** repair.
- **models**: the members are `EstimatorModel`s of static scikit-learn pipelines, generic by
  contract; the blend is a small logit-average `Model`.

**Fidelity requirement.** v11 keeps the original's logic everywhere (formulas, constants,
operation and iteration order, seeds, hyper-parameters, blend weights) **except one accepted
change**: the cross-fitted detector values of the training battles come from the repository's
standard folds instead of the original's `KFold` over battles (decision 2). Hence:

- **bitwise identical** (NaN equal) to the reference dumps, in this repository's `.env`: the
  domain modules, the detector fitted on all P1 rows (so every test / validation battle's
  probabilities), the feature matrices and the members / blend **when fed the original's
  detector probabilities** (injected by `tools/v11_check.py`, see step 1);
- **within noise** of the original, with v11's own repair: the training rows' detector values
  and everything downstream (end-to-end submission, metrics). Measured on the detector: mean
  |Δp| 0.012 on the 39,204 P2 candidates of the training battles, 1.0 % of them on the other
  side of 0.5, expected hidden faints per battle 1.0448 vs 1.0442; re-seeding the original's
  own `KFold` (seed 43) moves them more (0.015, 1.3 %).

## Decisions taken by the user (do not reopen)

1. **Domain additions are generic.** New modules only; no change to the logic of existing domain
   files. Name them by concept, never by version, model or feature set (no `v10`, `v11`, `ko_`
   in `domain/`); numeric choices are documented as assumptions of an estimate, with defaults.
   Pyright strict and tests in `tests/domain/`.
2. **The v11 detector uses the repository's standard `FoldPlan`** (`StratifiedGroupKFold`,
   grouped by duplicate group, stratified on the faint label), with the seeds of `--seeds`
   (`detector train-v11 --seeds 42 1 2`, as `v3`); a training battle's cross-fitted value is the
   **mean over the seeds** of its out-of-fold probabilities. It changes the logic, not the
   performance (decision based on the measurement above). Consequences:
   - **no change to `folds.py`**, no new fold plan;
   - **rule 2 holds** for the detector too (duplicates in the same fold);
   - battles that are in no validation fold because they have no usable P1 row (24 training
     battles, 15 of them with P2 candidates) get the full model's probabilities.
3. **One repair at a time, independent of the predictive models.** v11's features read
   `BattleContext.repair` and work with every repair (`none`, `rule`, `model=…` such as `v3`,
   `v11=…`). The reference is `--repair v11=<artifact>`; v11 with `v3` is a legitimate variant
   (to be compared with the corrected t-test). Other feature sets may also run with the v11
   repair.
4. **v11 is otherwise faithful to the original:**
   - the inner cross-validations of `StackedLogit` / `TeamEffect` (not grouped by duplicates)
     and of the forecast stay as in the original: declared as v11's logic (the only remaining
     exception to rule 2);
   - the detector (and the Pokédex) are fitted **once on the training file**, like `v3`, not
     inside each fold of `evaluate`.
5. **The detector artifact is built by the application with the gym**: `train_v11` in
   `detector.py`, `detector train-v11` in `main.py`. The gym, `FoldPlan` and `TrainedRun` are
   used as they are.
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
8. **A versioned v11 regression check**: `tools/v11_check.py` (`export` / `check`, like
   `tools/golden.py`, which stays unchanged), comparing with the dumps in
   `results/v11_alignment/dump/`. `AGENTS.md` lists it among the commands, to run on every change
   to the domain modules v11 uses or to v11's addon, features or models.

## Hard constraints

- Do not change the logic of existing files of `domain/`, nor `interfaces.py`, `feature_set.py`,
  `registry.py`, `gym.py`, `folds.py`, `data.py`, `artifacts.py`, `stats_tables.py`,
  `tools/golden.py`, `results/golden/`, `requirements.txt`, any other feature set or model, or the
  tests of anything that is not v11 / the new code.
- Allowed changes:
  - new `domain/timeline.py`, `domain/expected_damage.py`, `domain/switches.py` and their tests;
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
- At most two heavy processes at a time; poll long jobs with sleeps ≤ 5 min and log progress.
- **Run all steps without stopping** (no pauses to show intermediate results: they go in the
  final report). Stop and ask only for a **blocker** that needs the user: a bitwise stage of
  `tools/v11_check.py` or the golden check fails and cannot be fixed without changing v11's
  logic or a file outside "Allowed changes"; a decision of this file turns out impossible; a
  test, lint or type error that can only be fixed outside "Allowed changes"; missing data,
  dumps or artifacts that cannot be regenerated with the documented commands; an environment
  failure (library, disk, memory) you cannot resolve. Anything else (bugs in the new code,
  failing new tests, lint, typing, a slow step): fix it and continue.
- One change per commit, after golden check, `pytest`, `ruff check . && ruff format --check .`,
  `pyright`. Messages `feat(domain): …`, `feat(addons): …`, `refactor(v11): …`, `docs: …`.
  No push, no history rewrite.
- If something cannot be expressed with the repository's constructs without changing values
  (beyond decision 2), stop and ask instead of changing the logic.

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
  - scripts to reuse: `results/v11_alignment/{compare.py, cv_step3.py, port_dump.py,
    original_dump.py}` and the copy of the original in `results/v11_alignment/original/`
    (importable as `src` with `FDS_DATA_DIR=results/v11_alignment/data`).
- Facts already checked:
  - the repository Pokédex (`results/cache/species.json`) has the same 239 species as v11's
    Pokédex with identical base stats, types (`notype` dropped) and base stat total (ints);
    `Pokedex.lookup(record, Missing())` = v11's `Pokedex.for_battle(record).get` (None for a
    species without stats);
  - the type chart file is the same; `TypeChart.multiplier` raises `KeyError` on a type outside
    `domain.pokedex.TYPES`; the records contain the move type `threequestionmarks` (Curse) and
    the defender type `notype`, which v11 treats as neutral (1.0) / skipped;
  - `features/core/v11/state.p2_track` takes the max of the repair's probability and the
    recorded `fnt` status, so probabilities given only for candidate switches (as `v3`) work;
  - 24 training battles have no usable P1 row (`final & x_hp.notna()`), 15 of them with P2
    candidates.

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

### Addon: `addons/knockouts_v11.py` (`V11KnockoutRepair`, implements `TimelineRepair`)

- Uses the infrastructure of `v3`: the protocol, the repair specs (`check_spec`, `load_repair`
  accept `v11=<artifact folder>`), the artifact layout (`results/addons/knockouts_v11/<version>/`,
  `info.json`, never overwritten, a clear error when missing), `TrainedRun.load`, the run's
  recorded repair.
- `probabilities(record)`, same shape as the original's `battle.p2_faint`: for every P2 row `i`
  of `SwitchExtractor`, 1.0 if `recorded`, P(faint) if candidate, else 0.0; same keys and order.
  - **training battles of the artifact**: the cross-fitted value of the tables (mean over the
    seeds), key `(battle_id, fingerprint(record))` (training duplicates have distinct ids; test
    ids can repeat training ids);
  - **truncated training battles** (`len(timeline) == cut`, `cut` in `CUTS = (12, 14, …, 24)`):
    the cross-fitted value of that cut's detector, same key on the truncated record;
  - **any other record**: the full detector (`TrainedRun.predict_proba` on the record's P2
    candidates, one batch per record; bitwise equal to the original's scoring of the test
    battles).
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
    id = row, **group = the battle's duplicate group** (`fingerprint`), as `v3`), battles in file
    order, rows in order;
  - cross-fitted tables: for each seed of `--seeds`, `Gym([lgbm_ko_v11], [ko_v11],
    plan=FoldPlan(5, (seed,)), probes={"p2": probe})` (the standard plan, one seed per gym run,
    so the probe knows the seed); the probe receives each fold's fitted model and returns
    `(battle, i, p)` for the P2 candidates of the battles whose P1 rows are in that fold's
    validation part (recorded P2 rows 1.0, other P2 rows 0.0). The validation rows are
    `FitContext.x_valid.index` (row positions of the task: the `ko_v11` set is stateless, so the
    gym slices one shared table built with a `RangeIndex`); assert it in a test, or recompute
    the splits with `folds.make_splits` on the same plan. A battle's value is the mean over the
    seeds; battles in no validation fold (no P1 row) get
    the full model's value. The gym's out-of-fold AUC on P1 rows goes to `info.json`
    (`detector_auc`);
  - the same for each `cut` on the training battles truncated to `cut` entries;
  - full model: `gym.train(task)` → `TrainedRun` saved with `run.save(target)`;
  - `tables.pkl` (full length and per cut) and `info.json` (version, params, columns, cuts,
    seeds, `train_sha256`, `pokedex_sha256`, `detector_auc`, `detector_pos_rate`,
    `p2_hidden_per_battle`, the gym summary, timestamp).
- `main.py`: `python main.py detector train-v11 --version v1 --seeds 42 1 2` (arguments, then the
  call; the seed options as `detector train`).

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

1. **`tools/v11_check.py`** (versioned, decision 8; `check` exits non-zero on any difference
   beyond the stated tolerance and prints rows × columns and max |diff| per stage; stages that
   need code not yet written are skipped until it exists). `export` also writes, once, the
   original's cut-level detector probabilities (original `fit_detector` on `parse(train, dex,
   cut=c)`) next to the dumps. Stages:
   - **domain** (bitwise): `ObservedTimeline` / `SwitchExtractor` vs the original `Side` /
     `transitions` on all 15,000 battles;
   - **full detector** (bitwise): `V11KnockoutRepair.probabilities` of every test battle vs
     `p2_faint.pkl["test"]`;
   - **detector tables** (within noise, reported, not failing): train / cut tables vs the
     original's: mean |Δp|, share across 0.5, hidden faints per battle; expected about 0.012,
     1 %, 1.045;
   - **features** (bitwise): `build("v11").fit_transform(train contexts)` vs `X_train[cols]`,
     `.to_frame(test contexts)` vs `X_test[cols]`, with the contexts replayed with an
     **injected repair** defined in the tool (`TimelineRepair` returning the original's
     `p2_faint` for the full battles and the original's cut-level probabilities for truncated
     ones); first check that the Dataset keeps the records in file order with the same content
     minus the label;
   - **models** (bitwise): members and blend fitted on the dumped `X_train[cols]` vs the dumped
     test probabilities; the original's CV protocol (`cv_step3.py port`) on the dumped matrix vs
     `dump/cv/original_*.npy`;
   - **end to end** (within noise, reported): the run with `--repair v11=…` vs
     `dump/original/blend_v11.npy` (max |diff|, label differences) and vs
     `results/v11_ref/sub20_blend_v11.csv`.
2. **Domain modules** (`timeline.py`, `expected_damage.py`, `switches.py`, tests); the current
   port switched to them; check: domain and features stages bitwise. Commit
   `feat(domain): observed timeline, expected damage estimate, switch descriptions`.
3. **Detector addon + artifact** (`ko_v11` set, `lgbm_ko_v11`, `knockouts_v11.py`,
   `addons/__init__.py`, `train_v11`, `main.py`); `python main.py detector train-v11 --version
   v1 --seeds 42 1 2`; check: full detector bitwise, tables within noise. Commit
   `feat(addons): v11 hidden-faint detector as a repair addon`.
4. **Features on `BattleContext`** (the 14 features, `battle.py`, `features/v11.py`, delete the
   monolith); check: features stage bitwise (0 differing cells on 10,000 × 819 and 5,000 × 819)
   with the injected repair. Commit `refactor(v11): one feature per group on BattleContext and
   the repair`.
5. **Models** (`EstimatorModel` members, `LogitBlend`); check: models stage bitwise. Commit
   `refactor(v11): members as EstimatorModel pipelines, logit blend`.
6. **End to end**: `train` (with `--repair v11=results/addons/knockouts_v11/v1`) and `predict`
   twice (same files); report the end-to-end stage (expected: a few tens of labels apart from
   the dump, within noise) and the original's protocol (`cv_step3.py` on the new `X_train`)
   next to the original's numbers (expected within the seed-to-seed spread). Also run `train` +
   `predict` once with `--repair model=results/addons/knockouts/v3`, to show v11 works with
   another repair.
7. **Tests and docs.** Tests on `tests/fixtures/` only (tiny v11 artifact built in `tmp_path`):
   domain modules, addon (table hit, truncated hit, fallback, no-P1-row battles, recorded →
   1.0, keys), v11 features (columns and order, forecast out of fold, works with `none` / `rule`
   repairs, determinism), members and blend (fit / predict / clone / save-load, members on a
   non-v11 table), the `--test` guard; the v11 entry of the hash-seed contract test.
   `AGENTS.md` and `README.md` already describe the target: check them against what you built,
   fix mismatches, remove the "In progress" note and the README's "pending" line. Commit
   `docs: v11 on the repository's architecture`.
8. **Evaluation** (run it, then the report):
   `python main.py evaluate --features v11 --models v11_lr_spline v11_xgb v11_lgbm v11_cat v11_blend logreg hgb lightgbm xgboost catboost --seeds 33 66 133 --reference v11/v11_blend --repair v11=results/addons/knockouts_v11/v1 --out results/evaluations/models_v11`
   (several hours), then the learning curve / plots and the README rows as in `V11_SUB20.md`
   step 5 (merged with `results/logs/compare.py`, corrected t-test); commit `docs(readme): …`.

## Pitfalls that break bitwise identity (where it is required)

- **Battle order**: the forecast's KFold and the gym's task run over the battles in file order;
  `FeatureSet.fit` receives them in `Dataset` order (= file order).
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
- Every stage of `tools/v11_check.py`: rows × columns compared, max |diff| (0 where bitwise is
  required), and the noise statistics of the detector tables and of the end-to-end run.
- The original's protocol on the new `X_train` next to the original's numbers, the label
  differences vs the dump and vs `sub20_blend_v11.csv`, the run with the `v3` repair.
- The metrics of step 8 (repository protocol) next to the original's protocol in this `.env`:

  | pipeline | accuracy | log loss | ROC AUC | Brier |
  |---|---|---|---|---|
  | lr_spline_v11 | 0.71663 | 0.54681 | 0.79272 | 0.18547 |
  | xgb_v11 | 0.71963 | 0.54337 | 0.79647 | 0.18397 |
  | lgbm_v11 | 0.72013 | 0.54286 | 0.79673 | 0.18379 |
  | cat_v11 | 0.72043 | 0.54174 | 0.79819 | 0.18325 |
  | blend_v11 | 0.72027 (nested 0.7204) | 0.54096 | 0.79830 | 0.18305 |

- Anything that could not be expressed with the repository's constructs without changing values,
  and the accepted limits: the detector folds (decision 2), the forecast's own cross-fitting
  cache (not `FeatureSet`'s inner folds), no `@t` / `--turns` for v11, the v11 detector not
  comparable in `detector evaluate`, the addon's training-file tables; the possible future
  "fully generic members" variant (decision 7).
- End with `git log --oneline -10` and `git status`.
