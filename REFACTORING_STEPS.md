# Port of v11 (`blend_v11`) – one prompt per step

The refactoring steps 0-9 that this file held until commit `f681c2d` are finished; read them
with `git show f681c2d:REFACTORING_STEPS.md`. This file now drives the native port of the v11
model of the FDS repository (`tmp/FDS-Pokemon-Battle-Prediction-main/`, "the original").

Give the agent one step at a time, in a new session, with this preamble:

    Read `CLAUDE.md`, `REFACTORING.md` §1 and §10, and this whole file. Find the first step
    whose status (table at the end) is not `done` and do only that step. Before finishing run
    the checks under "Acceptance" and paste their output. If a check fails, try to fix the new
    code (never the original, the reference, existing tests or golden files); if it still
    fails, set the status to `blocked: <reason>` and stop. If the checks pass, set the status
    to `done (<commit>)` with the numbers asked, commit as `CLAUDE.md` says (explicit paths,
    no push, message `feat(v11 step N): <what>`) and end with `git log --oneline -3` and
    `git status`.

## What is ported

`blend_v11` = the logit average, with the fixed weights of `configs/blends.json`, of four
pipelines of `configs/pipelines.json`: `lr_spline_v11`, `xgb_v11`, `lgbm_v11`, `cat_v11`.
Their features come from `src/features/build.py` (`build`), which includes v11's own
**hidden-faint detector** (`src/faint.py`, LightGBM on P1 transitions, cross-fitted on the
training battles). Nothing of `compact_v5` is involved: "blend" means only this blend.

The goal is an exact copy: the submission written by

    python main.py train --features v11 --models v11_blend --out results/runs/v11
    python main.py predict --run results/runs/v11/v11__v11_blend --test input/test.jsonl \
        --out results/submissions/submission_v11_blend.csv \
        --proba results/submissions/submission_v11_blend_proba.csv

must equal the one of the original (`python -m src.cli submit --blend blend_v11`) trained on
`input/train.jsonl`, generated in step 0 (`results/v11_reference/`).

**Not the target:** `submission_blend_v11.csv` (root, and its copy in `tmp/`) is the
afternoon blend `0.55·logit(ensemble_from_lr on compact_v5) + 0.45·logit(v11)` (see
`SUBMISSIONS.md`); its labels differ from v11 alone on 339 of the 5,000 test battles (328 of
the 4,993 non-NaN rows of the recovered `results/submissions/v11_test_proba.csv`), so a port
of `blend_v11` cannot reproduce it.

## Rules of the port (on top of `CLAUDE.md`)

1. **Existing files are not modified**, with exactly two exceptions: one import line in
   `features/sets.py` (step 3b) and one in `models/__init__.py` (step 4). Everything else is a
   new file. `REFACTORING_STEPS.md` (this file) is updated with the status of each step.
   Never edit `tmp/`, existing tests, `tools/golden.py`, `results/golden/`.
2. **Behaviour first**: copy the original's code and formulas byte for byte (same constants,
   same operation order, same iteration order, same seeds, `SEED = 42`). Only rename to the
   repository's style, add type hints and docstrings, split modules over ~400 lines.
   Do not "fix" anything of the original, even if it looks wrong (note it in the status).
3. **Drop only what `blend_v11` never runs**: `usage_bags` (`bt_`, `bc_`), `hidden_features`
   (`bf_`, `bg_`, `u_`), the `cut` column as an output, `Monotone`, `SeedBag`, `et`, `rf`,
   `lr`, the training-set augmentation, `TeamEffect(joint=…, split=…)`, `cv.py`, `tune.py`,
   `ensemble.fit_weights` / `nested_accuracy`, the CLI. Keep everything that reaches the
   submission. `TeamModel` stays (used by `condition` and `simulation`).
4. **Thread count**: the original reads `FDS_N_JOBS` (default 4). The port uses the constant
   `N_JOBS = 4` in the detector, the forecast and the tree models; the reference is generated
   without `FDS_N_JOBS`, so with 4 threads.
5. **Layers** (`tests/test_architecture.py`): the features never import `models`, the models
   never import `domain` / `features`. The features may import `domain.records.fingerprint`,
   `lightgbm`, `sklearn`, `pandas`.
6. Data: the original reads `type_chart_gen2.csv` from its data folder; the port reads
   `assets/type_chart_gen2.csv` (`features.common.TYPE_CHART_PATH`) with the original's own
   semantics (product over the defender's types, `notype` skipped, missing pair = 1.0). Do not
   use `domain.typechart.TypeChart` (it rejects unknown types) nor `domain.damage` (other
   formulas) nor `domain.pokedex` (other conflict rules): v11 has its own Pokédex (first record
   wins, `for_battle` adds the battle's own species), fitted on the training battles only.
7. Checks against the original run in scratch scripts under `results/v11_reference/checks/`
   (git-ignored, not committed). Committed tests read only `tests/fixtures/` (CLAUDE.md).
8. Quality gates: `tmp/` is untracked and not ours, so run
   `ruff check . --extend-exclude tmp && ruff format --check . --extend-exclude tmp` and
   `pyright addons domain features models tests tuning tools *.py` (pyright's `include = ["."]`
   would scan `tmp/`). New modules must be clean; `pytest -q` must pass.

## Design

```
features/core/v11/            # the original's src/ (features side), one module per file
  mechanics.py   chart, effectiveness, Pokedex, stat / damage formulas, move classes
  replay.py      Side, Battle (timeline parsing, events, Spikes, transitions), parse
  faint.py       the hidden-faint detector (fit_detector)
  static.py state.py active.py events.py matchup.py sleep.py bags.py team.py
  team_model.py simulation.py (first version, MoveTable) simulation2.py (simulation_features)
  switching.py condition.py forecast.py (fit_forecast / predict_forecast)
  build.py       battle_features, the training / new-battle matrices
  feature.py     V11Features (the Feature of the set)
features/v11.py               # @feature_set("v11", requires=("lightgbm",))
models/core/v11_stacking.py   # spline_lr, PrefixScale, StackedLogit
models/core/v11_team.py       # TeamEffect (link, pair, soft, raw)
models/core/v11_member.py     # V11Member (one pipeline) and V11Blend (fixed logit weights)
models/v11.py                 # @model v11_lr_spline, v11_xgb, v11_lgbm, v11_cat, v11_blend
```

**`V11Features`** (one `Feature[BattleContext]`, `requires_fit = True`, `uses_labels = True`,
`inner_oof = False`). It never reads the labels: `uses_labels` only keeps
`FeatureSet.fit` from passing it unlabeled (test) battles, because v11 fits on the training
battles alone; `inner_oof = False` because its training rows are already cross-fitted by v11
itself. It reads `context.record` (the raw record without `player_won`), so the repository's
knockout repair (`--repair`) does not change its values: v11 uses its own detector (REFACTORING
rule 4 does not apply to it; the old bridge did the same).

- `fit(battles, labels)`: from `[c.record for c in battles]` (in this order) run exactly the
  original `build()` with these records as train and no test: Pokédex, `parse`, `fit_detector`
  (cross-fitted P2 probabilities), `Vocab`, `TeamModel`, `MoveTable`, `SwitchRates`, the
  training matrix, `truncated_features` at `forecast.CUTS`, the forecast (out of fold on the
  training rows). Keep: the training rows (the output columns only) keyed by
  `(battle_id, fingerprint(record))`, the Pokédex, the full detector model (fitted on all P1
  rows, as `fit_detector(train, test)` does for test), the helpers, the 6 forecast regressors
  fitted on all truncated rows, `cut` (= length of the first training timeline), the columns.
- `prepare(battles)`: for the battles whose key is not a training key and not cached, compute
  their rows in one batch exactly as the original treats `test` (detector: full model on P2
  candidates, recorded faints = 1; features; forecast with the full regressors; columns
  reindexed to the training columns).
- `compute(battle)`: the training row for a training key, else the cached row (calling
  `prepare([battle])` if missing). Why the key: a training battle must get its cross-fitted
  values, a new one the full-model values; train duplicates have distinct ids, and on the
  competition files no test battle shares `(battle_id, fingerprint)` with a training one.
- `output_names`: the columns after `fit` (None before). Order: the original matrix order
  (the order of `battle_features`, then `q_*`) restricted to the groups of the four pipelines:
  `static state traj active events matchup wincon sleep bag_species bag_hp bag_moves sim
  behaviour sim2 context forecast` plus `team` (`k_*`). Column order matters (tree models
  sample columns by position).
- The state (`get_state`) holds the training rows: `predict` of a saved run needs no
  training battle.

**Models.** `V11Member(kind, params)` builds at fit time, from the columns it receives, the
original `pipelines.estimator(spec, cols)`: `stack_full` (`StackedLogit` on the non-`k_`
columns, `C = 0.003`, `keep=True`) and the bag stack (`bs_ bh_ bm_` columns, `C = 0.01`) for
the tree members; `team_base` (spline `StackedLogit`, `base_C`, `base_spline`) for
`lr_spline`; then `TeamEffect(a=1.0, p2=True, link=True, pair=True, soft=True, raw=True)`;
then the estimator of `models.make`. Each step is added only when its columns exist (bag
stack: bag columns; team steps: `k_p1team`; spline keys: the present ones), so the model
contract test, which fits every registered model on four random columns, also passes; on the
`v11` table this is exactly the original. `V11Blend` fits the four members and returns
`expit(Σ w_i · logit(clip(p_i, 1e-6, 1 - 1e-6)))` with `np.tensordot`, the weights of
`blends.json` `blend_v11` as constants in `models/v11.py`. The parameters of the four
pipelines are constants copied from `configs/pipelines.json` (no file read at run time).

## Reference (`results/v11_reference/`, git-ignored)

Written in step 0 by `results/v11_reference/make_reference.py`, which imports the original
(`results/v11_reference/repo/`, a copy of `tmp/.../src` and `configs`) and repeats `build()`
and `cli.predict_test` step by step, saving:

| file | content |
|---|---|
| `p2_faint_train.pkl`, `p2_faint_test.pkl` | `{battle_id: {row: p}}` after `fit_detector(train, test)` |
| `features.pkl` | `X_train`, `X_test` (all columns, `cut` and `q_*` included), `y`, `groups`, `info` |
| `members_test.csv` | battle_id + test probability of each of the four members |
| `blend_test_proba.csv` | battle_id, `p_player_won` of `blend_v11` |
| `submission_blend_v11.csv` | battle_id, `player_won` (`p > 0.5`), test file order |
| `run_<env>_<k>/` | the same files of each run (env = `ours` / `orig` venv) |

---

## Step 0 – Reference from the original (model: small; ~18 min per run)

**Do**
- Copy `tmp/FDS-Pokemon-Battle-Prediction-main/{src,configs}` to `results/v11_reference/repo/`;
  data folder `results/v11_reference/data/` with links to `input/train.jsonl`,
  `input/test.jsonl`, `assets/type_chart_gen2.csv`.
- `make_reference.py <out dir>` (env `FDS_DATA_DIR=results/v11_reference/data`, no
  `FDS_N_JOBS`): the steps of `build()` (keeping the battles to dump `p2_faint`) and of
  `predict_test(members)` + `ensemble.blend` for `blend_v11`.
- Run it twice with this repository's venv (`.env`) and once with the original's venv
  (`tmp/.../.env`: numpy 2.4.6, scipy 1.17.1, xgboost 3.2.0 instead of 2.5.4 / 1.18.1 / 3.4.1).
  The two `.env` runs must be identical; copy one of them to the top of
  `results/v11_reference/` (the reference). Report the differences of the `orig` run (they
  only show what the library versions change; the port is compared with `.env`).

**Acceptance** – the two `.env` runs identical (all files); report the label flips and max
probability difference of the `orig` run and of the old `results/submissions/v11_test_proba.csv`
(v11 recovered from the afternoon blend) against the reference. Nothing to commit except this
file's status.

## Step 1 – Mechanics and replay (model: small)

**Do** – `features/core/v11/mechanics.py` (`data.py`'s `type_chart` / `effectiveness` /
`ALL_TYPES` / `STATS` / `Pokedex` + `mechanics.py`) and `replay.py`. Tests in
`tests/features/test_v11_replay.py` on the fixtures: hand-checked `stat_value`,
`damage_fraction` (STAB, immunity, Seismic Toss, Explosion, burn, Thick Club), events of a
repeated move, Spikes laid and spun, a P1 faint skipping the `fnt` row in `transitions`.

**Acceptance** – scratch check: for every train and test battle, `Battle` of the port and of
the original (same Pokédex built from train) give identical `sides` (names, hp, status,
effects, boosts, moves, events, revealed, first / last index), `spikes` and
`transitions("p1")`, `transitions("p2")` (same keys, same values, NaN equal). pytest, ruff,
pyright on the new files.

## Step 2 – Hidden-faint detector (model: small)

**Do** – `features/core/v11/faint.py`: `fit_detector(train, test)` as the original
(`N_JOBS = 4`). Test on the fixtures: a recorded faint gets 1, candidates get a probability in
[0, 1], the same call twice gives the same values.

**Acceptance** – scratch check: with the battles of step 1, `p2_faint` of every train battle
(cross-fitted) and test battle equals `p2_faint_train.pkl` / `p2_faint_test.pkl` exactly
(report the number of entries compared and the max difference, which must be 0).

## Step 3a – Feature groups (model: strong)

**Do** – `static`, `state` (`p1_track`, `p2_track`), `active`, `events`, `matchup`
(+ `wincon`), `sleep` (only `sleep_features`), `bags` (`Vocab`, `bag_features`), `team`
(`team_id`, `team_keys`), `team_model` (`TeamModel`), `simulation`, `switching`, `condition`,
and `build.battle_features` without `usage_bags` / `hidden_features`. Tests on the fixtures:
every group returns the same keys for every battle, `team_id` is stable, `TeamModel`
probabilities sum to the hidden slots.

**Acceptance** – scratch check: with the detector of step 2 and the helpers fitted on train,
`battle_features` of every train and test battle equals the reference `X_train` / `X_test` on
every column it produces (bitwise, NaN equal; `cut` and `q_*` excluded).

## Step 3b – Forecast, `V11Features`, feature set `v11` (model: strong)

**Do** – `forecast.py` (`CUTS`, `TARGETS`, `inputs`, `paths`, `forecast_features`, split into
"fit" and "predict" so that `prepare` can reuse the fitted regressors with identical values),
`build.py` (training and new-battle matrices, `truncated_features`), `feature.py`
(`V11Features`, see Design), `features/v11.py` (`@feature_set("v11", requires=("lightgbm",))`,
refuses a turn like `hp30`), and the import line in `features/sets.py`. Tests: the feature
contract test now covers `v11` automatically (keep it passing, it fits v11 on the 200
fixtures); a test that a saved state predicts a new battle like the fitted feature.

**Acceptance**
- scratch check: `build("v11")` fitted on the train contexts (`Dataset.from_jsonl`, any
  repair) — `fit_transform` on train and `to_frame` on test — equals the reference `X_train` /
  `X_test` restricted to the output columns, same column order, bitwise (report rows ×
  columns compared, the fit time).
- golden check passes (the import line must not move anything); pytest; ruff; pyright.

## Step 4 – Models (model: strong)

**Do** – `models/core/v11_stacking.py`, `models/core/v11_team.py`,
`models/core/v11_member.py`, `models/v11.py` (`v11_lr_spline`, `v11_xgb`, `v11_lgbm`,
`v11_cat` with `requires` of their library, `v11_blend` with all three, as `@model`, not
`@ensemble`: `build_tuned` would turn an ensemble into a probability average), and the import
line in `models/__init__.py`. Tests: `TeamEffect` on a small hand-made table (P1 / P2 team
keys, soft matching of a 5-species team, raw ratings), `StackedLogit.fit_transform` is out of
fold; the model contract test covers the new models automatically.

**Acceptance**
- scratch check: each member, fitted on the reference `X_train` restricted to the `v11`
  output columns, predicts on `X_test` the probabilities of `members_test.csv` exactly (max
  difference 0; report it per member and the fit times).
- golden check passes; pytest; ruff; pyright.

## Step 5 – `v11_blend` end to end (model: small; ~1 h)

**Do** – run the two commands of "What is ported" (pokedex and repair at their defaults).

**Acceptance** – `diff results/submissions/submission_v11_blend.csv
results/v11_reference/submission_blend_v11.csv` is empty; the probabilities equal
`blend_test_proba.csv` (max difference reported, expected 0, at most 1e-12). Report the run
times and the size of `results/runs/v11/v11__v11_blend/`. A second `predict` from the saved run
gives the same files.

## Step 6 – Quality gates and report (model: small)

**Do** – full `pytest -q`, ruff, pyright (rule 8), golden check. Add a short "v11" section at
the end of this file: commands, what is exact, the known differences with the original's
protocol (its own CV uses features fitted once on all of train; `main.py evaluate` refits v11
inside every fold, about 30 min per fold), the deviations from `REFACTORING.md` (rule 4: v11
has its own detector inside the feature; `uses_labels` declared without reading labels).

**Acceptance** – all green; golden check passes.

---

## Status

| step | status |
|---|---|
| 0 | done (3c17ecd): the two `.env` runs are identical (every file; `features.pkl` differs only in `info["seconds"]`), ~18 min per run, detector AUC 0.98307; `orig` venv vs reference: 50 / 5,000 label flips, max prob. diff 0.0439 (same `p2_faint`, features differ only in `t_hd_slope10` and the `q_*_h10` forecasts; members up to 0.10); old `results/submissions/v11_test_proba.csv` (4,993 non-NaN rows): 107 flips, max diff 0.0977 |
| 1 | done (b6cbee1): port and original give identical `Battle`s (sides, spikes, `transitions` of p1 and p2, NaN equal) on 15,000 battles (10,000 train + 5,000 test), 306,475 transitions, Pokédex of 239 species; 11 tests |
| 2 | done (2176d68): `p2_faint` equals the reference exactly: train 10,000 battles / 101,294 entries (cross-fitted), test 5,000 battles / 50,798 entries, max diff 0; detector AUC 0.98307, ~20 s |
| 3a | done (33fefb6): `battle_features` equals the reference bitwise (NaN equal) on 813 columns × 10,000 train + 5,000 test rows (all reference columns except the 528 `bt_ bc_ bf_ bg_ u_`, `cut` and the 6 `q_*`), ~60 s with the detector; 3 tests. `simulation.py` (332 lines in the original, ~470 formatted) is split into `simulation.py` (first version, `MoveTable`, `used_move`) and `simulation2.py` (`simulation_features`); `_proxies` / `_attack` became public `proxies` / `is_attack`; `simulate` / `simulate2` return early when a side is empty (the original breaks out of the loop at once: same result) so the indices are never `None` for pyright |
| 3b | done (e7c64e6): `build("v11")` (`fit_transform` on the 10,000 train contexts, `to_frame` on the 5,000 test contexts, `Dataset` with no repair) equals the reference `X_train` / `X_test` restricted to the output columns bitwise, same order: 10,000 + 5,000 rows × 819 columns (813 of `battle_features` + 6 `q_*`); fit ~9 min (533 s), test batch 20 s; golden check passes; 3 tests (the contract test covers `v11`). `faint.py` is unchanged: `build.py` imports its `_collect` / `_fit` for the full detector of new battles; the `q_*` columns are set by position instead of `join` (same result with unique battle ids); `features/v11.py` imports `V11Features` inside the builder so that listing the sets needs no LightGBM |
| 4 | done (this commit): each member fitted on the reference `X_train` (10,000 × 819 `v11` output columns) predicts on `X_test` exactly the probabilities of `members_test.csv`: max diff 0 for `v11_lr_spline` (fit + predict 16 s), `v11_xgb` (26 s), `v11_lgbm` (24 s), `v11_cat` (49 s); their blend equals `blend_test_proba.csv` (max diff 0); golden check passes; 6 tests (the contract test covers the five models). `TeamEffect` keeps only the options `blend_v11` uses (`joint`, `split`, `lam`, `lam_pair` dropped); `spline_lr` falls back to `SPLINE_KEYS` only for `keys=None` (the original also for an empty list) so that a table with none of the keys, like the contract test's, fits; `StackedLogit` / `TeamEffect` / `make` are otherwise copies; `SEED = 42` lives in `v11_stacking.py`, `N_JOBS = 4` in `v11_member.py` (models may not import the features' constants); the lr_spline base uses `BASE_C = 0.0007` and the model's keys (the configuration's `base_spline` is the same list) |
| 5 | todo |
| 6 | todo |
