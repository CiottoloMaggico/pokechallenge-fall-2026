# Refactoring – one prompt per step

Give the agent one step at a time, in a new session, with this preamble:

> Read `CLAUDE.md`, `REFACTORING.md` and this step of `REFACTORING_STEPS.md`.
> Do only this step. Before finishing run the checks listed under
> "Acceptance" and paste their output. If a golden check fails, stop and report.
> If they pass, commit as `CLAUDE.md` says (explicit paths, no push) and end the
> report with `git log --oneline -3` and `git status`.

"Model" says where a smaller model is enough (mechanical work with a tight
check) and where a stronger one is safer (design decisions across many files).

---

## Step 0 – Tooling and reference (model: small)

**Goal** – Prepare the safety net before touching code.

**Do**
- `pyproject.toml`: ruff (pep8, isort, pyupgrade, bugbear; line length 100),
  pyright (basic for now), pytest; no change to runtime dependencies.
- `tests/fixtures/battles.jsonl`: 200 battles of `input/train.jsonl` chosen to
  cover a P1 faint, a double KO, a voluntary switch, Roar / Whirlwind,
  Explosion, Spikes, a duplicated battle, a species without stats; a script
  `tests/fixtures/make_fixtures.py` that rebuilds the file from train.
- `python tools/golden.py export --out results/golden/reference` on the
  untouched code; `results/` git-ignored.

**Do not** – change any `.py` outside `tests/` and `tools/golden.py`.

**Acceptance** – `results/golden/reference/metrics.json` exists; `ruff check`
runs (errors are fine at this point, list their count); `pytest` collects 0+
tests without import errors.

---

## Step 1 – Domain values: records, errors, type chart, moves, damage (model: strong)

**Goal** – `domain/records.py`, `domain/errors.py`, `fingerprint()`,
`TypeChart`, `Move`, `Field`, `Combatant`, `DamageCalculator` as in
REFACTORING.md §3.1, §3.2, §3.4, wrapping the current formulas.

**Do**
- Move code, keep formulas byte-for-byte; old call sites call the new objects.
- One `fingerprint()` used by duplicate groups (today in `data.py`).
- Tests: damage table of hand-checked cases (STAB, immunity, crit, Reflect,
  Light Screen, sand, burn, paralysis Speed), record validation errors.

**Do not** – touch the Pokédex, the replay, the detector.

**Acceptance** – golden check passes; new tests pass; `pyright` strict passes
on the new modules.

---

## Step 2 – Pokédex from train only (model: strong) — planned result change

**Before (separate session, ~30 min, no code change)** – on the step 1 commit
run the full regression (3 seeds) and write the numbers and the run folder in
`CLAUDE.md` as "Baseline of step 1"; commit only `CLAUDE.md`
(`docs: regression baseline of step 1`). This keeps the logreg seed fix out of
the Pokédex comparison.

**Goal** – REFACTORING.md §3.3: `PokedexBuilder` from competition-format
records only, `for_battle(record)`, policies `Strict` and `Missing`; remove the
manual species file and the random replacement; `python -m domain build-pokedex`.

**Do**
- Count and report the species that lose their stats (they were only in the
  manual file) and the battles that contain them.
- Features that need stats of a missing species return NaN; the damage
  calculator returns `None` for them.

**Acceptance**
- Golden check: report the differences (expected in `species.csv` and in the
  rows of battles with those species); everything else must match.
- Full regression (3 seeds): report the numbers and the paired difference with
  the corrected t-test against the baseline run of step 1 (folder in
  `CLAUDE.md`, same seeds). Stop here, without committing: the user decides;
  then commit, re-export `results/golden/reference` (twice, checked) and
  update the expected numbers in `CLAUDE.md`.

---

## Step 3 – Replay, addon and repair ports, transitions (model: strong)

**Goal** – §3.5–§3.7: `BattleReplay` / `BattleContext` / `TurnState`,
`TimelineRepair` (with `describe()`), `NoRepair`,
`TransitionExtractor` (the switch description moved out of
`domain/knockouts.py`), `python -m domain replay | check`.

**Do**
- The current detector keeps working through an adapter implementing
  `TimelineRepair` (it moves for real in step 7).
- `domain` imports only stdlib and numpy; add `tests/test_architecture.py`
  for the domain rule now.
- Tests per transition case with the fixtures (P1 faint skips the `fnt` row,
  double KO carries no features, phazing, Explosion).

**Acceptance** – golden check passes; architecture test passes; pyright strict
on `domain/`.

---

## Step 4 – Interfaces, registry, artifacts, features (model: strong for 4a, small for 4b)

**4a** – `interfaces.py` (Feature with `prepare`, `inner_oof`; FeatureSet with
`without`), `registry.py` (`@feature_set`, `@model`, `@ensemble`, `@t<N>`
parsing), `artifacts.py` (`TrainedRun` save / load, storing the repair
manifest). Contract test parametrised over every registered feature. The
contract test also runs each feature in two subprocesses with different
`PYTHONHASHSEED` and requires identical output; fix the features that fail
(`TeamKnowledgeFeatures` and any other set / frozenset iteration) by sorting.
If a compact_v5 column changes, report it like the Pokédex change of step 2.

**4b** – Move every feature to `features/core/<family>.py` (field_features.py
included) and every feature set to its own file with its history in the
docstring (`compact`, `compact_v2` … `compact_v5`, `*_orig`); delete
`feature_catalog.py`. Pure moves: no renaming of columns or ids. 4a left the
catalog declaring its sets with `@feature_set`, so they move as they are. Deleting
the catalog breaks `tools/v11_common.py`, and `tools/blend_from_saved.py old-base`
if it is run on this code instead of the old worktree of `SUBMISSIONS.md`: ask the
user before deleting it.

**Acceptance** – golden check passes; contract tests pass for all features;
`python main.py evaluate --features compact_v5 --models logreg --seeds 33`
gives the logreg number of `metrics.json`.

---

## Step 5 – Models (model: small, with the contract test)

**Goal** – §5: `models/core/` (EstimatorModel, EnsembleModel,
BoostFromLogreg, LogitBlend, preprocessing), one file per model with its
`ModelSpec` (builder, search space, required modules); delete
`model_catalog.py`; `build`, `build_tuned`, `available` in the registry.
`ModelSpec`, `@model` and `@ensemble` exist since 4a, used by `model_catalog.py`.

**Do not** – change any default parameter or search space.

**Acceptance** – golden check passes; model contract test (fit / predict /
clone / get_params JSON / save-load) for every registered model whose
libraries are installed; `--params` files of `results/tuning/` still load.

---

## Step 6 – Gym, statistics, storage, settings, CLI (model: strong)

**Goal** – §6: `FoldPlan`, `Gym` on any rows / labels / groups
(evaluate, train, fold tables cached), `GymResult` with seeds and corrected
paired tests, `stats_tables.py`, `ResultsLayout` (everything under `results/`),
`Settings`, thin `main.py` with the same commands and options.

**Do**
- `--jobs` disabled automatically (with a message) when a selected model or
  feature uses LightGBM / XGBoost / CatBoost.
- Old `runs/` and `results/` folders: a one-off move script, not code paths
  (`tools/move_old_outputs.py`; not applied, the user keeps the old outputs in place).

**Acceptance** – golden check passes; full regression (3 seeds) matches the
reference numbers; `train` + `predict` produce the same submission as before
the step (`diff` of the CSVs is empty).

---

## Step 7 – Knockout detector as a task and a versioned addon (model: strong)

**Goal** – §3.8: `detector.py` (table of P1 switches) with its checks (symmetry estimate,
adversarial AUC P1 vs P2, agreement with a reference artifact),
`features/transitions/ko_raw50.py` and `ko_domain30.py`, model `hgb_ko`,
commands `detector evaluate | train | tune`, artifact
`results/addons/knockouts/<version>/` with `info.json`,
`addons/knockouts_model.py` (`ModelKnockoutRepair`) and
`addons/knockouts_rule.py` (deterministic), `--repair none | rule | model=<path>`.
Delete `knockout_detector.py` and the column lists of `domain/knockouts.py`.

**Do**
- Check whether the detector's input uses anything outside the timeline; if not,
  the timeline key is correct, document why; if yes, switch to fingerprint with
  a golden check, and report the 5 shared keys.

**Acceptance**
- `detector evaluate --features ko_domain30 --models hgb_ko --seeds 42 1 2`
  (GroupKFold on battles, as the old experiments) gives log loss ≈ 0.0988,
  AUC ≈ 0.9918; the symmetry probe reports ≈ 10,130 repaired P2 switches
  (all three: re-measure after step 2 (10,109 repairs)).
- `detector train --version v3`, then golden check with
  `--repair model=results/addons/knockouts/v3`: `repairs_p2.csv` and every
  other file identical to the reference.
- A run trained with one repair refuses to predict with another.

**Result (commit ca1a010)** – log loss 0.0992 (folds: the gym's `StratifiedGroupKFold`,
grouped by duplicate battle), AUC 0.9918, 10,109 P2 repairs, symmetry estimate 10,135,
adversarial AUC 0.586; golden check passed with `v3`. The detector reads the team details
(species stats), so its cache key is now `fingerprint` (5 shared timeline keys, no golden
change). Not done: `--refit-detector-per-fold` (REFACTORING.md §3.8).

---

## Step 8 – Tuning, selection, cleanup (model: small)

**Goal** – `tuning/tuner.py`, `tuning/params.py`, `selection.py`
(`backward` with corrected one-sided p, `forward`) on the generic gym; delete
`exp/`, `tools/` except `tools/golden.py`, `selection.log`, old scripts; the v11
bridge is not carried over. The v11 bridge and `tools/blend_from_saved.py` are
needed to reproduce `submission_blend_step2.csv`, so deleting them is the user's
decision.

**Acceptance** – golden check passes; `tune --trials 3` and
`select --repeats 1` run end to end on the fixtures.

**Result (commit b5fa791)** – golden check passed; `tune --trials 3` and
`select --repeats 1` (forward and backward) ran on the fixtures. `tuning/params.py` has
two functions instead of the `ParamsFile` class; `select` is a command of `main.py`
(seeds 11, 22, 33 … as the old script, report in `results/selections/`); `forward` no
longer drops the last column (it crashed on the fixtures). With the user's agreement the
v11 bridge (`features/v11_bridge.py`, `compact_v5_v11`, `blend_v11`), `tools/v11_*`,
`tools/blend_from_saved.py` and `tools/move_old_outputs.py` were deleted: they are at the
tag `submission-final` (and `c62b744` for the last one); `LogitBlend` stays (§10). There
was no `exp/`.

---

## Step 9 – Quality gates (model: small)

**Goal** – §8: ruff clean, pyright (strict where listed) clean, architecture
test for all rules of §1, end-to-end test on the fixtures (evaluate → train →
predict, under a minute), CI workflow running ruff, pyright, pytest.

**Acceptance** – all green locally and in CI; final full regression matches
the reference; `detector evaluate` matches step 7.

**Result (commit 60707d3)** – ruff, ruff format, pyright: clean (`tools/golden.py` and
the markdown files excluded); `pytest`: 651 passed, 43 skipped, also in a copy without
`input/` and `results/`; end-to-end test about 10 s. Final regression
(`results/reference_runs/step9_final`) identical to `step2_pokedex` in all 30 folds and
60,000 predictions; detector: log loss 0.0992, AUC 0.9918, 10,109 P2 repairs, symmetry
estimate 10,135. The CI workflow was not run on GitHub (nothing pushed).
