# PokeChallenge 2026 – instructions for coding agents

Predict the winner (`player_won`, P1 wins) of Gen 2 OU battles from the first
30 turns. Read this file before any change; when it and the code disagree, the
code's behaviour wins: note the conflict and ask.

## Environment
- Python 3.14, venv in `.env/` (`requirements.txt`): `source .env/bin/activate` before every
  command.
- Data: `input/train.jsonl`, `input/test.jsonl`, stored with Git LFS. Never modify or re-commit
  them.
- Outputs only under `results/` (git-ignored, never committed; paths in
  `storage.ResultsLayout`). The evaluations behind the tables of `README.md` are in
  `results/evaluations/`. Two kinds of result are tracked, as copies: the tuned
  hyper-parameters in `assets/params/<set>__<model>.json` (copied from `results/tuning/` once a
  search is adopted) and the small tables and figures of the notebook in
  `docs/evaluation/data/` and `docs/evaluation/figures/` (rebuilt with the scripts of
  `docs/evaluation/`).
- Knockout detector: a versioned artifact in `results/addons/knockouts/<version>/`, loaded by
  every command (`--repair`, default `model=results/addons/knockouts/v3`). If `v3` is missing,
  rebuild it with `python main.py detector train --version v3 --seeds 42 1 2` (~2 min, same
  P2 repairs). Versions are never overwritten: a new detector gets a new version.
- v11 hidden-faint detector: a versioned artifact in `results/addons/knockouts_v11/<version>/`,
  built with `python main.py detector train-v11 --version v1 --seeds 42 1 2` (~6 min; the
  repository's standard `FoldPlan`, as `v3`) and loaded with
  `--repair v11=results/addons/knockouts_v11/v1` by every v11 command. Never overwritten.
- The original FDS repository of v11 is outside this repository, in
  `~/Downloads/FDS-Pokemon-Battle-Prediction/FDS-Pokemon-Battle-Prediction`. Never edit it.
  `results/v11_ref/sub20_blend_v11.csv` is the submission it produced with `blend_v11`;
  `results/v11_alignment/dump/` holds its outputs recomputed in this `.env` (detector
  probabilities, feature matrices, member / blend probabilities, out-of-fold predictions of its
  own protocol). v11's code must match them bit for bit when fed the original's detector
  probabilities (`tools/v11_check.py`); with v11's own repair (standard folds for the detector)
  the numbers are within noise of them. Never delete it.
- **The faithful v11 is frozen** (a historical reference): `features/v11.py`,
  `features/core/v11/*`, `models/v11.py`, `models/core/v11_*.py`, `models/ko_v11.py`,
  `addons/knockouts_v11.py`, `tools/v11_check.py` are never modified again, not even to move
  code. Every future change to v11 is made on the living line: the feature set `v11_fs`
  (`features/v11_fs.py`, its own package `features/core/v11_fs/`, which never imports
  `features/core/v11/`) read by the standard models with tuned parameters
  (`--params assets/params/v11_fs__<model>.json`; blend: `ensemble_boost`).

## Architecture
Layers (enforced on the imports by `tests/test_architecture.py`):

```
main.py (CLI) ──► application: usecases, gym, folds, detector, tuning/, selection,
                               stats_tables, storage, settings, artifacts
                        │
          ┌─────────────┼──────────────┐
          ▼             ▼              ▼
       addons ──►   features        models      (addons may use both)
          │             │       (no domain import)
          ▼             ▼
                 domain  (pure logic + TimelineRepair protocol)
```

- `domain/` imports only the standard library and numpy: records, Pokédex, type chart, damage,
  replay (`BattleReplay`, `BattleContext`), transitions, observed timeline (`timeline.py`),
  expected-damage estimate (`expected_damage.py`), switch descriptions (`switches.py`). It
  defines `TimelineRepair` but never imports an implementation. The domain is agnostic: no
  model, feature set or version appears in it (modules are named by concept); new needs are
  met by new generic modules, not by changing the logic of existing ones.
- `addons/`: the knockout repairs (`none`, `rule`, `model=<artifact>`, `v11=<artifact>`);
  never import the application or the CLI. An artifact is built by `detector.py`
  (`detector train`, `detector train-v11`) under `results/addons/<kind>/<version>/`.
- `features/`: one file per feature set (`features/<set>.py`, registered with `@feature_set`
  and imported by `features/sets.py`), feature families in `features/core/`, the detector's
  sets in `features/transitions/`. v11: one `Feature` per group in `features/core/v11/`
  (frozen) and in `features/core/v11_fs/` (the living copy), computed from `BattleContext`
  (record, the repository's Pokédex, the repair's probabilities); they work with any repair.
  Never import `models`, `addons` or the application.
- `staged_feature_set.py` (features layer): `StagedFeatureSet`, a set in two stages: a
  `FeatureSet` of battle features, then `TableFeature`s that read its table (and may consume
  some of its columns). Use it for transformations that learn on columns already computed (a
  regression of a block of columns, a team effect against a base logit): the first
  stage runs once and only the cheap learned part is refitted in the inner folds (same grouped
  inner folds and leakage rules as `FeatureSet`). `v11_fs` is one: the v11 groups, then
  `V11BagLogit` (`stack_logit_0`) and `V11TeamEffect` (`stack_logit_full`, `tm*`), both
  label-aware: the table of the original's boosters.
- Groups without fitted state that several features compute on the same battle are remembered
  on the context (`context.cached((<set>, <group>), ...)`, e.g. `features/core/v11_fs/
  stateless.py`); never remember anything that depends on fitted state.
- `models/`: one file per model (`@model` / `@ensemble` in `registry.py`), shared pieces in
  `models/core/`; a model sees a `DataFrame` and labels only (no `domain`, no `features`).
  A model is never tied to one feature set: steps that need particular columns (the v11
  members' team effect and bag logit) take the column roles as parameters and are skipped when
  the table has no such columns.
- `interfaces.py` (`Feature`, `Model`, …), `feature_set.py` (`FeatureSet`),
  `staged_feature_set.py` (`StagedFeatureSet`, `TableFeature`), `registry.py`.
- `main.py` only parses arguments, builds `Settings` and calls one function of `usecases.py`.
- `docs/notebook/pokechallenge.ipynb` (the Kaggle notebook of the three submissions) and
  `docs/evaluation/` (its scripts, tables and figures; application-level
  scripts like `tools/`). `docs/evaluation/plots.py` imports only pandas, numpy and matplotlib:
  the notebook's kernel reads tables and draws, everything else runs through `main.py` and the
  scripts in the pinned environment.

## Commands
- Tests: `pytest` (no test reads `input/` or `results/`); lint:
  `ruff check . && ruff format --check .`; types: `pyright`. CI
  (`.github/workflows/ci.yml`) runs the three.
- Golden check (any change that must not move the results, ~10 min; needs the `v3`
  artifact, `export --repair` picks another):
  `python tools/golden.py export --out results/golden/candidate`
  `python tools/golden.py check results/golden/reference results/golden/candidate`
- Detector check (~2 min):
  `python main.py detector evaluate --features ko_domain30 --models hgb_ko --seeds 42 1 2`
  Expected: log loss 0.0992, AUC 0.9918, 10,109 P2 repairs, symmetry estimate 10,135.
- Tuning and selection: `python main.py tune --features <set> --model <m> --trials N`
  (seed 2027, results in `results/tuning/`, used with `--params`);
  `python main.py select --method forward|backward --start <set> [--pool <set>]`
  (corrected t-test, report in `results/selections/`).
- Full regression (~30 min):
  `python main.py evaluate --features compact_v5 --models logreg ensemble_from_lr --seeds 33 66 133 --out results/evaluations/<name>`
  Reference: the `compact_v5` / `logreg` and `ensemble_from_lr` rows of
  `results/evaluations/models_compact_v5/folds.csv` (same seeds; ensemble_from_lr
  0.7207 / 0.5402, logreg 0.7191 / 0.5426, accuracy / log loss); a change that must not
  move the results matches them fold by fold.
- v11 (the FDS repository's `blend_v11` on this repository's constructs, ~15 min train,
  ~1 min predict; build the detector artifact first):
  `python main.py detector train-v11 --version v1 --seeds 42 1 2`
  `python main.py train --features v11 --models v11_blend --repair v11=results/addons/knockouts_v11/v1 --out results/runs/v11`
  `python main.py predict --run results/runs/v11/v11__v11_blend --test input/test.jsonl --repair v11=results/addons/knockouts_v11/v1 --out results/submissions/submission_v11_blend.csv --proba results/submissions/submission_v11_blend_proba.csv`
  Its probabilities are within noise of `results/v11_alignment/dump/original/blend_v11.npy`
  (the detector's training values use the repository's folds); the original run in this
  `.env` differs by 97 labels from `results/v11_ref/sub20_blend_v11.csv` (library versions).
  `train` / `evaluate` refuse `--test` with v11 (its label-free features fit on the training
  battles only).
- v11_fs, the living line (~11 min per fold of features on 8,000 battles, about 60 min per seed
  of `evaluate`), with `R=v11=results/addons/knockouts_v11/v1` and `P` = the four files
  `assets/params/v11_fs__{logreg,xgboost,lightgbm,catboost}.json`:
  `python main.py tune --features v11_fs --model <model> --trials N --repair $R`
  `python main.py evaluate --features v11_fs --models logreg xgboost lightgbm catboost ensemble_boost --params $P --seeds <seed> --repair $R --out results/evaluations/<name>_s<seed>`
  `python main.py train --features v11_fs --models ensemble_boost --params $P --repair $R --out results/runs/tuned`
  `python main.py predict --run results/runs/tuned/v11_fs__ensemble_boost-tuned --test input/test.jsonl --repair $R --out results/submissions/submission_v11_fs_ensemble_boost.csv`
  Results: `results/evaluations/tuned_v11_fs_s33` (README, "v11"; one seed), the blend within
  noise of `v11_blend`. `train` / `evaluate` refuse `--test`. The fold tables of `tune`
  (`results/cache/folds`) are keyed by the set's name, not by the repair or the code: pass
  `--refresh` after changing either.
  Long evaluations: one process per seed (at most two at a time; `--jobs` is ignored with the
  boosters, rule 6), each with its own `--out`, then merged with `GymResult.load` +
  `GymResult.combine` (and `export`) before the comparisons.
- v11 regression check (~50 min, of which ~35 for the `cv` stage; `--stages` runs a subset;
  needs the v11 artifact, `results/v11_alignment/dump/` and, once,
  `python tools/v11_check.py export`): `python tools/v11_check.py check` must pass: max |diff| 0 at every bitwise stage (domain,
  full detector, features, models and the original's CV protocol, fed the original's detector
  probabilities); the noise
  stages (detector training values, end to end) are reported, not failed. Run it on every
  change to the domain modules v11 uses (`timeline.py`, `expected_damage.py`, `switches.py`)
  or to v11's addon, features or models; if it fails, stop and report as for the golden check.
- No feature depends on the hash seed (contract test in
  `tests/features/test_feature_contract.py`).

## Rules that must never be broken
1. Only `train.jsonl` fits anything (Pokédex, detector, features, models);
   test battles are only predicted. The Pokédex comes only from records in the
   competition format, never from external data.
2. Duplicated battles (same fingerprint) stay in the same fold; every split is
   checked for leakage. Declared exception, for fidelity to the original: the faithful v11's
   own inner cross-validations (`StackedLogit`, `TeamEffect`, forecast) are not grouped by
   duplicates, as in the original. (v11's detector artifact uses the standard grouped
   `FoldPlan`.) In `v11_fs` the training rows of the label-aware features (bag logit, team
   effect) are out of fold on the grouped inner folds, so the rule holds for them; the
   cross-fits kept from v11 by decision stay ungrouped: the base logit inside `V11TeamEffect`
   and the forecast's `KFold`.
3. Label-aware features give out-of-fold values on their training rows
   (grouped inner folds) or NaN (`inner_oof = False`). Never in-sample values.
4. The knockout repair is applied to every battle before any feature. Addons
   (repairs) never see `player_won`. A run records its repair and predicts only
   with the same one. One repair at a time; repairs are independent of the models, so any
   feature set can run with any repair. The faithful v11 uses the v11 repair
   (`--repair v11=<artifact>`); v11 with `v3` is a variant, compared with the corrected
   t-test. Like `v3`, a detector artifact (and the Pokédex) is fitted once on the training
   file, not inside each fold of `evaluate` (accepted: it never reads `player_won`).
5. Do not change formulas, hyper-parameters, seeds, thresholds, column names or
   column order of existing feature sets or models unless the user explicitly asks.
   v11 (`features/core/v11/`, `addons/knockouts_v11.py`, `models/ko_v11.py`,
   `models/core/v11_*.py`, `models/v11.py`) keeps the logic of the original's `blend_v11`
   exactly (formulas, constants, operation and iteration order, seeds, hyper-parameters, blend
   weights): do not "fix" it. Its implementation uses the repository's constructs (domain
   Pokédex, type chart and the generic timeline / expected-damage / switch modules, detector
   as a repair addon built with the gym, one `Feature` per group, `EstimatorModel` members),
   and fed the original's detector probabilities its outputs must stay bitwise identical to
   `results/v11_alignment/dump/` (`tools/v11_check.py`). The one accepted change of logic: the
   detector's cross-fitted training values come from the standard `FoldPlan` (mean over
   `--seeds`) instead of the original's `KFold` over battles (measured: within noise).
6. No `fork` (multiprocessing, `--jobs`) after LightGBM / XGBoost / CatBoost
   have been used in the parent process: the workers hang at 0% CPU.
7. Every comparison of models or feature sets uses the corrected repeated
   k-fold t-test (`stats_tables.corrected_ttest`), never the naive one.

## How to work
- One change per commit, made only after its checks pass (golden check if the results must
  not move, `pytest`, ruff, pyright). Messages: `feat(<scope>): …`, `fix(<scope>): …`,
  `docs: …`, `refactor(<scope>): …`. A fix that changes results goes in its own `fix`
  commit, only after the user agrees.
- Add files by explicit path (deletions included); never commit `results/` or
  changes to `input/`.
- Git: by default never push and never rewrite history (no amend, rebase, reset, filter of
  commits). Both are allowed only when the user explicitly asks for them; before rewriting,
  save a backup (e.g. `git bundle create … --all`) and check that only the requested parts
  changed.
- To run older code, use `git worktree add <scratch dir> <commit>` instead of `git stash`.
- The golden reference (`results/golden/reference`) is re-exported only when the user asks:
  export it twice in two processes and check that the two copies match before using it.
  `tools/golden.py` may only change in `_export` (imports / calls of moved code); its outputs
  and format must stay identical.
- If the golden check fails: stop, do not edit tests or golden files to make it pass, and
  report which file / column / rows differ and the likely cause.
- Prefer moving code over rewriting it; keep behaviour first, then clean up.
- Determinism: every model has a fixed seed; a new model or tuned-parameter file must not
  change that. Features sort before iterating a set / frozenset: the contract test requires
  the same output under two hash seeds, and a new feature must be added to it.

## Simplicity
- Write the simplest code that does the job; a human must be able to read any file without
  the agent.
- No abstraction without two concrete uses today (no protocol, base class or
  registry for a single implementation, no "future" hooks).
- Functions under ~40 lines, modules under ~400 lines; split by responsibility.
- No defensive checks for impossible states; validate input once at the edges
  (record loading, CLI).
- Docstrings say why and the contract, not a line-by-line paraphrase.
- Delete what is replaced in the same change: no dead code, no compatibility shims, no
  commented-out code.
- If a request looks over-engineered for its use, say so and propose the simpler version.

## Code style
- PEP 8 via ruff (line length 100), imports sorted, type hints everywhere;
  pyright strict on `domain/`, `interfaces.py`, `stats_tables.py`, basic elsewhere.
- English comments, only on lines whose reason is not obvious; no separator comments in new
  code; a docstring on every public class and function.
- Tests next to each module family under `tests/`, small fixtures in
  `tests/fixtures/` (no full data in tests).
