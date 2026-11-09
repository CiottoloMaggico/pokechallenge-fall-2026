# PokeChallenge 2026 – instructions for coding agents

Predict the winner (`player_won`, P1 wins) of Gen 2 OU battles from the first
30 turns. Read this file before any change; when it and the code disagree, the
code's behaviour wins: note the conflict and ask.

## Environment
- Python 3.14, venv in `.env/` (`requirements.txt`): `source .env/bin/activate` before every
  command.
- Data: `input/train.jsonl`, `input/test.jsonl`, stored with Git LFS. Never modify or re-commit
  them.
- Outputs only under `results/` (git-ignored, paths in `storage.ResultsLayout`). The
  evaluations behind the tables of `README.md` are in `results/evaluations/`.
- Knockout detector: a versioned artifact in `results/addons/knockouts/<version>/`, loaded by
  every command (`--repair`, default `model=results/addons/knockouts/v3`). If `v3` is missing,
  rebuild it with `python main.py detector train --version v3 --seeds 42 1 2` (~2 min, same
  P2 repairs). Versions are never overwritten: a new detector gets a new version.
- `tmp/` (untracked, not ours) holds the original FDS repository that the v11 port copies.
  Never edit it.

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
  replay (`BattleReplay`, `BattleContext`), transitions. It defines `TimelineRepair` but never
  imports an implementation.
- `addons/`: the knockout repairs (`none`, `rule`, `model=<artifact>`); never import the
  application or the CLI.
- `features/`: one file per feature set (`features/<set>.py`, registered with `@feature_set`
  and imported by `features/sets.py`), feature families in `features/core/`, the detector's
  sets in `features/transitions/`. Never import `models`, `addons` or the application.
- `models/`: one file per model (`@model` / `@ensemble` in `registry.py`), shared pieces in
  `models/core/`; a model sees a `DataFrame` and labels only (no `domain`, no `features`).
- `interfaces.py` (`Feature`, `Model`, …), `feature_set.py` (`FeatureSet`), `registry.py`.
- `main.py` only parses arguments, builds `Settings` and calls one function of `usecases.py`.

## Commands
- Tests: `pytest` (no test reads `input/` or `results/`); lint:
  `ruff check . && ruff format --check .`; types: `pyright`. CI
  (`.github/workflows/ci.yml`) runs the three. Locally `tmp/` exists, so use
  `ruff check . --extend-exclude tmp && ruff format --check . --extend-exclude tmp` and
  `pyright addons domain features models tests tuning tools *.py`.
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
- v11 (native port of the FDS repository's `blend_v11`, ~13 min train, ~1.5 min predict):
  `python main.py train --features v11 --models v11_blend --out results/runs/v11`
  `python main.py predict --run results/runs/v11/v11__v11_blend --test input/test.jsonl --out results/submissions/submission_v11_blend.csv --proba results/submissions/submission_v11_blend_proba.csv`
  Its submission equals the original's (`results/v11_reference/submission_blend_v11.csv`,
  same probabilities). The scratch checks against the original are in
  `results/v11_reference/checks/`.
- No feature depends on the hash seed (contract test in
  `tests/features/test_feature_contract.py`).

## Rules that must never be broken
1. Only `train.jsonl` fits anything (Pokédex, detector, features, models);
   test battles are only predicted. The Pokédex comes only from records in the
   competition format, never from external data.
2. Duplicated battles (same fingerprint) stay in the same fold; every split is
   checked for leakage.
3. Label-aware features give out-of-fold values on their training rows
   (grouped inner folds) or NaN (`inner_oof = False`). Never in-sample values.
4. The knockout repair is applied to every battle before any feature. Addons
   (repairs) never see `player_won`. A run records its repair and predicts only
   with the same one. The one exception is `v11`: `V11Features` reads the raw record
   and runs v11's own hidden-faint detector, so `--repair` does not change its columns.
5. Do not change formulas, hyper-parameters, seeds, thresholds, column names or
   column order of existing feature sets or models unless the user explicitly asks.
   The v11 port (`features/core/v11/`, `models/core/v11_*.py`, `models/v11.py`) must stay an
   exact copy of the original: do not "fix" it.
6. No `fork` (multiprocessing, `--jobs`) after LightGBM / XGBoost / CatBoost
   have been used in the parent process: the workers hang at 0% CPU.
7. Every comparison of models or feature sets uses the corrected repeated
   k-fold t-test (`stats_tables.corrected_ttest`), never the naive one.

## How to work
- One change per commit, made only after its checks pass (golden check if the results must
  not move, `pytest`, ruff, pyright). Messages: `feat(<scope>): …`, `fix(<scope>): …`,
  `docs: …`, `refactor(<scope>): …`. A fix that changes results goes in its own `fix`
  commit, only after the user agrees.
- Add files by explicit path (deletions included); never commit `results/`, `tmp/` or
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
