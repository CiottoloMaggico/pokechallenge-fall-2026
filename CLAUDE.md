# PokeChallenge 2026 – instructions for coding agents

Predict the winner (`player_won`, P1 wins) of Gen 2 OU battles from the first
30 turns. The repository is being refactored following `REFACTORING.md`;
`REFACTORING_STEPS.md` has one prompt per step. Read both before any change.

## Environment
- Python venv in `.env/`: `source .env/bin/activate` before every command.
- Data: `input/train.jsonl`, `input/test.jsonl` (never commit them).
- Outputs only under `results/` (after the refactoring; `runs/` today).

## Commands
Since step 4a no feature depends on the hash seed (contract test in
`tests/features/test_feature_contract.py`). `export PYTHONHASHSEED=0` is needed
only to run commits older than step 4a, such as the recipes in `SUBMISSIONS.md`.

- Tests: `pytest -q` (once `tests/` exists); lint: `ruff check . && ruff format --check .`;
  types: `pyright`.
- Golden check (every step, a few minutes):
  `python tools/golden.py export --out results/golden/candidate`
  `python tools/golden.py check results/golden/reference results/golden/candidate`
- Full regression (only where a step says so, ~30 min):
  `python main.py evaluate --features compact_v5 --models logreg ensemble_from_lr --seeds 33 66 133 --out results/reference_runs/<name>`
  Baseline of step 2 (commit ad7e4c4, run folder:
  `results/reference_runs/step2_pokedex`): ensemble_from_lr 0.7207 / 0.5402,
  logreg 0.7191 / 0.5426 (accuracy / log loss). Both must match that run fold by
  fold.

## Rules that must never be broken
1. Only `train.jsonl` fits anything (Pokédex, detector, features, models);
   test battles are only predicted. The Pokédex comes only from records in the
   competition format, never from external data.
2. Duplicated battles (same fingerprint) stay in the same fold; every split is
   checked for leakage.
3. Label-aware features give out-of-fold values on their training rows
   (grouped inner folds) or NaN (`inner_oof = False`). Never in-sample values.
4. The knockout repair is applied to every battle before any feature. Addons
   (repairs) never see `player_won`.
5. Do not change formulas, hyper-parameters, seeds, thresholds, column names or
   column order of existing feature sets unless the step explicitly asks it.
6. No `fork` (multiprocessing, `--jobs`) after LightGBM / XGBoost / CatBoost
   have been used in the parent process: the workers hang at 0% CPU.
7. Every comparison of models or feature sets uses the corrected repeated
   k-fold t-test (`corrected_ttest`), never the naive one.

## How to work
- One step of `REFACTORING_STEPS.md` at a time; do not start the next step in
  the same session.
- Commits: one per step, message `refactor(step N): <what>`, made only after
  the acceptance checks pass. A bug found during a step that changes results
  (like the unseeded logreg of step 1) goes in its own earlier commit
  `fix(<scope>): <what>`, only after the user agrees. Add files by explicit
  path (deletions included); never commit `input/` or `results/`; never push.
- To run older code, use `git worktree add <scratch dir> <commit>` instead of
  `git stash`; never rewrite history (no amend, rebase, reset of commits).
- The golden reference (`results/golden/reference`) is re-exported only when
  the user asks (step 2, or a `fix` commit): export it twice in two processes
  and check that the two copies match before using it.
- After each step: golden check must pass with the default tolerance (1e-9).
  `tools/golden.py` may only change in `_export` (imports / calls of the new
  modules); its outputs and format must stay identical.
- If the golden check fails: stop, do not edit tests or golden files to make it
  pass, and report which file / column / rows differ and the likely cause.
  The only planned exception is the Pokédex change of step 2: report the new
  numbers, then the user decides and re-exports the reference.
- Prefer moving code over rewriting it; keep behaviour first, then clean up.
- If something in `REFACTORING.md` is ambiguous or conflicts with the code,
  keep the code's behaviour, note the conflict in the step report and ask.
- Determinism: every model has a fixed seed (logreg / liblinear included, since
  step 1); a new model or tuned-parameter file must not change that. Features
  sort before iterating a set / frozenset: the contract test requires the same
  output under two hash seeds, and a new feature must be added to it.

## Simplicity
- Write the simplest code that satisfies the step; a human must be able to read
  any file without the agent.
- No abstraction without two concrete uses today (no protocol, base class or
  registry for a single implementation, no "future" hooks).
- Functions under ~40 lines, modules under ~400 lines; split by responsibility.
- No defensive checks for impossible states; validate input once at the edges
  (record loading, CLI).
- Docstrings say why and the contract, not a line-by-line paraphrase.
- Delete what is replaced in the same step: no dead code, no compatibility shims
  left behind, no commented-out code.
- If the spec asks for something that looks over-engineered for its use, say so
  in the step report and propose the simpler version.

## Code style
- PEP 8 via ruff (line length 100), imports sorted, type hints everywhere;
  pyright strict on `domain/`, `interfaces.py`, `statistics.py`.
- English comments, only on lines whose reason is not obvious; no separator
  comments; a docstring on every public class and function.
- Tests next to each module family under `tests/`, small fixtures in
  `tests/fixtures/` (no full data in tests).
