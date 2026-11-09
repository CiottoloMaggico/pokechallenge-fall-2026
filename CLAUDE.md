# PokeChallenge 2026 – instructions for coding agents

Predict the winner (`player_won`, P1 wins) of Gen 2 OU battles from the first
30 turns. The refactoring of `REFACTORING.md` is complete (steps 0-9 of
`REFACTORING_STEPS.md`, last commit `60707d3`): `REFACTORING.md` describes the
architecture and its rules; read it before any change.

## Environment
- Python venv in `.env/`: `source .env/bin/activate` before every command.
- Data: `input/train.jsonl`, `input/test.jsonl` (never commit them).
- Outputs only under `results/` (`storage.ResultsLayout`, since step 6). Older outputs in `runs/`
  and `results/<evaluation>/` stay where they are: do not move them.
- Knockout detector (since step 7): a versioned artifact in `results/addons/knockouts/<version>/`,
  loaded by every command (`--repair`, default `model=results/addons/knockouts/v3`). If `v3` is
  missing, rebuild it with `python main.py detector train --version v3 --seeds 42 1 2`
  (~2 min, same P2 repairs). Versions are never overwritten: a new detector gets a new version.

## Commands
Since step 4a no feature depends on the hash seed (contract test in
`tests/features/test_feature_contract.py`). `export PYTHONHASHSEED=0` is needed
only to run commits older than step 4a, such as the recipes in `SUBMISSIONS.md`.

- Tests: `pytest -q`; lint: `ruff check . && ruff format --check .`;
  types: `pyright`. CI (`.github/workflows/ci.yml`, since step 9) runs the three; no test
  reads `input/` or `results/`.
- Golden check (every step, ~10 min; needs the `v3` artifact, `export --repair` picks
  another):
  `python tools/golden.py export --out results/golden/candidate`
  `python tools/golden.py check results/golden/reference results/golden/candidate`
- Detector check (where a step says so, ~2 min):
  `python main.py detector evaluate --features ko_domain30 --models hgb_ko --seeds 42 1 2`
  Step 7 numbers: log loss 0.0992, AUC 0.9918, 10,109 P2 repairs, symmetry estimate 10,135.
- Tuning and selection (step 8): `python main.py tune --features <set> --model <m> --trials N`
  (seeds 2027, results in `results/tuning/`, used with `--params`);
  `python main.py select --method forward|backward --start <set> [--pool <set>]`
  (corrected t-test, report in `results/selections/`).
- Full regression (only where a step says so, ~30 min):
  `python main.py evaluate --features compact_v5 --models logreg ensemble_from_lr --seeds 33 66 133 --out results/reference_runs/<name>`
  Baseline of step 2 (commit ad7e4c4, run folder:
  `results/reference_runs/step2_pokedex`): ensemble_from_lr 0.7207 / 0.5402,
  logreg 0.7191 / 0.5426 (accuracy / log loss). Both must match that run fold by
  fold (step 9 re-checked it: `results/reference_runs/step9_final`, identical).

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
   with the same one.
5. Do not change formulas, hyper-parameters, seeds, thresholds, column names or
   column order of existing feature sets unless the step explicitly asks it.
6. No `fork` (multiprocessing, `--jobs`) after LightGBM / XGBoost / CatBoost
   have been used in the parent process: the workers hang at 0% CPU.
7. Every comparison of models or feature sets uses the corrected repeated
   k-fold t-test (`corrected_ttest`), never the naive one.

## How to work
- The refactoring steps are finished; new work is one change per commit, made only after
  its checks pass (golden check if results must not move, `pytest`, ruff, pyright).
  While the steps ran: one step of `REFACTORING_STEPS.md` at a time, commits
  `refactor(step N): <what>`. A bug fix that changes results (like the unseeded
  logreg of step 1) goes in its own commit `fix(<scope>): <what>`, only after the
  user agrees. Add files by explicit
  path (deletions included); never commit `input/` or `results/`; never push.
- To run older code, use `git worktree add <scratch dir> <commit>` instead of
  `git stash`; never rewrite history (no amend, rebase, reset of commits).
- The golden reference (`results/golden/reference`) is re-exported only when
  the user asks (step 2, or a `fix` commit): export it twice in two processes
  and check that the two copies match before using it.
- After each step (and each later change that must not move the results): golden check must pass with the default tolerance (1e-9).
  `tools/golden.py` may only change in `_export` (imports / calls of the new
  modules; step 7 also added `export --repair`); its outputs and format must stay
  identical.
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
  pyright strict on `domain/`, `interfaces.py`, `stats_tables.py`.
- English comments, only on lines whose reason is not obvious; no separator
  comments; a docstring on every public class and function.
- Tests next to each module family under `tests/`, small fixtures in
  `tests/fixtures/` (no full data in tests).
