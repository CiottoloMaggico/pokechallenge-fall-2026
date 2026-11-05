# PokeChallenge 2026 – instructions for coding agents

Predict the winner (`player_won`, P1 wins) of Gen 2 OU battles from the first
30 turns. The repository is being refactored following `REFACTORING.md`;
`REFACTORING_STEPS.md` has one prompt per step. Read both before any change.

## Environment
- Python venv in `.env/`: `source .env/bin/activate` before every command.
- Data: `input/train.jsonl`, `input/test.jsonl` (never commit them).
- Outputs only under `results/` (after the refactoring; `runs/` today).

## Commands
- Tests: `pytest -q` (once `tests/` exists); lint: `ruff check . && ruff format --check .`;
  types: `pyright`.
- Golden check (every step, a few minutes):
  `python tools/golden.py export --out results/golden/candidate`
  `python tools/golden.py check results/golden/reference results/golden/candidate`
- Full regression (only where a step says so, ~30 min):
  `python main.py evaluate --features compact_v5 --models logreg ensemble_from_lr --seeds 33 66 99`
  expected: ensemble_from_lr 0.7202 / 0.5401, logreg 0.7186 / 0.5425
  (re-measured once after the Pokédex change of step 2).

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
- One step of `REFACTORING_STEPS.md` at a time, one commit per step, message
  `refactor(step N): <what>`; do not start the next step in the same session.
- Before the first step: `python tools/golden.py export --out results/golden/reference`
  on the untouched code, and commit nothing under `results/`.
- After each step: golden check must pass with the default tolerance (1e-9).
  `tools/golden.py` may only change in `_export` (imports / calls of the new
  modules); its outputs and format must stay identical.
- If the golden check fails: stop, do not edit tests or golden files to make it
  pass, and report which file / column / rows differ and the likely cause.
  The only planned exception is the Pokédex change of step 2: report the new
  numbers, then the user decides and re-exports the reference.
- Prefer moving code over rewriting it; keep behaviour first, then clean up.
- If something in `REFACTORING.md` is ambiguous or conflicts with the code,
  ask instead of guessing.

## Code style
- PEP 8 via ruff (line length 100), imports sorted, type hints everywhere;
  pyright strict on `domain/`, `interfaces.py`, `statistics.py`.
- English comments, only on lines whose reason is not obvious; no separator
  comments; a docstring on every public class and function.
- Tests next to each module family under `tests/`, small fixtures in
  `tests/fixtures/` (no full data in tests).
