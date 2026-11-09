# PokeChallenge 2026 – who wins a Gen 2 OU battle?

Given the first 30 turns of a Pokémon Showdown Gen 2 OU battle, predict whether player 1 wins
(`player_won`). The training set has 10,000 labelled battles (`input/train.jsonl`), the test
set 5,000 unlabelled ones (`input/test.jsonl`). Each record holds P1's full team, P2's lead
and a timeline of up to 30 turns (both active Pokémon's HP, status, effects and boosts, and the
move each player used).

The pipeline:

1. **Replay** (`domain/`): every battle is replayed turn by turn into typed states (teams, HP,
   statuses, hazards, screens, revealed moves), with a Gen 2 type chart and damage formula. The
   Pokédex is built from the training records only.
2. **Knockout repair** (`addons/`): the records do not show every faint of P2's Pokémon. A
   detector (gradient boosting on P1's switches, a versioned artifact in
   `results/addons/knockouts/`) estimates the hidden knockouts and repairs the replay before any
   feature is computed.
3. **Features** (`features/`): named feature sets, from a single HP difference (`hp30`) to the
   selected `compact_v5` (18 columns) and `v11` (819 columns, a native port of the `blend_v11`
   model of the FDS course repository, with its own hidden-faint detector, team effects,
   simulations and a learned forecast of the next turns).
4. **Models** (`models/`): logistic regression, gradient boosting (scikit-learn, LightGBM,
   XGBoost, CatBoost), boosting started from the logistic regression's log-odds, probability
   averages, and the four v11 pipelines with their fixed logit blend.
5. **Gym** (`gym.py`, `folds.py`): repeated grouped k-fold cross-validation (duplicated battles
   always in the same fold), features refitted inside every fold, comparisons with the
   corrected repeated k-fold t-test (Nadeau & Bengio).

## Setup

Python 3.14 and Git LFS (the data files are stored with LFS).

```sh
git lfs pull                      # input/train.jsonl, input/test.jsonl
python -m venv .env
source .env/bin/activate
pip install -r requirements.txt   # pinned versions: the results depend on them
```

Two files are built once from the training data (both under `results/`, git-ignored):

```sh
# the Pokédex (species stats and types, from the training records only), ~20 s
python -m domain build-pokedex input/train.jsonl --out results/cache/species.json
# the knockout detector used by every command by default, ~1.5 min
python main.py detector train --version v3 --seeds 42 1 2
```

## Usage

Every command is `python main.py <command>`; `--help` lists the options. Outputs go under
`results/` unless `--out` says otherwise.

```sh
# inspect the data and the repair
python main.py check --train input/train.jsonl --test input/test.jsonl
python main.py replay --battle 42

# cross-validate feature sets x models (3 fold assignments x 5 folds)
python main.py evaluate --features compact_v5 --models logreg ensemble_from_lr \
    --seeds 33 66 133 --reference compact_v5/logreg --out results/evaluations/example

# fit on all the training battles and save the run, then write a submission
python main.py train --features compact_v5 --models ensemble_from_lr
python main.py predict --run results/runs/compact_v5__ensemble_from_lr \
    --test input/test.jsonl --out results/submissions/submission.csv \
    --proba results/submissions/submission_proba.csv
```

The v11 model (`v11` features, `v11_blend` model) trains in about 13 minutes and predicts the
test set in about 1.5 minutes; its submission equals the one of the original repository:

```sh
python main.py train --features v11 --models v11_blend --out results/runs/v11
python main.py predict --run results/runs/v11/v11__v11_blend --test input/test.jsonl \
    --out results/submissions/submission_v11_blend.csv
```

Other commands:

- `evaluate --turns 5 10 15 20 25 30`: the same features sampled at earlier turns
  (accuracy-vs-turn table).
- `tune --features <set> --model <model> --trials N`: Optuna search of a model's
  hyperparameters (on folds seeded differently from `evaluate`); `--params <file>` then uses
  them in `evaluate` / `train`.
- `select --method forward|backward --start <set> [--pool <set>]`: column selection with the
  corrected t-test.
- `detector evaluate | train | tune`: the knockout detector on its own task (rows = P1's
  switches).
- `--repair none | rule | model=<folder>`: which knockout repair to apply (a saved run predicts
  only with its own repair).

Feature sets: `hp30` (HP difference only), `material`, `hazards`, `switch`, `core` (the three
together), `compact` … `compact_v5` (successive selections, each with an `_orig` variant),
`compact_ko`, `compact_plus`, `selection_pool` (every candidate column) and `v11`. Models:
`logreg`, `hgb`, `lightgbm`, `xgboost`, `catboost`, the `*_lr` boosters (`gb_lr`, `lightgbm_lr`,
`xgboost_lr`, `catboost_lr`), the averages `ensemble` (logreg + hgb), `ensemble_boost`,
`ensemble_all`, `ensemble_from_lr` (the four `*_lr`), the v11 members `v11_lr_spline`,
`v11_xgb`, `v11_lgbm`, `v11_cat` and their blend `v11_blend`.

## Repository layout

```
main.py            command line (parses arguments, calls usecases.py)
usecases.py        one function per command
gym.py folds.py    cross-validation, fold plans, leakage checks
domain/            records, Pokédex, type chart, damage, replay, transitions (stdlib + numpy)
addons/            knockout repairs: none, rule, trained detector artifact
features/          one file per feature set; families in features/core/, v11 in features/core/v11/
models/            one file per model; shared pieces in models/core/
tuning/ selection.py stats_tables.py   Optuna search, column selection, corrected t-test
tests/             unit, contract and end-to-end tests on tests/fixtures/ (200 battles)
tools/golden.py    regression check: frozen outputs of the pipeline
```

The layers and the rules every change must respect (only training data fits anything,
duplicates in the same fold, out-of-fold label-aware features, …) are in `AGENTS.md`.

## Tests and checks

```sh
pytest                                         # ~4 min, reads only tests/fixtures/
ruff check . && ruff format --check .
pyright
```

CI (`.github/workflows/ci.yml`) runs the three on every push.

## Evaluation

(pending: the experiments are running)
