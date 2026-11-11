# PokeChallenge 2026 – who wins a Gen 2 OU battle?

Given the first 30 turns of a Pokémon Showdown Gen 2 OU battle, predict whether player 1 wins
(`player_won`). The three submissions and their experimental evaluation are in the notebook
[`docs/notebook/pokechallenge.ipynb`](docs/notebook/pokechallenge.ipynb) (its scripts, tables
and figures in [`docs/evaluation/`](docs/evaluation/README.md)). The rules every change must
respect are in [`AGENTS.md`](AGENTS.md).

## Setup

Python 3.14 and Git LFS (the data files are stored with LFS).

```sh
git lfs pull                      # input/train.jsonl, input/test.jsonl
python -m venv .env
source .env/bin/activate
pip install -r requirements.txt   # pinned versions: the results depend on them

# built once from the training data, under results/ (git-ignored)
python -m domain build-pokedex input/train.jsonl --out results/cache/species.json
python main.py detector train --version v3 --seeds 42 1 2       # knockout detector, ~2 min
python main.py detector train-v11 --version v1 --seeds 42 1 2   # v11's detector, ~6 min
```

## Usage

Every command is `python main.py <command>` (`--help` lists the options); outputs go under
`results/` unless `--out` says otherwise.

```sh
# cross-validate feature sets x models on the same folds (corrected t-test vs --reference)
python main.py evaluate --features compact_v5 --models logreg ensemble_from_lr \
    --seeds 33 66 133 --reference compact_v5/logreg --out results/evaluations/example

# fit on all the training battles, then write a submission with the run
python main.py train --features compact_v5 --models ensemble_from_lr
python main.py predict --run results/runs/compact_v5__ensemble_from_lr \
    --test input/test.jsonl --out results/submissions/submission.csv
```

- `--repair none | rule | model=<folder> | v11=<folder>`: the knockout repair applied before
  any feature (default `model=results/addons/knockouts/v3`; `v11` uses
  `v11=results/addons/knockouts_v11/v1`). A run predicts only with its own repair.
- `--params <file> …`: tuned hyper-parameters (`assets/params/<set>__<model>.json`), produced by
  `tune --features <set> --model <model> --trials N`.
- `select --method forward|backward --start <set>`: column selection with the corrected t-test.
- `check` and `replay --battle <id>`: inspect the data and the repaired replay.

The exact commands of the three submissions are the cells of the notebook.

## Layout

```
main.py  usecases.py      command line (parses arguments) and one function per command
gym.py  folds.py          cross-validation on grouped folds, leakage checks
domain/                   records, Pokédex, type chart, damage, replay (stdlib + numpy only)
addons/                   knockout repairs applied to every battle before the features
features/                 one file per feature set; shared families in features/core/
models/                   one file per model; shared pieces in models/core/
tuning/  selection.py     Optuna search, column selection (stats_tables.py: corrected t-test)
docs/                     the Kaggle notebook and its evaluation
tests/                    unit, contract and architecture tests on tests/fixtures/
```

Imports go only downwards: `main.py` → application (`usecases`, `gym`, …) → `addons`,
`features`, `models` → `domain`. Features never import models or addons, models never import
the domain or the features (they see a `DataFrame` and labels only), the domain never imports
anything of the project. `tests/test_architecture.py` enforces it.

## Adding a feature set

1. Create `features/<name>.py` with a builder registered under its name:

   ```python
   from domain.battle import BattleContext
   from feature_set import FeatureSet
   from interfaces import FeatureValue
   from registry import feature_set


   def hp_diff_t30(context: BattleContext) -> FeatureValue:
       """P1 total HP minus P2 total HP at the last recorded turn."""
       last = context.last
       return "hp_diff_t30", last.p1.hp_total - last.p2.hp_total


   @feature_set("my_set")
   def my_set(turn: int | None = None) -> FeatureSet[BattleContext]:
       """What the set measures."""
       features = FeatureSet[BattleContext]("my_set")
       features.register(hp_diff_t30)
       return features
   ```

2. Import the file in `features/sets.py`; `--features my_set` then works in every command.
3. Reuse the families of `features/core/` (`make_set` in `features/common.py` builds a set from
   them, `AtTurn` samples a feature at an earlier turn); put new reusable features there.
4. A feature that learns from battles is a `Feature` subclass (`interfaces.py`) with
   `requires_fit = True`; if it reads the labels, `uses_labels = True`: the gym fits it inside
   every fold and gives its training rows out-of-fold values, never in-sample ones.
5. A feature never depends on the hash seed (sort sets before iterating) and the existing sets
   are never changed (their columns, order and formulas are part of the results): a variation
   is a new set. `tests/features/test_feature_contract.py` checks every registered set.

## Adding a model

1. Create `models/<name>.py` with a builder registered under its name; keyword arguments
   override the defaults, and the seed is fixed:

   ```python
   from typing import Any

   from sklearn.ensemble import RandomForestClassifier

   from interfaces import Model
   from models.core.estimator import EstimatorModel
   from registry import model

   PARAMS: dict[str, Any] = {"n_estimators": 500, "random_state": 0}


   @model("forest")
   def build(name: str = "forest", **params: Any) -> Model:
       """Random forest on the raw columns."""
       return EstimatorModel(name, RandomForestClassifier(**{**PARAMS, **params}))
   ```

2. Import the file in `models/__init__.py`; `--models forest` then works in every command.
3. `EstimatorModel` wraps anything with `fit` and `predict_proba`; a model that is not an
   estimator subclasses `Model` (`interfaces.py`). Pass `space=` to `@model` to make it tunable
   with `tune`, and `requires=("lightgbm",)` for an optional library.
4. An average of registered models goes in `models/ensembles.py` with `@ensemble(name,
   members=(...))`.
5. A model is never tied to a feature set: it reads only the `DataFrame` it is given.
   `tests/models/test_model_contract.py` checks every registered model (add a small size to
   its `SMALL` table for a slow one).

## Checks

```sh
pytest                                    # reads only tests/fixtures/
ruff check . && ruff format --check .
pyright
```

CI (`.github/workflows/ci.yml`) runs the three on every push.
