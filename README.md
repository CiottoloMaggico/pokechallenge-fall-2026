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
   feature is computed. v11 has its own detector, also a repair addon with a versioned artifact
   (`results/addons/knockouts_v11/`).
3. **Features** (`features/`): named feature sets, from a single HP difference (`hp30`) to the
   selected `compact_v5` (18 columns) and `v11` (819 columns, the `blend_v11` model of the FDS
   course repository rebuilt on this repository's constructs: same logic, with team effects,
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

The v11 model (`v11` features, `v11_blend` model) reproduces the `blend_v11` model of the
original FDS repository (same feature and model logic; its detector is cross-fitted with this
repository's folds, which moves the numbers only within noise). It needs its detector artifact
(about 6 minutes, once), then trains in about 15 minutes and predicts the test set in about
a minute:

```sh
python main.py detector train-v11 --version v1 --seeds 42 1 2
python main.py train --features v11 --models v11_blend \
    --repair v11=results/addons/knockouts_v11/v1 --out results/runs/v11
python main.py predict --run results/runs/v11/v11__v11_blend --test input/test.jsonl \
    --repair v11=results/addons/knockouts_v11/v1 \
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
  switches); `detector train-v11`: the artifact of v11's detector.
- `--repair none | rule | model=<folder> | v11=<folder>`: which knockout repair to apply (a
  saved run predicts only with its own repair; v11 uses `v11=`).

Feature sets: `hp30` (HP difference only), `material`, `hazards`, `switch`, `core` (the three
together), `compact` … `compact_v5` (successive selections, each with an `_orig` variant),
`compact_ko`, `compact_plus`, `selection_pool` (every candidate column), `v11` (frozen) and
`v11_fs` (the living line of v11, a two-stage set: v11's groups, then bag logit, full logit
and team effect as features; about 11 min of features per fold of 8,000 battles). Models:
`logreg`, `hgb`, `lightgbm`, `xgboost`, `catboost`, the `*_lr` boosters (`gb_lr`, `lightgbm_lr`,
`xgboost_lr`, `catboost_lr`), the averages `ensemble` (logreg + hgb), `ensemble_boost`,
`ensemble_all`, `ensemble_from_lr` (the four `*_lr`), the v11 members `v11_lr_spline`,
`v11_xgb`, `v11_lgbm`, `v11_cat` and their blend `v11_blend` (frozen), the generic models of
`v11_fs` `logreg_l2`, `boost_xgboost`, `boost_lightgbm`, `boost_catboost` and their blend
`boost_blend`.

## Repository layout

```
main.py            command line (parses arguments, calls usecases.py)
usecases.py        one function per command
gym.py folds.py    cross-validation, fold plans, leakage checks
domain/            records, Pokédex, type chart, damage, replay, transitions, observed timeline,
                   expected-damage estimate, switch descriptions (stdlib + numpy)
addons/            knockout repairs: none, rule, trained detector artifact, v11's detector
features/          one file per feature set; families in features/core/, v11 in features/core/v11/
                   (frozen) and v11_fs in features/core/v11_fs/ (the living line)
staged_feature_set.py   two-stage feature sets: battle features, then features on their table
models/            one file per model; shared pieces in models/core/
tuning/ selection.py stats_tables.py   Optuna search, column selection, corrected t-test
tests/             unit, contract and end-to-end tests on tests/fixtures/ (200 battles)
tools/golden.py    regression check: frozen outputs of the pipeline
tools/v11_check.py v11 regression check against the original's outputs (bit for bit)
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

**Protocol.** Every number below is a cross-validation on the 10,000 training battles:
stratified grouped 5-fold (duplicated battles in the same fold), repeated on three fold
assignments (seeds 33, 66, 133), so 15 folds per row; all rows share the same folds. Features
are refitted inside every training fold (label-aware ones out of fold). "± fold" is the
fold-to-fold std, "± seed" the std of the mean across the three fold assignments. Δ columns
compare each row with a reference on the same 15 folds; p is the corrected repeated k-fold
t-test (Nadeau & Bengio), which does not treat overlapping folds as independent. The raw tables
are in `results/evaluations/` (`summary.csv`, `folds.csv`, `predictions.csv`).

### Models on `compact_v5` (18 columns)

Reference for Δ: `logreg`. For log loss, a negative Δ is better.

| model | accuracy | ± fold | ± seed | log loss | ROC AUC | Brier | Δ accuracy (p) | Δ log loss (p) |
|---|---|---|---|---|---|---|---|---|
| lightgbm_lr | 0.7208 | 0.0081 | 0.0004 | 0.5404 | 0.7986 | 0.1829 | +0.0017 (0.36) | −0.0022 (0.08) |
| xgboost_lr | 0.7208 | 0.0077 | 0.0015 | 0.5404 | 0.7986 | 0.1829 | +0.0017 (0.35) | −0.0022 (0.07) |
| ensemble_from_lr | 0.7207 | 0.0084 | 0.0009 | 0.5402 | 0.7987 | 0.1829 | +0.0016 (0.38) | −0.0024 (0.05) |
| catboost_lr | 0.7205 | 0.0085 | 0.0008 | 0.5401 | 0.7987 | 0.1829 | +0.0014 (0.44) | −0.0025 (0.04) |
| gb_lr | 0.7202 | 0.0091 | 0.0012 | 0.5402 | 0.7987 | 0.1829 | +0.0011 (0.52) | −0.0024 (0.06) |
| ensemble_from_lr-tuned | 0.7199 | 0.0070 | 0.0013 | 0.5399 | 0.7987 | 0.1828 | +0.0008 (0.75) | −0.0026 (0.11) |
| logreg | 0.7191 | 0.0084 | 0.0016 | 0.5426 | 0.7968 | 0.1837 | — | — |
| ensemble | 0.7185 | 0.0093 | 0.0027 | 0.5419 | 0.7971 | 0.1836 | −0.0006 (0.79) | −0.0007 (0.54) |
| xgboost | 0.7151 | 0.0088 | 0.0025 | 0.5446 | 0.7947 | 0.1847 | −0.0040 (0.35) | +0.0020 (0.42) |
| ensemble_boost | 0.7148 | 0.0090 | 0.0012 | 0.5434 | 0.7955 | 0.1843 | −0.0043 (0.22) | +0.0009 (0.65) |
| ensemble_all | 0.7147 | 0.0095 | 0.0019 | 0.5441 | 0.7949 | 0.1845 | −0.0044 (0.28) | +0.0015 (0.46) |
| lightgbm | 0.7133 | 0.0078 | 0.0022 | 0.5476 | 0.7923 | 0.1859 | −0.0058 (0.18) | +0.0050 (0.09) |
| hgb | 0.7117 | 0.0081 | 0.0023 | 0.5495 | 0.7906 | 0.1867 | −0.0074 (0.10) | +0.0070 (0.02) |
| catboost | 0.7054 | 0.0090 | 0.0025 | 0.5611 | 0.7849 | 0.1905 | −0.0137 (0.02) | +0.0185 (<0.001) |

- No model is significantly more accurate than the logistic regression: the 18 columns are
  already close to linear in the log-odds, and the fold-to-fold std (≈0.008) is five times the
  largest gain.
- The boosters started from the logistic regression's log-odds (`*_lr`) and their average
  (`ensemble_from_lr`, the submitted model with tuned members) improve the log loss by about
  0.0024 (p ≈ 0.04–0.08): better-calibrated probabilities, the same decisions.
- Plain gradient boosting from scratch is worse than the linear model on these few, smooth
  columns; CatBoost with default settings is significantly worse (p = 0.02 on accuracy).
- Tuning the members (`ensemble_from_lr-tuned`, Optuna on log loss) moves nothing beyond noise.

### Feature sets with `logreg`

Reference for Δ: `compact_v5`.

| feature set | columns | accuracy | ± fold | log loss | ROC AUC | Δ accuracy (p) | Δ log loss (p) |
|---|---|---|---|---|---|---|---|
| compact_v5 | 18 | 0.7191 | 0.0084 | 0.5426 | 0.7968 | — | — |
| compact_v5_orig | 26 | 0.7191 | 0.0075 | 0.5421 | 0.7973 | −0.0000 (1.00) | −0.0005 (0.20) |
| compact_v4_orig | 20 | 0.7149 | 0.0077 | 0.5482 | 0.7922 | −0.0042 (0.22) | +0.0056 (<0.001) |
| selection_pool | 37 | 0.7146 | 0.0084 | 0.5485 | 0.7918 | −0.0045 (0.23) | +0.0059 (<0.001) |
| compact_v2_orig | 19 | 0.7146 | 0.0083 | 0.5496 | 0.7907 | −0.0045 (0.21) | +0.0070 (<0.001) |
| compact_v3_orig | 19 | 0.7146 | 0.0086 | 0.5496 | 0.7907 | −0.0045 (0.25) | +0.0070 (<0.001) |
| compact_plus | 17 | 0.7146 | 0.0072 | 0.5539 | 0.7873 | −0.0045 (0.19) | +0.0114 (<0.001) |
| compact_v4 | 16 | 0.7142 | 0.0101 | 0.5485 | 0.7919 | −0.0049 (0.13) | +0.0059 (<0.001) |
| compact_v2 | 15 | 0.7140 | 0.0083 | 0.5498 | 0.7906 | −0.0051 (0.05) | +0.0072 (<0.001) |
| compact_v3 | 15 | 0.7140 | 0.0083 | 0.5498 | 0.7906 | −0.0051 (0.05) | +0.0072 (<0.001) |
| compact_ko | 19 | 0.7137 | 0.0073 | 0.5508 | 0.7897 | −0.0054 (0.12) | +0.0082 (<0.001) |
| compact | 11 | 0.7000 | 0.0102 | 0.5650 | 0.7756 | −0.0191 (0.002) | +0.0224 (<0.001) |
| compact_orig | 14 | 0.6999 | 0.0100 | 0.5649 | 0.7756 | −0.0192 (0.002) | +0.0224 (<0.001) |
| core | 14 | 0.6990 | 0.0071 | 0.5700 | 0.7702 | −0.0201 (<0.001) | +0.0274 (<0.001) |
| material | 8 | 0.6842 | 0.0059 | 0.5812 | 0.7567 | −0.0349 (<0.001) | +0.0386 (<0.001) |
| hp30 | 1 | 0.6765 | 0.0103 | 0.5925 | 0.7452 | −0.0426 (<0.001) | +0.0499 (<0.001) |
| hazards | 4 | 0.5574 | 0.0082 | 0.6829 | 0.5805 | −0.1617 (<0.001) | +0.1403 (<0.001) |
| switch | 2 | 0.5496 | 0.0101 | 0.6840 | 0.5730 | −0.1695 (<0.001) | +0.1414 (<0.001) |

- The HP difference at the last turn alone (`hp30`) already reaches 67.7 %; material, hazards
  and switching together (`core`) reach 69.9 %, and the knockout repair with the team / pool
  features (`compact_ko` … `compact_v5`) 71.9 %.
- `compact_v5` has a significantly lower log loss than every earlier set (p < 0.001); on
  accuracy only the gap to `compact` and the smaller sets is significant. The revalidation that
  removed 8 columns of `compact_v5_orig` lost nothing.

### Knockout detector (rows = P1's switches, seeds 42, 1, 2)

| features | model | accuracy | log loss | ROC AUC | Brier |
|---|---|---|---|---|---|
| ko_domain30 | hgb_ko | 0.9627 | 0.0992 | 0.9918 | 0.0281 |
| ko_raw50 | hgb_ko | 0.9625 | 0.0993 | 0.9917 | 0.0280 |
| ko_raw50 | lightgbm | 0.9619 | 0.1056 | 0.9911 | 0.0293 |
| ko_domain30 | lightgbm | 0.9616 | 0.1067 | 0.9909 | 0.0296 |
| ko_raw50 | logreg | 0.9218 | 0.2005 | 0.9677 | 0.0579 |
| ko_domain30 | logreg | 0.9052 | 0.2289 | 0.9590 | 0.0685 |

The 30 domain columns match the 50 raw ones with the default detector (`hgb_ko`, artifact
`v3`: 10,109 repaired P2 knockouts on the training battles).

### v11 (`--repair v11=results/addons/knockouts_v11/v1`)

Reference for Δ: `v11` / `v11_blend`. The plain models (`logreg`, `hgb`, …) are fitted on the
same 819 columns.

| model | accuracy | ± fold | ± seed | log loss | ROC AUC | Brier | Δ accuracy (p) | Δ log loss (p) |
|---|---|---|---|---|---|---|---|---|
| v11_lgbm | 0.7192 | 0.0069 | 0.0022 | 0.5424 | 0.7973 | 0.1836 | +0.0001 (0.98) | +0.0015 (0.04) |
| v11_blend | 0.7192 | 0.0048 | 0.0016 | 0.5409 | 0.7985 | 0.1830 | — | — |
| v11_cat | 0.7185 | 0.0070 | 0.0019 | 0.5421 | 0.7980 | 0.1834 | −0.0007 (0.75) | +0.0012 (0.09) |
| v11_xgb | 0.7179 | 0.0063 | 0.0013 | 0.5432 | 0.7968 | 0.1839 | −0.0013 (0.64) | +0.0022 (0.003) |
| v11_lr_spline | 0.7155 | 0.0061 | 0.0012 | 0.5467 | 0.7928 | 0.1855 | −0.0036 (0.22) | +0.0058 (0.004) |
| xgboost | 0.7105 | 0.0043 | 0.0019 | 0.5518 | 0.7879 | 0.1876 | −0.0087 (0.01) | +0.0109 (<0.001) |
| lightgbm | 0.7088 | 0.0058 | 0.0020 | 0.5535 | 0.7870 | 0.1882 | −0.0104 (0.01) | +0.0126 (<0.001) |
| catboost | 0.7083 | 0.0082 | 0.0039 | 0.5581 | 0.7863 | 0.1897 | −0.0108 (0.05) | +0.0172 (<0.001) |
| logreg | 0.7072 | 0.0058 | 0.0017 | 0.5553 | 0.7851 | 0.1889 | −0.0119 (0.002) | +0.0143 (<0.001) |
| hgb | 0.7071 | 0.0057 | 0.0017 | 0.5542 | 0.7860 | 0.1886 | −0.0121 (0.01) | +0.0133 (<0.001) |

`v11_fs`, the living line of v11 (`results/evaluations/models_v11_fs_closeness`, same seeds,
folds and repair): the bag logit, the logit of every non-key column and the team effect against
it are label-aware features of a two-stage set (out of fold on grouped inner folds), which gives
every member the table of the original's boosters; the generic models name no column and the
boosters run alone. Δ against each model's frozen twin on the same 15 folds (corrected t-test);
"apart" is the share of out-of-fold labels that differ from the twin's:

| model (twin) | accuracy | ± fold | ± seed | log loss | ROC AUC | Brier | Δ accuracy (p) | Δ log loss (p) | apart |
|---|---|---|---|---|---|---|---|---|---|
| boost_blend (v11_blend) | 0.7186 | 0.0044 | 0.0012 | 0.5407 | 0.7987 | 0.1829 | −0.0005 (0.75) | −0.0002 (0.76) | 2.5 % |
| boost_catboost (v11_cat) | 0.7177 | 0.0045 | 0.0020 | 0.5421 | 0.7980 | 0.1834 | −0.0008 (0.78) | −0.0001 (0.93) | 3.1 % |
| boost_xgboost (v11_xgb) | 0.7176 | 0.0061 | 0.0005 | 0.5430 | 0.7971 | 0.1838 | −0.0003 (0.89) | −0.0002 (0.88) | 3.3 % |
| boost_lightgbm (v11_lgbm) | 0.7172 | 0.0063 | 0.0002 | 0.5424 | 0.7972 | 0.1837 | −0.0020 (0.41) | +0.0000 (0.97) | 3.3 % |
| logreg_l2 (v11_lr_spline) | 0.7136 | 0.0077 | 0.0010 | 0.5472 | 0.7927 | 0.1856 | −0.0020 (0.57) | +0.0005 (0.64) | 5.1 % |

- Every generic model is within noise of its twin (all p ≥ 0.41). The boosters are as close to
  their twins as the twins are to themselves when only the seed of their inner folds changes
  (seed 33: 2.9–3.3 % of the labels apart, blend 2.0 %); `logreg_l2` is not (the original's
  logistic member reads the raw bags and splines of 25 columns, not this table). On the test
  file `boost_blend` and `v11_blend` give 122 different labels (mean |Δp| 0.014).

- Against the best `compact_v5` models on the same 15 folds (merged comparison): `v11_blend`
  ties `compact_v5` / `logreg` (Δ accuracy +0.0001, p 0.99; Δ log loss −0.0017, p 0.50) and
  `ensemble_from_lr` (−0.0015, p 0.73; +0.0007, p 0.74). Its fold-to-fold std is the lowest
  (0.0048).
- The four members are within noise of each other in accuracy; the blend has the lowest log
  loss, significantly lower than the XGBoost and spline members'. The 819 columns need the
  members' stacked logits and team effect: plain models on them lose about 1 point (p ≤ 0.05).
- In the original's own protocol (below) the rebuilt v11 gives 0.7197 / 0.5406 / 0.7988 for the
  blend (original 0.7203 / 0.5410 / 0.7983), within the seed-to-seed spread; this repository's
  protocol (features refitted inside each fold) is slightly lower in accuracy, as expected.

Protocol notes. v11's detector is a repair addon fitted once on the training file, like `v3`, so
in these cross-validations it is not refitted inside each fold; it never reads the outcome. Its
cross-fitted training values use this repository's grouped, stratified folds (mean over seeds
42, 1, 2) instead of the original's plain `KFold` over battles: on the detector this moves the
probabilities less than re-seeding the original's own folds (mean |Δp| 0.012 vs 0.015). As in
the original, v11's inner cross-validations (stacked logits, team effect, forecast) are not
grouped by duplicates. The repair is independent of the model: the reference rows use
`--repair v11=…`; v11 can also run with `v3` (a variant). The original, rerun in this
environment with its own protocol (features built once on all training battles, 5 folds × 3
repeats, seeds 42–44), gives:

| pipeline | accuracy | log loss | ROC AUC | Brier |
|---|---|---|---|---|
| lr_spline_v11 | 0.7166 | 0.5468 | 0.7927 | 0.1855 |
| xgb_v11 | 0.7196 | 0.5434 | 0.7965 | 0.1840 |
| lgbm_v11 | 0.7201 | 0.5429 | 0.7967 | 0.1838 |
| cat_v11 | 0.7204 | 0.5417 | 0.7982 | 0.1832 |
| blend_v11 | 0.7203 (nested 0.7204) | 0.5410 | 0.7983 | 0.1831 |

The same protocol on the rebuilt v11's training matrix (`--repair v11=…`, same folds): lr_spline
0.7173 / 0.5467 / 0.7929 / 0.1854, xgb 0.7197 / 0.5424 / 0.7975 / 0.1835, lgbm 0.7186 /
0.5423 / 0.7975 / 0.1835, cat 0.7209 / 0.5417 / 0.7984 / 0.1832, blend 0.7197 (nested 0.7203) /
0.5406 / 0.7988 / 0.1829 (accuracy / log loss / ROC AUC / Brier).
