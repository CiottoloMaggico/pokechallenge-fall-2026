# Comparison with two other teams

Our three submissions (`compact_v5`, `v11`, `blend`) next to the models of two other teams,
rebuilt here from their code and scored on our folds:

| name | team | repository | public rank | private rank |
|---|---|---|---:|---:|
| ours | | this one | 2 | 1 |
| 2nd place (`second*`) | | `FDS-Pokemon-Challenge-2026-2027`, commit `8479064` | 7 | 2 |
| 8th place (`third*`) | Fantastic Four | `FDS-Challenge-2026`, commit `1a3ef2d` | 3 | 8 |

**The analysis is in [`comparison.ipynb`](comparison.ipynb)**; this file says how its tables
were produced. The notebook also runs on Kaggle: there it clones this repository (Internet
access on, no competition data needed) and reads the tables of `data/`.

| file | what it does | environment |
|---|---|---|
| `comparison.ipynb` | the analysis: reads `data/`, draws with `draw.py` | pandas, numpy, matplotlib |
| `draw.py` | the figures and tables of the notebook | pandas, numpy, matplotlib |
| `compare.py` | the tables of `data/` from every team's predictions | `.env` |
| `stability.py` | validation estimates under three protocols, simulated leaderboards | `.env` |
| `second_place.py` | 2nd place: their models on our folds (`cv`, `stage`), on the subsets of our learning curve (`curve`), with their own protocol (`test`) | theirs (Python 3.12) |
| `fantastic_four.py` | 8th place: a notebook with its own and our folds (`run`), refitted with each of our folds as test set (`deploy`), on the subsets of our learning curve (`curve`) | theirs (Python 3.13) |
| `curve_subsets.py` | the nested training subsets of `../learning_curve.py` (fold 0) | `.env` |

The other teams' repositories are never modified: the scripts import the 2nd place's `src`
package and call its own worker, and execute the 8th place's notebooks cell by cell in memory.
Every number comes from models rebuilt in this environment, except the leaderboard scores.

## Models compared

| name in `data/` | submission | model |
|---|---|---|
| `compact_v5`, `v11`, `blend` | ours | see `docs/notebook/pokechallenge.ipynb` |
| `second_five_views` | `matched_core_summary` | mean of five feature views (LightGBM + XGBoost) |
| `second_teams` | `team_similarity` | five views (with their `--feature-fixes`), then a logistic regression with similar-team win rates |
| `second` | `tactical_team_spikes` | their tactical view, then the logistic regression with similar teams and Spikes: their best private score |
| `third_ensemble` | notebook 6 | (2 × XGBoost / LightGBM / HistGB + 2 × CatBoost + logistic regression) / 5 |
| `third_catboost` | notebook 7 | CatBoost |
| `third` | notebook 8 | (3 × XGBoost / LightGBM / HistGB + logistic regression) / 4: their best private score |

## Tables of `data/`

All out of fold on our seed-33 stratified grouped 5 folds (those of `../data/oof.csv`).

| table | content | produced by |
|---|---|---|
| `oof.csv`, `folds.csv`, `metrics.csv` | probabilities per battle, metrics per fold and their mean, for the nine submissions | `compare.py` |
| `paired.csv` | paired differences per fold, corrected repeated k-fold t-test | `compare.py` |
| `loss_curve.csv` | validation log loss on fold 0 by number of training battles | `compare.py` |
| `agreement.csv` | share of battles with the same predicted winner, out of fold and on the test set | `compare.py` |
| `estimates.csv` | accuracy and log loss with each team's own validation, on our folds, and on our folds refitted | `stability.py` |
| `split_simulation.csv` | accuracy on the 1,500 "public" and 3,500 "private" battles of 5,000 resampled test sets | `stability.py` |
| `calibration.csv`, `accuracy_vs_confidence.csv`, `disagreement.csv` | computed from `oof.csv` alone | `draw.py` (the notebook) |
| `leaderboard.csv` | public and private scores: ours from Kaggle, the others as reported in their repositories | written by hand |

## How the other teams' models were rebuilt

**Environments**, outside every repository (`uv`): Python 3.12.14 and the 2nd place's
`requirements.txt`; Python 3.13 with numpy 2.1.3, pandas 2.3.3, scikit-learn 1.6.1, xgboost 3.4.1,
lightgbm 4.6.0, catboost 1.2.10 (the versions the 8th place's notebooks print). A third one runs
the notebook (pandas, numpy, matplotlib, jinja2, nbconvert).

**Data**: `results/comparison/data/` with links to `input/train.jsonl`, `input/test.jsonl`,
`assets/type_chart_gen2.csv` and a `sample_submission.csv` with the ids of `test.jsonl`.

**Checks of the rebuild.**

| | rebuilt here | reported by the team |
|---|---:|---:|
| 2nd place: unit tests | 9 / 9 pass | |
| 2nd place: `team_similarity`, accuracy with their validation (mean of 3 partitions / pooled) | 0.7077 / 0.7106 | 0.7080 / 0.7118 (their Kaggle rebuild: 0.7106) |
| 2nd place: `tactical_team_spikes`, same | 0.7072 / 0.7085 | 0.7077 / 0.7085 |
| 2nd place: test labels equal to the submitted files | 4,964 and 4,922 of 5,000 | their Kaggle rebuild: 4,964 and 4,922 |
| 8th place: accuracy and AUC of every fold, notebooks 6 / 7 / 8 | 0.7003 / 0.6968 / 0.6984, every fold equal | 0.7003 / 0.6968 / 0.6984 |

**On our folds.**

* *2nd place.* Our partition is stored next to theirs and fitted by their worker. Their groups
  are identical timelines, ours identical battles: our folds split three timelines (six battles
  that differ only in P1's team details), which get one group per fold, or their leakage
  assertion stops the worker. Their last step (the logistic regression) is fitted on our four
  training folds and applied to the fifth. Their pipeline is deterministic (a second run gives
  identical probabilities).
* *8th place.* `run --cv ours` runs their validation cell with our partition; their lookup
  tables are then built on all the training battles, as in their notebook. `deploy` runs the
  whole notebook once per fold, with the training folds as `train.jsonl` and the held-out fold,
  without labels, as `test.jsonl`: these are the `third*` columns of `oof.csv`.
* *Learning curve.* Both teams' models are refitted on the nested subsets of our own curve
  (fold 0). The 2nd place's line is their tactical view without the last step, which needs
  out-of-fold rows of every battle. For both teams the 8,000-battle point equals fold 0 of the
  runs above bit for bit.

## Commands

`PY2` / `PY3`: the Python of the 2nd / 8th place's environment; `REPO2` / `REPO3`: their
repositories; `R=$PWD/results/comparison`. Set
`export PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0` first.

```bash
# sample_submission.csv: the test ids in the order of test.jsonl
python -c "import json; ids = [json.loads(l)['battle_id'] for l in open('input/test.jsonl')]; \
  open('$R/data/sample_submission.csv', 'w').write('battle_id,player_won\n' + ''.join(f'{i},0\n' for i in ids))"
python docs/evaluation/comparison/curve_subsets.py --out $R/curve_subsets.csv
```

2nd place: own protocol (~1 h 15 with 3 jobs), our folds (~10 min per fold), learning curve:

```bash
$PY2 $REPO2/run.py predict --feature-fixes --data $R/data --output $R/own_protocol/fixed --jobs 3
$PY2 $REPO2/run.py stage --data $R/data --output $R/own_protocol/fixed --view blend --signals team --name team_similarity
$PY2 $REPO2/run.py stage --data $R/data --output $R/own_protocol/fixed --view tactical --signals team,spikes --name tactical_team_spikes
$PY2 docs/evaluation/comparison/second_place.py test --repo $REPO2 --data $R/data --output $R/own_protocol/fixed
$PY2 docs/evaluation/comparison/second_place.py cv --repo $REPO2 --data $R/data --output $R/our_folds/original --jobs 5
$PY2 docs/evaluation/comparison/second_place.py cv --repo $REPO2 --data $R/data --output $R/our_folds/fixed --jobs 5 --feature-fixes
$PY2 docs/evaluation/comparison/second_place.py stage --repo $REPO2 --data $R/data --output $R/our_folds/fixed
$PY2 docs/evaluation/comparison/second_place.py curve --repo $REPO2 --data $R/data --output $R/curve --subsets $R/curve_subsets.csv --jobs 5 --feature-fixes
```

8th place (~20–50 min per notebook and step):

```bash
for n in 6 7 8; do nb=$(ls "$REPO3"/submission\ $n\ *.ipynb)
  $PY3 docs/evaluation/comparison/fantastic_four.py run --notebook "$nb" --data $R/data --output $R/third/nb$n --cv own ours
  $PY3 docs/evaluation/comparison/fantastic_four.py deploy --notebook "$nb" --data $R/data --output $R/third/deploy_nb$n --jobs 2
done
$PY3 docs/evaluation/comparison/fantastic_four.py curve --notebook "$(ls "$REPO3"/submission\ 8\ *.ipynb)" --data $R/data --output $R/third/curve_nb8 --subsets $R/curve_subsets.csv
```

Our test probabilities (rebuilt from the notebook's runs), then the tables and the notebook:

```bash
python main.py predict --run results/notebook/runs/compact_v5__ensemble_from_lr-tuned --test input/test.jsonl --out $R/ours/submission_compact_v5.csv --proba $R/ours/proba_compact_v5.csv
python main.py predict --run results/notebook/runs/v11__v11_blend --test input/test.jsonl --repair v11=results/addons/knockouts_v11/v1 --out $R/ours/submission_v11.csv --proba $R/ours/proba_v11.csv
python docs/evaluation/blend.py $R/ours/proba_compact_v5.csv $R/ours/proba_v11.csv --out $R/ours/submission_blend.csv --proba $R/ours/proba_blend.csv
python docs/evaluation/comparison/compare.py --results $R
python docs/evaluation/comparison/stability.py --results $R
jupyter nbconvert --to notebook --execute --inplace docs/evaluation/comparison/comparison.ipynb
```
