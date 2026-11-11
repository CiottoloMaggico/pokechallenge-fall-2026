# Evaluation of the three submissions

The scripts, parameters and tables behind `docs/notebook/pokechallenge.ipynb`, the Kaggle
notebook that rebuilds the three submissions (`compact_v5`, `v11`, their 50/50 blend) and
reports their experimental evaluation.

| file | what it does |
|---|---|
| `blend.py` | the 50/50 logit blend of two models' probabilities (the third submission) |
| `truncate.py` | cuts every battle of a file to its first N turns |
| `learning_curve.py` | predictions of a feature set's models trained on nested subsets of a fold |
| `summarize.py` | turns the outputs of the experiments into the tables of `data/` |
| `plots.py` | figures and tables of the notebook (pandas, numpy, matplotlib only; Apple's system colours) |
| `params/` | the tuned hyper-parameters of `compact_v5` / `ensemble_from_lr` (`python main.py tune`) |
| `data/` | the tables the notebook shows when it does not recompute them |
| `figures/` | the figures of the notebook |

Tables of `data/` (one fold assignment, seed 33, stratified grouped 5-fold):

| table | content | produced by |
|---|---|---|
| `oof.csv`, `metrics.csv`, `paired.csv` | out-of-fold probabilities, metrics and paired differences (corrected t-test) of the three models | `evaluate` of `compact_v5` and `v11`, then `summarize.py models` |
| `accuracy_by_turn.csv` | `compact_v5` on battles cut to their first N turns, per fold | `truncate.py`, `evaluate`, `summarize.py turns` |
| `loss_curve.csv` | training and validation log loss by number of training battles (a point every 250, fold 0) | `learning_curve.py`, `summarize.py curve` |
| `feature_mapping.csv` | the groups of `v11` next to the `compact_v5` features that measure the same quantity | written by hand from the two feature sets |

The exact commands are the cells of the notebook guarded by `RUN_EVALUATION`; setting it to
`True` runs them again (several hours, most of them for `v11`).
