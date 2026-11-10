"""Generic stacked members and their logit blend (the models of ``v11_fs``, on any table).

``stack_xgboost``, ``stack_lightgbm``, ``stack_catboost``: the cross-fitted logit of every
column (``StackedLogit``, C=0.003, kept next to the columns), then the booster, with the
hyper-parameters, seeds and threads of the frozen v11 members (``models/v11.py``). ``logreg_l2``:
median imputation, standardisation, L2 logistic regression (C=0.0007). ``stack_blend``: the
weighted mean of the four logits with the weights of ``blend_v11``, not renormalised. No column
name or prefix anywhere: the label-aware steps of the v11 members are features of ``v11_fs``.
"""

from __future__ import annotations

from typing import Any

from sklearn.pipeline import Pipeline

from interfaces import Model
from models.core.estimator import EstimatorModel
from models.core.logit_blend import LogitBlend
from models.core.preprocessing import require
from models.core.stacking import SEED, StackedLogit, logit_pipeline
from registry import model

N_JOBS = 4
STACK_C = 0.003
LOGREG_C = 0.0007
# the parameters of the frozen members (XGB, LGBM, CAT of models/v11.py)
XGBOOST: dict[str, Any] = {
    "n_estimators": 2294,
    "learning_rate": 0.007913887246783498,
    "max_depth": 3,
    "min_child_weight": 1.9697526022874337,
    "subsample": 0.49697118197877566,
    "colsample_bytree": 0.3877025327355442,
    "reg_lambda": 0.18317956984810999,
    "reg_alpha": 0.3574349749071232,
    "gamma": 1.1158538846021717,
}
LIGHTGBM: dict[str, Any] = {
    "n_estimators": 1923,
    "learning_rate": 0.008081605764973849,
    "num_leaves": 9,
    "min_child_samples": 76,
    "subsample": 0.9092954187203037,
    "subsample_freq": 1,
    "colsample_bytree": 0.21440317409883963,
    "reg_lambda": 0.009142949784245947,
    "max_depth": 4,
    "reg_alpha": 0.010952625812536086,
    "min_split_gain": 0.0796977105167146,
}
CATBOOST: dict[str, Any] = {
    "iterations": 2883,
    "learning_rate": 0.006388766767796231,
    "depth": 6,
    "l2_leaf_reg": 1.0125444690360867,
    "random_strength": 2.7787738509370583,
    "bagging_temperature": 0.16902526989254424,
    "rsm": 0.551655095551719,
}
# the weights of blend_v11 (WEIGHTS of models/v11.py), by twin
WEIGHTS = {
    "logreg_l2": 0.21039345977721677,
    "stack_xgboost": 0.14756062275115248,
    "stack_lightgbm": 0.2747445724394519,
    "stack_catboost": 0.3673013450321788,
}


def stacked(name: str, booster: Any) -> Model:
    """The cross-fitted logit of every column, kept, then ``booster``."""
    stack = StackedLogit(C=STACK_C, seed=SEED, name="stack_logit_full", keep=True)
    return EstimatorModel(name, Pipeline([("stack", stack), ("model", booster)]))


@model("logreg_l2")
def logreg_l2(name: str = "logreg_l2", C: float = LOGREG_C) -> Model:
    """Median imputation, standardisation, L2 logistic regression."""
    return EstimatorModel(name, logit_pipeline(C))


@model("stack_xgboost", requires=("xgboost",))
def stack_xgboost(name: str = "stack_xgboost", **params: Any) -> Model:
    """Stacked logit, XGBoost."""
    booster = require("xgboost").XGBClassifier(
        random_state=SEED, n_jobs=N_JOBS, tree_method="hist", eval_metric="logloss",
        **{**XGBOOST, **params},
    )  # fmt: skip
    return stacked(name, booster)


@model("stack_lightgbm", requires=("lightgbm",))
def stack_lightgbm(name: str = "stack_lightgbm", **params: Any) -> Model:
    """Stacked logit, LightGBM."""
    booster = require("lightgbm").LGBMClassifier(
        random_state=SEED, n_jobs=N_JOBS, deterministic=True, force_row_wise=True, verbose=-1,
        **{**LIGHTGBM, **params},
    )  # fmt: skip
    return stacked(name, booster)


@model("stack_catboost", requires=("catboost",))
def stack_catboost(name: str = "stack_catboost", **params: Any) -> Model:
    """Stacked logit, CatBoost."""
    booster = require("catboost").CatBoostClassifier(
        random_seed=SEED, thread_count=N_JOBS, verbose=0, allow_writing_files=False,
        **{**CATBOOST, **params},
    )  # fmt: skip
    return stacked(name, booster)


@model("stack_blend", requires=("xgboost", "lightgbm", "catboost"))
def stack_blend(name: str = "stack_blend") -> Model:
    """The four members, logit-averaged with the weights of ``blend_v11``."""
    members = [logreg_l2(), stack_xgboost(), stack_lightgbm(), stack_catboost()]
    return LogitBlend(name, members, [WEIGHTS[m.name] for m in members])
