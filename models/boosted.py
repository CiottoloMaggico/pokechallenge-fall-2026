"""Generic boosted members and their logit blend (the models of ``v11_fs``, on any table).

``boost_xgboost``, ``boost_lightgbm``, ``boost_catboost``: the booster alone, with the
hyper-parameters, seeds and threads of the frozen v11 members (``models/v11.py``). ``logreg_l2``:
median imputation, standardisation, L2 logistic regression (C=0.0007). ``boost_blend``: the
weighted mean of the four logits with the weights of ``blend_v11``, not renormalised. No column
name or prefix anywhere: the label-aware steps of the v11 members (stacked logits, team effect)
are features of ``v11_fs``.
"""

from __future__ import annotations

from typing import Any

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from interfaces import Model
from models.core.estimator import EstimatorModel
from models.core.logit_blend import LogitBlend
from models.core.preprocessing import require
from registry import model

SEED = 42
N_JOBS = 4
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
    "boost_xgboost": 0.14756062275115248,
    "boost_lightgbm": 0.2747445724394519,
    "boost_catboost": 0.3673013450321788,
}


@model("logreg_l2")
def logreg_l2(name: str = "logreg_l2", C: float = LOGREG_C) -> Model:
    """Median imputation, standardisation, L2 logistic regression."""
    steps = [
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("lr", LogisticRegression(C=C, max_iter=5000)),
    ]
    return EstimatorModel(name, Pipeline(steps))


@model("boost_xgboost", requires=("xgboost",))
def boost_xgboost(name: str = "boost_xgboost", **params: Any) -> Model:
    """XGBoost with the parameters of ``v11_xgb``."""
    booster = require("xgboost").XGBClassifier(
        random_state=SEED, n_jobs=N_JOBS, tree_method="hist", eval_metric="logloss",
        **{**XGBOOST, **params},
    )  # fmt: skip
    return EstimatorModel(name, booster)


@model("boost_lightgbm", requires=("lightgbm",))
def boost_lightgbm(name: str = "boost_lightgbm", **params: Any) -> Model:
    """LightGBM with the parameters of ``v11_lgbm``."""
    booster = require("lightgbm").LGBMClassifier(
        random_state=SEED, n_jobs=N_JOBS, deterministic=True, force_row_wise=True, verbose=-1,
        **{**LIGHTGBM, **params},
    )  # fmt: skip
    return EstimatorModel(name, booster)


@model("boost_catboost", requires=("catboost",))
def boost_catboost(name: str = "boost_catboost", **params: Any) -> Model:
    """CatBoost with the parameters of ``v11_cat``."""
    booster = require("catboost").CatBoostClassifier(
        random_seed=SEED, thread_count=N_JOBS, verbose=0, allow_writing_files=False,
        **{**CATBOOST, **params},
    )  # fmt: skip
    return EstimatorModel(name, booster)


@model("boost_blend", requires=("xgboost", "lightgbm", "catboost"))
def boost_blend(name: str = "boost_blend") -> Model:
    """The four members, logit-averaged with the weights of ``blend_v11``."""
    members = [logreg_l2(), boost_xgboost(), boost_lightgbm(), boost_catboost()]
    return LogitBlend(name, members, [WEIGHTS[m.name] for m in members])
