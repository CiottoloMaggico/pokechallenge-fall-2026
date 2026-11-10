"""The four pipelines of v11 (``blend_v11`` of the FDS repository) and their blend.

Each member is an ``EstimatorModel`` of a static scikit-learn pipeline, the original's
``pipelines.estimator``: for a booster, the stacked logit of every column but the team keys
(kept), the stacked logit of the bag columns (replacing them), the team effect, the booster; for
the spline regression, its own cross-fitted spline base, the team effect (which drops the base),
the spline regression. The members are generic: the column roles are parameters (key prefix
``k_``, bag prefixes ``bs_ bh_ bm_``) and a step whose columns are missing is the identity, so on
any table a member is at least "stacked logit + booster"; on the ``v11`` table every step runs.
Parameters from the original's ``configs/pipelines.json`` (merged with the defaults of its
``models.make``), weights from ``configs/blends.json``. ``v11_blend`` is a ``@model``, not an
``@ensemble``: ``build_tuned`` would turn an ensemble into a probability average.
"""

from __future__ import annotations

from typing import Any

from sklearn.pipeline import Pipeline

from interfaces import Model
from models import ko_v11  # noqa: F401  (registers v11's detector model with the members)
from models.core.estimator import EstimatorModel
from models.core.preprocessing import require
from models.core.v11_blend import LogitBlend
from models.core.v11_stacking import SEED, SplineLogit, StackedLogit
from models.core.v11_team import TeamEffect
from registry import model

N_JOBS = 4
KEYS = "k_"
BAGS = ("bs_", "bh_", "bm_")
TEAM: dict[str, Any] = {"a": 1.0, "p2": True, "link": True, "pair": True, "soft": True, "raw": True}
FULL_C = 0.003
BAG_C = 0.01
BASE_C = 0.0007

LR_SPLINE: dict[str, Any] = {
    "C": 0.0007,
    "n_knots": 4,
    "bag_scale": 1.0,
    "keys": [
        "f_d_fainted",
        "f_d_hp_sum",
        "f_p1_fainted",
        "f_p2_fainted",
        "f_p1_hp_sum",
        "f_p2_hp_sum",
        "f_d_strength",
        "t_hd_mean",
        "t_fd_mean",
        "a_ttk_adv",
        "m_win_mean",
        "f_d_alive",
        "a_a1_hp",
        "a_a2_hp",
        "f_d_hp_hard",
        "f_d_fainted_hard",
        "f_p2_unrevealed",
        "m_p1_cover",
        "m_p2_cover",
        "a_dmg12",
        "a_dmg21",
        "a_speed_logratio",
        "t_hd_delta10",
        "f_d_disabled",
        "e_d_dealt",
    ],
}
XGB: dict[str, Any] = {
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
LGBM: dict[str, Any] = {
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
CAT: dict[str, Any] = {
    "iterations": 2883,
    "learning_rate": 0.006388766767796231,
    "depth": 6,
    "l2_leaf_reg": 1.0125444690360867,
    "random_strength": 2.7787738509370583,
    "bagging_temperature": 0.16902526989254424,
    "rsm": 0.551655095551719,
}
WEIGHTS = {
    "v11_lr_spline": 0.21039345977721677,
    "v11_xgb": 0.14756062275115248,
    "v11_lgbm": 0.2747445724394519,
    "v11_cat": 0.3673013450321788,
}


def booster_member(name: str, booster: Any) -> Model:
    """Stacked logits (all but the keys, kept; the bags, replaced), team effect, ``booster``."""
    return EstimatorModel(
        name,
        Pipeline(
            [
                ("stack_full", full_stack(FULL_C)),
                ("stack0", StackedLogit(prefixes=BAGS, C=BAG_C, seed=SEED, name="stack_logit_0")),
                ("team", TeamEffect(**TEAM, seed=SEED, keys=KEYS)),
                ("model", booster),
            ]
        ),
    )


def full_stack(c: float, spline_keys: list[str] | None = None) -> StackedLogit:
    """The stacked logit of every column but the team keys, kept next to them."""
    return StackedLogit(
        exclude=(KEYS,), C=c, seed=SEED, name="stack_logit_full", keep=True, spline_keys=spline_keys
    )


@model("v11_lr_spline")
def v11_lr_spline(name: str = "v11_lr_spline", **params: Any) -> Model:
    """``lr_spline_v11``: spline logistic regression with its own base and the team effect."""
    params = {**LR_SPLINE, **params}
    steps = [
        ("team_base", full_stack(BASE_C, list(params["keys"]))),
        ("team", TeamEffect(**TEAM, seed=SEED, drop_base=True, keys=KEYS)),
        ("model", SplineLogit(**params)),
    ]
    return EstimatorModel(name, Pipeline(steps))


@model("v11_xgb", requires=("xgboost",))
def v11_xgb(name: str = "v11_xgb", **params: Any) -> Model:
    """``xgb_v11``: stacked logits, team effect, XGBoost."""
    booster = require("xgboost").XGBClassifier(
        random_state=SEED, n_jobs=N_JOBS, tree_method="hist", eval_metric="logloss",
        **{**XGB, **params},
    )  # fmt: skip
    return booster_member(name, booster)


@model("v11_lgbm", requires=("lightgbm",))
def v11_lgbm(name: str = "v11_lgbm", **params: Any) -> Model:
    """``lgbm_v11``: stacked logits, team effect, LightGBM."""
    booster = require("lightgbm").LGBMClassifier(
        random_state=SEED, n_jobs=N_JOBS, deterministic=True, force_row_wise=True, verbose=-1,
        **{**LGBM, **params},
    )  # fmt: skip
    return booster_member(name, booster)


@model("v11_cat", requires=("catboost",))
def v11_cat(name: str = "v11_cat", **params: Any) -> Model:
    """``cat_v11``: stacked logits, team effect, CatBoost."""
    booster = require("catboost").CatBoostClassifier(
        random_seed=SEED, thread_count=N_JOBS, verbose=0, allow_writing_files=False,
        **{**CAT, **params},
    )  # fmt: skip
    return booster_member(name, booster)


@model("v11_blend", requires=("xgboost", "lightgbm", "catboost"))
def v11_blend(name: str = "v11_blend") -> Model:
    """``blend_v11``: the four members, logit-averaged with the weights of ``blends.json``."""
    members = [v11_lr_spline(), v11_xgb(), v11_lgbm(), v11_cat()]
    return LogitBlend(name, members, [WEIGHTS[m.name] for m in members])
