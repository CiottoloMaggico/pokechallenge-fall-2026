"""The four pipelines of v11 (``blend_v11`` of the FDS repository) and their blend.

Parameters copied from the original's ``configs/pipelines.json`` (merged with the defaults of
its ``models.make``), weights from ``configs/blends.json``. Meant for the ``v11`` feature set;
``v11_blend`` is a ``@model``, not an ``@ensemble``: ``build_tuned`` would turn an ensemble into
a probability average.
"""

from __future__ import annotations

from typing import Any

from interfaces import Model
from models import ko_v11  # noqa: F401  (registers v11's detector model with the members)
from models.core.v11_member import V11Blend, V11Member
from registry import model

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


@model("v11_lr_spline")
def v11_lr_spline(name: str = "v11_lr_spline", **params: Any) -> Model:
    """``lr_spline_v11``: spline logistic regression with its own base and the team effect."""
    return V11Member(name, "lr_spline", {**LR_SPLINE, **params})


@model("v11_xgb", requires=("xgboost",))
def v11_xgb(name: str = "v11_xgb", **params: Any) -> Model:
    """``xgb_v11``: stacked logits, team effect, XGBoost."""
    return V11Member(name, "xgb", {**XGB, **params})


@model("v11_lgbm", requires=("lightgbm",))
def v11_lgbm(name: str = "v11_lgbm", **params: Any) -> Model:
    """``lgbm_v11``: stacked logits, team effect, LightGBM."""
    return V11Member(name, "lgbm", {**LGBM, **params})


@model("v11_cat", requires=("catboost",))
def v11_cat(name: str = "v11_cat", **params: Any) -> Model:
    """``cat_v11``: stacked logits, team effect, CatBoost."""
    return V11Member(name, "cat", {**CAT, **params})


@model("v11_blend", requires=("xgboost", "lightgbm", "catboost"))
def v11_blend(name: str = "v11_blend") -> Model:
    """``blend_v11``: the four members, logit-averaged with the weights of ``blends.json``."""
    members = [
        V11Member("v11_lr_spline", "lr_spline", LR_SPLINE),
        V11Member("v11_xgb", "xgb", XGB),
        V11Member("v11_lgbm", "lgbm", LGBM),
        V11Member("v11_cat", "cat", CAT),
    ]
    return V11Blend(name, members, [WEIGHTS[m.name] for m in members])
