"""Paired check of the Monte Carlo simulation features (prototype).

Compares compact_v5's columns with and without the ``mc_*`` features, with
logistic regression, on the same stratified grouped folds of a random
subsample of the training battles. Only training data is used.

Run from the project root::

    python -m experiments.mc_eval --battles 2000 --runs 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import feature_catalog
from data import Dataset
from domain.pokedex import Pokedex
from features.montecarlo_features import MC_OUTPUTS, MonteCarloSimulation
from gym import prepare_folds
from knockout_detector import detector_from_files
from selection import RepeatedGroupFolds, Selector, paired_t

COMPACT_V5 = (
    "material_hp",
    "material_healthy",
    "spikes_on_field",
    "bst_sum",
    "hpw_spd",
    "eff_substitute",
    "eff_trapped",
    "hp_healed",
    "act_duel",
    "snorlax_diff",
    "mv_res",
    "p_switch_diff",
    "switch_excess_all",
    "hidden_bst",
    "te_p1_team",
    "te_pair_mean",
    "sim_seen_left",
    "p2_hidden_ko_unsure",
    "tm_team",
    "bag_logit",
    "act_ttk_adv",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--train", type=Path, default=Path("input/train.jsonl"))
    parser.add_argument("--pokedex", type=Path, default=Path("assets/species.json"))
    parser.add_argument("--battles", type=int, default=2000)
    parser.add_argument("--runs", type=int, default=8)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    pokedex = Pokedex.from_json(args.pokedex)
    chart = feature_catalog.type_chart()
    detector = detector_from_files(pokedex, chart, (args.train,))
    full = Dataset.from_jsonl(args.train, pokedex, repair=detector)
    rng = np.random.default_rng(args.seed)
    picked = np.sort(rng.choice(len(full), size=args.battles, replace=False))
    data = full.subset(list(picked))

    pool = feature_catalog.build("selection_pool_v5")
    pool.register(MonteCarloSimulation(chart, runs=args.runs))
    folds = prepare_folds(
        pool,
        data.contexts,
        data.labels,
        RepeatedGroupFolds(args.folds, [args.seed]),
        data.duplicate_groups(),
        progress=None,
    )
    selector = Selector(folds, ["logreg"])
    base_acc, base_loss = selector.score(list(COMPACT_V5))
    mc_acc, mc_loss = selector.score([*COMPACT_V5, *MC_OUTPUTS])
    base = (base_acc[0], base_loss[0])  # the single model's rows
    with_mc = (mc_acc[0], mc_loss[0])

    d_acc = with_mc[0] - base[0]
    d_loss = with_mc[1] - base[1]
    t = paired_t(d_loss)
    print(f"battles {args.battles}, runs {args.runs}, folds {len(folds)}")
    print(
        f"accuracy  base {base[0].mean():.4f}  +mc {with_mc[0].mean():.4f}  "
        f"diff {d_acc.mean() * 100:+.2f} points"
    )
    print(
        f"log-loss  base {base[1].mean():.4f}  +mc {with_mc[1].mean():.4f}  "
        f"diff {d_loss.mean():+.5f}  t {t:+.2f}"
    )
    print("per fold accuracy diff:", np.round(d_acc * 100, 2).tolist())
    return 0


if __name__ == "__main__":
    sys.exit(main())
