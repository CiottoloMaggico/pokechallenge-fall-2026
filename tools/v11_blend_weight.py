"""Blend weight of compact_v5 (base model) and v11, chosen in cross-validation.

    python tools/v11_blend_weight.py --tuned runs/tuning/compact_v5__*.json \\
        --seeds 42 43 44 --jobs 3

For every seed and fold v11 is trained on the training part only and
predicts the validation part (``features/v11_bridge.py``), and the base model
(default ensemble_from_lr, tuned with ``--tuned``) is fitted on the
compact_v5 columns of the same training part. Then:

* w in 0, 0.05, ..., 1 (p = expit(w logit p_base + (1 - w) logit p_v11)):
  mean accuracy and log loss over every seed x fold;
* the chosen w minimises the log loss (accuracy is too noisy to pick it);
* nested check: within each seed every fold gets the w chosen on the other
  four folds, so the reported blend score does not reuse its own fold;
  compared with the base alone (w = 1) and v11 alone (w = 0), paired on the
  same folds with the corrected repeated k-fold t-test.

Writes ``--out`` (default runs/tuning/compact_v5_v11__blend_v11.json), a
parameter file for ``main.py``::

    python main.py train --features compact_v5_v11 --models blend_v11 \\
        --params runs/tuning/compact_v5_v11__blend_v11.json
    python main.py predict --run runs/compact_v5_v11__blend_v11-tuned --out submission.csv

The fold tables are cached in ``--cache``: rerunning with other seeds only
builds the new ones; ``tools/v11_blend_eval.py`` reuses them.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import v11_common as C

GRID = np.round(np.arange(0, 1.0001, 0.05), 2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    C.add_common_arguments(parser)
    parser.add_argument("--out", type=Path, default=C.ROOT / "runs" / "tuning" / "compact_v5_v11__blend_v11.json")
    args = parser.parse_args()
    seeds = C.seeds_from(args)
    print(f"seeds {seeds}; base {args.base}; tuned {[str(p) for p in args.tuned]}", flush=True)
    rows = C.fold_predictions(args, seeds)

    table = {w: C.scores(rows, w) for w in GRID}
    print("\n   w    accuracy  log loss   (w = 1: base alone, w = 0: v11 alone)")
    for w in GRID:
        a, l = table[w]
        print(f"  {w:.2f}   {a.mean():.4f}    {l.mean():.4f}")
    w_best = float(min(GRID, key=lambda w: table[w][1].mean()))

    nested = {}
    for s in seeds:
        mine = [r for r in rows if r["seed"] == s]
        for r in mine:
            others = [o for o in mine if o["fold"] != r["fold"]]
            nested[(s, r["fold"])] = float(min(GRID, key=lambda w: C.scores(others, w)[1].mean()))
    acc_n, ll_n = C.scores(rows, nested)
    acc_b, ll_b = table[1.0]
    acc_v, ll_v = table[0.0]

    print(f"\nchosen w = {w_best:.2f} (min log loss over {len(rows)} folds); "
          f"nested choices: {sorted(set(nested.values()))}")
    print(f"{'':<22} accuracy (seed sd)      log loss")
    for label, a, l in (("base alone (w = 1)", acc_b, ll_b), ("v11 alone (w = 0)", acc_v, ll_v),
                        ("blend, nested w", acc_n, ll_n)):
        print(f"  {label:<20} {a.mean():.4f} ({C.per_seed(rows, a).std(ddof=1) if len(seeds) > 1 else 0:.4f})"
              f"     {l.mean():.4f}")
    result = {}
    for label, a, l in (("vs_base", acc_b, ll_b), ("vs_v11", acc_v, ll_v)):
        ta, pa = C.corrected(a, acc_n)
        tl, pl = C.corrected(l, ll_n)
        result[label] = {"d_acc": float((acc_n - a).mean()), "p_acc": pa, "d_ll": float((ll_n - l).mean()), "p_ll": pl}
        print(f"  nested blend {label.replace('_', ' ')}: accuracy {result[label]['d_acc']:+.4f} (t {ta:+.2f}, p {pa:.3f}), "
              f"log loss {result[label]['d_ll']:+.5f} (t {tl:+.2f}, p {pl:.3f})")

    info = {
        "model": "blend_v11",
        "feature_set": C.FEATURE_SET,
        "params": {"w": w_best, "base": args.base, "tuned": [str(p) for p in args.tuned]},
        "seeds": seeds,
        "grid": {f"{w:.2f}": {"accuracy": float(table[w][0].mean()), "log_loss": float(table[w][1].mean())}
                 for w in GRID},
        "nested": {"accuracy": float(acc_n.mean()), "log_loss": float(ll_n.mean()), **result},
        "saved": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(info, indent=2))
    print(f"\nparameters written to {args.out}")
    print(f"next: python main.py train --features {C.FEATURE_SET} --models blend_v11 --params {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
