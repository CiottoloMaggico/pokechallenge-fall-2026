"""Multi-seed evaluation of the v11 blend against its two parts.

    python tools/v11_blend_eval.py --params runs/tuning/compact_v5_v11__blend_v11.json \\
        --seeds 33 66 133 --jobs 3

Use seeds that were not used to choose w (``tools/v11_blend_weight.py``),
otherwise the blend is evaluated on the folds that picked its weight. The
base model and its tuned files are read from ``--params`` (override with
``--base`` / ``--tuned``); ``--w`` overrides the weight.

Prints, for the base alone (w = 1), v11 alone (w = 0) and the blend:
accuracy and log loss (mean over seed x fold, fold sd, seed sd), the
accuracy of every seed, and the paired differences blend - base and
blend - v11 with the corrected repeated k-fold t-test. Writes a CSV with the
per-fold numbers next to the cache (``eval_<seeds>.csv``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import v11_common as C


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    C.add_common_arguments(parser)
    parser.add_argument("--params", type=Path, help="file written by tools/v11_blend_weight.py")
    parser.add_argument("--w", type=float, help="blend weight (default: from --params)")
    args = parser.parse_args()
    params = json.loads(args.params.read_text())["params"] if args.params else {}
    w = args.w if args.w is not None else params.get("w")
    if w is None:
        raise SystemExit("give --params or --w")
    if params and not args.tuned:
        args.tuned = [Path(p) for p in params.get("tuned", [])]
    if params and args.base == "ensemble_from_lr":
        args.base = params.get("base", args.base)
    seeds = C.seeds_from(args)
    print(f"seeds {seeds}; base {args.base}; w = {w:.2f}", flush=True)
    rows = C.fold_predictions(args, seeds)

    models = {"base alone": 1.0, "v11 alone": 0.0, f"blend w={w:.2f}": w}
    res = {name: C.scores(rows, wm) for name, wm in models.items()}
    seed_of = np.array([r["seed"] for r in rows])
    print(f"\n{'':<14} n_seeds  accuracy  acc fold sd  acc seed sd  log loss  ll seed sd")
    for name, (a, l) in res.items():
        fold_sd = np.mean([a[seed_of == s].std(ddof=1) for s in seeds])
        seed_sd = C.per_seed(rows, a).std(ddof=1) if len(seeds) > 1 else float("nan")
        ll_sd = C.per_seed(rows, l).std(ddof=1) if len(seeds) > 1 else float("nan")
        print(f"  {name:<14} {len(seeds):>5}    {a.mean():.4f}     {fold_sd:.4f}      {seed_sd:.4f}     "
              f"{l.mean():.4f}    {ll_sd:.4f}")
    print("\naccuracy by seed:")
    print(pd.DataFrame({name: C.per_seed(rows, a) for name, (a, _) in res.items()},
                       index=[f"seed {s}" for s in seeds]).to_string(float_format="{:.4f}".format))
    blend_name = list(models)[-1]
    print("\npaired differences (corrected repeated k-fold t-test):")
    for ref in ("base alone", "v11 alone"):
        ta, pa = C.corrected(res[ref][0], res[blend_name][0])
        tl, pl = C.corrected(res[ref][1], res[blend_name][1])
        print(f"  blend - {ref:<10}: accuracy {(res[blend_name][0] - res[ref][0]).mean():+.4f} (t {ta:+.2f}, p {pa:.3f}), "
              f"log loss {(res[blend_name][1] - res[ref][1]).mean():+.5f} (t {tl:+.2f}, p {pl:.3f})")
    out = args.cache / f"eval_{'_'.join(map(str, seeds))}.csv"
    pd.DataFrame([{"seed": r["seed"], "fold": r["fold"],
                   **{f"acc_{n}": res[n][0][i] for n in res}, **{f"ll_{n}": res[n][1][i] for n in res}}
                  for i, r in enumerate(rows)]).to_csv(out, index=False)
    print(f"\nper-fold numbers in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
