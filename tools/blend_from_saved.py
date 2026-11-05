"""Rebuild the v5 + v11 blend on a new base model without refitting v11.

v11 does not depend on this repository's Pokédex or features, so its test
probabilities are recovered from a saved blend by inverting

    logit(p_blend) = w * logit(p_base) + (1 - w) * logit(p_v11)

and blended again with the new base (``models/logit_blend.py``: same EPS, and
where v11 is NaN, a test battle identical to a training one, p = p_base).

    # 1. in a checkout of the commit whose features match the saved blend:
    cd <worktree> && python <repo>/tools/blend_from_saved.py old-base \\
        --blend-run <repo>/runs/compact_v5_v11__blend_v11-tuned --out old_base.csv
    # 2. in this repository:
    python tools/blend_from_saved.py blend \\
        --blend-run runs/compact_v5_v11__blend_v11-tuned \\
        --old-blend submission_blend_v11_proba.csv --old-base old_base.csv \\
        --new-base results/submissions/base_untuned_step2_proba.csv \\
        --out-dir results/submissions --name submission_blend_step2
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
V11_FEATURE = "v11_probability"
V11_COLUMN = "v11_proba"
PROBA = "p_player_won"


def old_base(blend_run: Path, out: Path) -> None:
    """Predict the test with the base saved inside the blend, using the current directory's code."""
    sys.path.insert(0, str(Path.cwd()))
    import feature_catalog
    from data import Dataset
    from domain.pokedex import Pokedex
    from knockout_detector import detector_from_files

    pokedex = Pokedex.from_json("assets/species.json")
    train, test = Path("input/train.jsonl"), Path("input/test.jsonl")
    detector = detector_from_files(pokedex, feature_catalog.type_chart(), (train,))
    contexts = Dataset.from_jsonl(test, pokedex, repair=detector)
    state = pickle.loads((blend_run / "feature_state.pkl").read_bytes())
    del state[V11_FEATURE]
    features = feature_catalog.build("compact_v5")
    features.set_state(state)
    table = features.to_frame(contexts.contexts)
    columns = json.loads((blend_run / "run.json").read_text())["columns"]
    if list(table.columns) != [c for c in columns if c != V11_COLUMN]:
        raise SystemExit(f"columns differ from the saved run: {list(table.columns)}")
    base = pickle.loads((blend_run / "model.pkl").read_bytes())._base
    frame = pd.DataFrame({"battle_id": contexts.ids, PROBA: base.predict_proba(table)})
    frame.to_csv(out, index=False)
    print(f"{base.name}: {len(frame)} rows written to {out}")


def _test_copies_of_train(blend_run: Path, test: Path) -> dict[int, bool]:
    """battle_id -> battle identical to a training one (v11 gives NaN there)."""
    from domain.records import fingerprint, read_records

    state = pickle.loads((blend_run / "feature_state.pkl").read_bytes())
    keys = state[V11_FEATURE]["_train_keys"]
    return {r["battle_id"]: fingerprint(r) in keys for r in read_records(test)}


def blend(args: argparse.Namespace) -> None:
    """Recover p_v11, blend it with the new base and write the three files."""
    sys.path.insert(0, str(ROOT))
    from models.logit_blend import _logit

    model = pickle.loads((args.blend_run / "model.pkl").read_bytes())
    w = model.w
    tables = [pd.read_csv(p)[["battle_id", PROBA]] for p in (args.old_blend, args.old_base)]
    merged = tables[0].merge(tables[1], on="battle_id", suffixes=("_blend", "_base"))
    new = pd.read_csv(args.new_base)[["battle_id", PROBA]].rename(columns={PROBA: "p_new"})
    merged = merged.merge(new, on="battle_id", validate="one_to_one")
    if not len(merged) == len(tables[0]) == len(new):
        raise SystemExit(f"rows not matched by battle_id: {len(merged)} of {len(tables[0])}")
    z_blend = _logit(merged[f"{PROBA}_blend"])
    z_old = _logit(merged[f"{PROBA}_base"])
    p_v11 = 1 / (1 + np.exp(-(z_blend - w * z_old) / (1 - w)))
    copies = merged["battle_id"].map(_test_copies_of_train(args.blend_run, args.test)).to_numpy()
    if copies.any():
        gap = np.abs(merged[f"{PROBA}_blend"] - merged[f"{PROBA}_base"]).to_numpy()[copies]
        print(f"test copies of train: {copies.sum()}, |p_blend - p_old_base| max {gap.max():.2e}")
    p_v11 = np.where(copies, np.nan, p_v11)
    p_new = merged["p_new"].to_numpy()
    z = w * _logit(p_new) + (1 - w) * _logit(np.where(np.isnan(p_v11), p_new, p_v11))
    p = 1 / (1 + np.exp(-z))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ids, labels = merged["battle_id"], (p > 0.5).astype(int)
    pd.DataFrame({"battle_id": ids, "p_v11": p_v11}).to_csv(
        args.out_dir / "v11_test_proba.csv", index=False
    )
    pd.DataFrame({"battle_id": ids, "player_won": labels}).to_csv(
        args.out_dir / f"{args.name}.csv", index=False
    )
    pd.DataFrame({"battle_id": ids, "player_won": labels, PROBA: p}).to_csv(
        args.out_dir / f"{args.name}_proba.csv", index=False
    )
    print(f"w = {w}, rows {len(p)}, p NaN {int(np.isnan(p).sum())}")


def main() -> int:
    """Parse the arguments and run one command."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    o = sub.add_parser("old-base")
    o.add_argument("--blend-run", type=Path, required=True)
    o.add_argument("--out", type=Path, required=True)
    b = sub.add_parser("blend")
    b.add_argument("--blend-run", type=Path, required=True)
    b.add_argument("--old-blend", type=Path, required=True)
    b.add_argument("--old-base", type=Path, required=True)
    b.add_argument("--new-base", type=Path, required=True)
    b.add_argument("--test", type=Path, default=ROOT / "input" / "test.jsonl")
    b.add_argument("--out-dir", type=Path, required=True)
    b.add_argument("--name", required=True)
    args = parser.parse_args()
    if args.command == "old-base":
        old_base(args.blend_run, args.out)
    else:
        blend(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
