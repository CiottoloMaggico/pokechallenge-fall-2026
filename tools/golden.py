"""Golden files: freeze what the pipeline computes, to check that a refactoring
step changes nothing.

    python tools/golden.py export --out results/golden/reference     # before the step
    python tools/golden.py export --out results/golden/candidate     # after the step
    python tools/golden.py check results/golden/reference results/golden/candidate

``export`` writes, for the training file:

* ``species.csv``: the Pokédex used by the pipeline;
* ``repairs_p2.csv``: every P2 transition scored by the knockout detector
  (battle_id, row index, probability, repaired);
* ``features_f<k>_{train,valid}.csv``: the compact_v5 table of each fold of
  seed 33 (StratifiedGroupKFold 5, duplicates grouped), with battle ids;
* ``oof_logreg.csv``: out-of-fold probability of logreg on those folds;
* ``metrics.json``: accuracy and log loss of logreg on seed 33.

``check`` compares two exports file by file on their keys and fails (exit 1)
on any missing row, column or value that differs by more than ``--tol``.

Only ``_export`` touches the codebase: after a refactoring step adapt its
imports to the new modules (the outputs and their format must not change).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEED = 33
FEATURE_SET = "compact_v5"
FLOAT = "%.17g"


def _export(out: Path, train: Path) -> None:
    # adapt these imports to the refactored modules; keep the outputs identical
    sys.path.insert(0, str(ROOT))
    from sklearn.metrics import log_loss
    from sklearn.model_selection import StratifiedGroupKFold

    import feature_catalog
    import model_catalog
    from data import Dataset
    from domain.pokedex import PokedexBuilder
    from domain.records import read_records
    from gym import make_splits, prepare_folds
    from knockout_detector import detector_from_files

    started = time.perf_counter()
    out.mkdir(parents=True, exist_ok=True)
    pokedex = PokedexBuilder().add_records(read_records(train)).build()
    rows = []
    for species in pokedex:
        rows.append({"name": species.name, "types": "/".join(species.types),
                     **{f"base_{s}": getattr(species, f"base_{s}")
                        for s in ("hp", "atk", "def", "spa", "spd", "spe")}})
    pd.DataFrame(rows).sort_values("name").to_csv(out / "species.csv", index=False)

    detector = detector_from_files(pokedex, feature_catalog.type_chart(), (train,))
    records = [json.loads(line) for line in train.open()]
    rows = []
    for record in records:
        for index, p in sorted(detector.probabilities(record["battle_timeline"]).items()):
            rows.append({"battle_id": record["battle_id"], "index": index, "p": p,
                         "repaired": int(p >= detector.threshold)})
    pd.DataFrame(rows).to_csv(out / "repairs_p2.csv", index=False, float_format=FLOAT)
    print(f"species and P2 repairs exported ({time.perf_counter() - started:.0f} s)", flush=True)

    dataset = Dataset.from_jsonl(train, pokedex, repair=detector)
    ids, y, groups = np.asarray(dataset.ids), dataset.labels, dataset.duplicate_groups()
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    splits = make_splits(splitter, y, groups)
    folds = prepare_folds(feature_catalog.build(FEATURE_SET), dataset.contexts, y, splitter,
                          groups=groups)
    oof = []
    for k, ((tr, va), (x_train, y_train, x_valid, _)) in enumerate(zip(splits, folds, strict=True)):
        x_train.assign(battle_id=ids[tr]).to_csv(out / f"features_f{k}_train.csv", index=False,
                                                 float_format=FLOAT)
        x_valid.assign(battle_id=ids[va]).to_csv(out / f"features_f{k}_valid.csv", index=False,
                                                 float_format=FLOAT)
        model = model_catalog.build("logreg").fit(x_train, y_train)
        oof.append(pd.DataFrame({"battle_id": ids[va], "fold": k, "y": y[va],
                                 "p": model.predict_proba(x_valid)}))
        print(f"fold {k + 1}/5 exported ({time.perf_counter() - started:.0f} s)", flush=True)
    oof_frame = pd.concat(oof, ignore_index=True)
    oof_frame.to_csv(out / "oof_logreg.csv", index=False, float_format=FLOAT)
    per_fold = oof_frame.groupby("fold")
    metrics = {
        "seed": SEED, "feature_set": FEATURE_SET, "model": "logreg",
        "accuracy": float(per_fold.apply(lambda f: np.mean((f.p > 0.5) == f.y)).mean()),
        "log_loss": float(per_fold.apply(lambda f: log_loss(f.y, f.p)).mean()),
        "n_repaired_p2": int(pd.read_csv(out / "repairs_p2.csv")["repaired"].sum()),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


KEYS = {
    "species.csv": ["name"],
    "repairs_p2.csv": ["battle_id", "index"],
    "oof_logreg.csv": ["battle_id"],
}


def _compare(name: str, ref: pd.DataFrame, new: pd.DataFrame, tol: float) -> list[str]:
    keys = KEYS.get(name, ["battle_id"])
    problems = []
    if list(ref.columns) != list(new.columns):
        problems.append(f"{name}: columns differ\n  ref {list(ref.columns)}\n  new {list(new.columns)}")
        common = [c for c in ref.columns if c in new.columns]
        ref, new = ref[common], new[common]
    if ref[keys].duplicated().any():
        # duplicated battles (same id twice cannot happen; keep the order as key)
        ref, new = ref.assign(_row=range(len(ref))), new.assign(_row=range(len(new)))
        keys = [*keys, "_row"]
    merged = ref.merge(new, on=keys, how="outer", suffixes=("_ref", "_new"), indicator=True)
    missing = merged[merged["_merge"] != "both"]
    if len(missing):
        problems.append(f"{name}: {len(missing)} rows only on one side, e.g.\n"
                        f"{missing.head(5)[keys + ['_merge']].to_string(index=False)}")
    both = merged[merged["_merge"] == "both"]
    for col in [c for c in ref.columns if c not in keys]:
        a, b = both[f"{col}_ref"], both[f"{col}_new"]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            bad = ~(np.isclose(a, b, rtol=0, atol=tol, equal_nan=True))
        else:
            bad = a.astype(str) != b.astype(str)
        if bad.any():
            ex = both.loc[bad, keys + [f"{col}_ref", f"{col}_new"]].head(5)
            problems.append(f"{name}: column {col!r} differs on {int(bad.sum())} rows, e.g.\n"
                            f"{ex.to_string(index=False)}")
    return problems


def _check(reference: Path, candidate: Path, tol: float) -> int:
    problems = []
    for path in sorted(reference.glob("*.csv")):
        other = candidate / path.name
        if not other.exists():
            problems.append(f"{path.name}: missing in {candidate}")
            continue
        problems += _compare(path.name, pd.read_csv(path), pd.read_csv(other), tol)
    ref_m = json.loads((reference / "metrics.json").read_text())
    new_m = json.loads((candidate / "metrics.json").read_text())
    for key, value in ref_m.items():
        if isinstance(value, float) and abs(value - new_m.get(key, np.nan)) > tol:
            problems.append(f"metrics.json: {key} {value} -> {new_m.get(key)}")
        elif not isinstance(value, float) and value != new_m.get(key):
            problems.append(f"metrics.json: {key} {value} -> {new_m.get(key)}")
    if problems:
        print("GOLDEN CHECK FAILED\n\n" + "\n\n".join(problems))
        return 1
    print(f"golden check passed: {candidate} matches {reference} (tol {tol})")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--train", type=Path, default=ROOT / "input" / "train.jsonl")
    c = sub.add_parser("check")
    c.add_argument("reference", type=Path)
    c.add_argument("candidate", type=Path)
    c.add_argument("--tol", type=float, default=1e-9)
    args = parser.parse_args()
    if args.cmd == "export":
        _export(args.out, args.train)
        return 0
    return _check(args.reference, args.candidate, args.tol)


if __name__ == "__main__":
    raise SystemExit(main())
