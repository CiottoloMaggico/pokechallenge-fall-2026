"""v11 regression check: this repository's v11 against the original's outputs, stage by stage.

    python tools/v11_check.py export     # once: the original's cut-level detector probabilities
    python tools/v11_check.py check      # every stage (about 50 minutes, ``cv`` is 35 of them)
    python tools/v11_check.py check --stages domain detector features models

The reference dumps live in ``results/v11_alignment/dump/`` (written by the original FDS
repository's code, run in this ``.env``; never delete them). ``export`` adds the probabilities
of the original's detector refitted on the training battles truncated at each cut of the
forecast (``dump/original/p2_faint_cuts.pkl``), which the ``features`` stage injects.

Stages (``bitwise``: max |diff| 0 with NaN equal to NaN, else the check fails; ``noise``:
reported, never failing):

* ``domain`` (bitwise): ``ObservedTimeline`` and ``SwitchExtractor`` against the original's
  ``Side`` and ``transitions`` on all 15,000 battles;
* ``detector`` (bitwise): the v11 repair addon's probabilities of every test battle against the
  original's detector fitted on all P1 rows;
* ``tables`` (noise): the addon's cross-fitted training values (standard folds, mean over the
  seeds) against the original's ``KFold`` ones, full length and per cut;
* ``features`` (bitwise): ``build("v11")`` on contexts replayed with an injected repair that gives
  the original's detector probabilities, against the original's ``X_train`` / ``X_test``;
* ``models`` (bitwise): members and blend fitted on the original's ``X_train``, test
  probabilities against the original's;
* ``cv`` (bitwise): the original's cross-validation protocol (5 folds x 3 repeats) with this
  repository's members on the original's ``X_train``, out-of-fold predictions;
* ``e2e`` (noise): a submission made with ``--repair v11=…`` against the original's blend
  probabilities and against ``results/v11_ref/sub20_blend_v11.csv``.

A stage whose code or artifact does not exist yet is skipped. The script may change with the
code it checks; its comparisons and tolerances may not.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import pickle
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ALIGN = ROOT / "results" / "v11_alignment"
DUMP = ALIGN / "dump"
ORIGINAL = DUMP / "original"
CUTS_FILE = ORIGINAL / "p2_faint_cuts.pkl"
TRAIN, TEST = ROOT / "input" / "train.jsonl", ROOT / "input" / "test.jsonl"
POKEDEX = ROOT / "results" / "cache" / "species.json"
ARTIFACT = ROOT / "results" / "addons" / "knockouts_v11" / "v1"
PROBA = ROOT / "results" / "submissions" / "submission_v11_blend_proba.csv"
REFERENCE_SUBMISSION = ROOT / "results" / "v11_ref" / "sub20_blend_v11.csv"
MEMBERS = {
    "lr_spline_v11": "v11_lr_spline",
    "xgb_v11": "v11_xgb",
    "lgbm_v11": "v11_lgbm",
    "cat_v11": "v11_cat",
}
STAGES = ("domain", "detector", "tables", "features", "models", "cv", "e2e")
BITWISE = {"domain", "detector", "features", "models", "cv"}
SIDE_FIELDS = ("names", "hp", "status", "effects", "boosts", "moves", "events", "revealed")
SIDE_FIELDS += ("first_index", "last_index", "n")

Record = dict[str, Any]
Probabilities = dict[int, float]
Key = tuple[Any, str]


class Skip(Exception):
    """The code or the artifact a stage needs does not exist yet."""


def original(module: str) -> Any:
    """A module of the original's copy in ``results/v11_alignment/original`` (package ``src``)."""
    os.environ.setdefault("FDS_DATA_DIR", str(ALIGN / "data"))
    if str(ALIGN / "original") not in sys.path:
        sys.path.insert(0, str(ALIGN / "original"))
    importlib.import_module("src.mechanics").set_version("v10")
    return importlib.import_module(f"src.{module}")


def repository(module: str) -> Any:
    """A module of this repository, or Skip when it does not exist yet."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as exc:
        raise Skip(f"no module {exc.name}") from exc


def load(path: Path) -> list[Record]:
    with path.open() as fh:
        return [json.loads(line) for line in fh]


def same(a: Any, b: Any) -> bool:
    """Equal values, NaN equal to NaN, recursively; floats compare by value (numpy or not)."""
    if isinstance(a, float) and isinstance(b, float):
        return (math.isnan(a) and math.isnan(b)) or a == b
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return list(a) == list(b) and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b, strict=True))
    return a == b


def max_diff(a: Any, b: Any) -> tuple[float, int]:
    """(max |a - b|, differing cells); inf when NaN positions differ."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        return math.inf, max(a.size, b.size)
    nan_a, nan_b = np.isnan(a), np.isnan(b)
    if (nan_a != nan_b).any():
        return math.inf, int((nan_a != nan_b).sum())
    d = np.abs(np.where(nan_a, 0.0, a) - np.where(nan_b, 0.0, b))
    return (float(d.max()) if d.size else 0.0), int((d != 0).sum())


def fingerprint(record: Mapping[str, Any]) -> str:
    return repository("domain.records").fingerprint(record)


def truncated(record: Record, cut: int) -> Record:
    return {**record, "battle_timeline": record["battle_timeline"][:cut]}


def key(record: Mapping[str, Any]) -> Key:
    return record["battle_id"], fingerprint(record)


def cut_probabilities() -> dict[int, list[Probabilities]]:
    if not CUTS_FILE.exists():
        raise Skip(f"{CUTS_FILE.name} missing: run `python tools/v11_check.py export`")
    return pickle.loads(CUTS_FILE.read_bytes())


def original_matrices() -> dict[str, Any]:
    """The original's ``X_train``, ``X_test`` (indexed by battle id), ``y`` and ``groups``."""
    return pickle.loads((ORIGINAL / "features_v10.pkl").read_bytes())


def reference_faints() -> dict[str, list[Probabilities]]:
    return pickle.loads((ORIGINAL / "p2_faint.pkl").read_bytes())


def export() -> int:
    """Write the original's detector probabilities of the training battles truncated at each cut
    of the forecast (as its ``truncated_features`` refits them); never overwritten."""
    if CUTS_FILE.exists():
        print(f"{CUTS_FILE} exists: nothing to do")
        return 0
    faint, replay, data = original("faint"), original("replay"), original("data")
    cuts = original("features.forecast").CUTS
    raws, dex = data.load_train(), data.pokedex()
    found: dict[int, list[Probabilities]] = {}
    for cut in cuts:
        battles = replay.parse(raws, dex, cut=cut)
        info = faint.fit_detector(battles, [])
        found[cut] = [b.p2_faint for b in battles]
        print(f"cut {cut}: {info}", flush=True)
    CUTS_FILE.write_bytes(pickle.dumps(found))
    print(f"written {CUTS_FILE}")
    return 0


def switch_row(switch: Any) -> Record:
    """A ``domain.switches.Switch`` as the original's transition row."""
    head = {"i": switch.index, "x": switch.leaver, "recorded": switch.recorded}
    return {**head, "final": switch.final, "label": switch.label, **switch.features}


def species_lookup() -> Callable[[Mapping[str, Any]], Any]:
    pokedex = repository("domain.pokedex")
    dex = pokedex.Pokedex.from_json(POKEDEX)
    return lambda record: dex.lookup(record, pokedex.Missing())


def stage_domain() -> str:
    timeline = repository("domain.timeline")
    switches = repository("domain.switches")
    chart = repository("features.common").type_chart()
    lookup = species_lookup()
    replay, data = original("replay"), original("data")
    used_move = original("features.simulation").used_move
    raws = load(TRAIN) + load(TEST)
    dex = data.Pokedex(load(TRAIN))
    bad: list[Any] = []
    rows = {"p1": 0, "p2": 0}
    values: list[tuple[float, float]] = []
    for raw in raws:
        ob, new = replay.Battle(raw, dex), timeline.ObservedTimeline(raw)
        ok = ob.n == new.n and same(ob.spikes, new.spikes)
        extractor = switches.SwitchExtractor(lookup(raw), chart)
        for s in ("p1", "p2"):
            ok = ok and all(
                same(getattr(ob.sides[s], f), getattr(new.sides[s], f)) for f in SIDE_FIELDS
            )
            ok = ok and all(same(used_move(ob, s, i), new.used_move(s, i)) for i in range(ob.n))
            expected, found = ob.transitions(s), [switch_row(t) for t in extractor.extract(raw, s)]
            ok = ok and same(expected, found)
            rows[s] += len(expected)
            for e, f in zip(expected, found, strict=False):
                values.extend((float(e[c]), float(f.get(c, math.nan))) for c in e if c != "x")
        if not ok:
            bad.append(raw["battle_id"])
    d, n = max_diff([v for v, _ in values], [v for _, v in values])
    print(
        f"domain: {len(raws)} battles; switches P1 {rows['p1']}, P2 {rows['p2']} "
        f"({len(values)} values): max |diff| {d}, differing values {n}, battles {len(bad)} "
        f"{bad[:10]}"
    )
    return "pass" if not bad and n == 0 else "FAIL"


def v11_repair(artifact: Path) -> Any:
    addons = repository("addons")
    repository("addons.knockouts_v11")
    if not (artifact / "info.json").exists():
        raise Skip(f"no artifact in {artifact}")
    pokedex = repository("domain.pokedex").Pokedex.from_json(POKEDEX)
    return addons.load_repair(
        f"v11={artifact}", pokedex, repository("features.common").type_chart()
    )


def stage_detector(artifact: Path) -> str:
    repair = v11_repair(artifact)
    expected = reference_faints()["test"]
    found = [dict(repair.probabilities(r)) for r in load(TEST)]
    keys = all(list(a) == list(b) for a, b in zip(expected, found, strict=True))
    d, n = max_diff(
        [v for p in expected for v in p.values()], [v for p in found for v in p.values()]
    )
    print(
        f"detector: {len(found)} test battles, {sum(map(len, found))} P2 switches, same keys "
        f"{keys}: max |diff| {d}, differing {n}"
    )
    return "pass" if keys and n == 0 else "FAIL"


def noise(
    name: str,
    records: Sequence[Record],
    ours: Sequence[Probabilities],
    theirs: Sequence[Probabilities],
) -> None:
    """Mean |Δp| on the P2 candidates, share across 0.5, expected hidden faints per battle."""
    extractor = repository("domain.switches").SwitchExtractor
    chart, lookup = repository("features.common").type_chart(), species_lookup()
    a: list[float] = []
    b: list[float] = []
    for record, p, q in zip(records, ours, theirs, strict=True):
        for switch in extractor(lookup(record), chart).candidates(record):
            a.append(p[switch.index])
            b.append(q[switch.index])
    x, y = np.array(a), np.array(b)
    print(
        f"tables {name}: {len(x)} P2 candidates, mean |dp| {np.abs(x - y).mean():.4f}, "
        f"across 0.5 {np.mean((x >= 0.5) != (y >= 0.5)):.4f}, hidden faints per battle "
        f"{x.sum() / len(records):.4f} (original {y.sum() / len(records):.4f})"
    )


def stage_tables(artifact: Path) -> str:
    repair = v11_repair(artifact)
    records = [{k: v for k, v in r.items() if k != "player_won"} for r in load(TRAIN)]
    ours = [dict(repair.probabilities(r)) for r in records]
    noise("full length", records, ours, reference_faints()["train"])
    for cut, theirs in cut_probabilities().items():
        cut_records = [truncated(r, cut) for r in records]
        ours = [dict(repair.probabilities(r)) for r in cut_records]
        noise(f"cut {cut}", cut_records, ours, theirs)
    return "reported"


class InjectedRepair:
    """``TimelineRepair`` giving the original's detector probabilities (full and cut level)."""

    def __init__(self) -> None:
        faints = reference_faints()
        self._table: dict[Key, Probabilities] = {}
        for path, part in ((TRAIN, "train"), (TEST, "test")):
            self._table.update(zip(map(key, load(path)), faints[part], strict=True))
        for cut, values in cut_probabilities().items():
            cut_keys = (key(truncated(r, cut)) for r in load(TRAIN))
            self._table.update(zip(cut_keys, values, strict=True))

    def probabilities(self, record: Mapping[str, Any]) -> Probabilities:
        return dict(self._table[key(record)])

    def repairs(self, record: Mapping[str, Any]) -> dict[int, str]:
        timeline = record["battle_timeline"]
        found = self.probabilities(record).items()
        return {i: timeline[i - 1]["p2_pokemon_state"]["name"] for i, p in found if p >= 0.5}

    def describe(self) -> dict[str, Any]:
        return {"name": "injected", "version": "original", "params": {}}


def compare_frames(name: str, expected: pd.DataFrame, found: pd.DataFrame) -> bool:
    columns = list(expected.columns) == list(found.columns)
    d, n = max_diff(expected.to_numpy(), found.to_numpy()) if columns else (math.inf, -1)
    print(
        f"{name}: {found.shape[0]} rows x {found.shape[1]} columns, same columns {columns}, "
        f"max |diff| {d}, differing cells {n}"
    )
    return columns and n == 0


def stage_features() -> str:
    data, sets = repository("data"), repository("features.sets")
    pokedex = repository("domain.pokedex").Pokedex.from_json(POKEDEX)
    records = load(TRAIN)
    repair = InjectedRepair()
    started = time.perf_counter()
    train = data.Dataset.from_jsonl(TRAIN, pokedex, repair=repair)
    visible = [{k: v for k, v in r.items() if k != "player_won"} for r in records]
    order = [c.record for c in train.contexts] == visible
    print(f"features: training contexts in file order, label removed: {order}")
    test = data.Dataset.from_jsonl(TEST, pokedex, repair=repair)
    dumped = original_matrices()
    cols = json.loads((ORIGINAL / "columns_lr_spline_v11.json").read_text())
    features = sets.build("v11")
    x_train = features.fit_transform(
        train.contexts, train.labels.tolist(), (), train.duplicate_groups().tolist()
    )
    print(f"  fit_transform {time.perf_counter() - started:.0f} s", flush=True)
    ok = compare_frames("X_train", dumped["X_train"][cols], x_train)
    ok = compare_frames("X_test", dumped["X_test"][cols], features.to_frame(test.contexts)) and ok
    return "pass" if ok and order else "FAIL"


def dumped_matrix() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    dumped = original_matrices()
    cols = json.loads((ORIGINAL / "columns_lr_spline_v11.json").read_text())
    groups = dumped["groups"].to_numpy()
    return dumped["X_train"][cols], dumped["X_test"][cols], dumped["y"].to_numpy(), groups, cols


def stage_models() -> str:
    build = repository("models").build
    x, x_test, y, _, _ = dumped_matrix()
    ok = True
    for name, ours in [*MEMBERS.items(), ("blend_v11", "v11_blend")]:
        started = time.perf_counter()
        found = build(ours).fit(x, y).predict_proba(x_test)
        file = "blend_v11.npy" if name == "blend_v11" else f"member_{name}.npy"
        d, n = max_diff(np.load(ORIGINAL / file), found)
        print(
            f"models {ours} vs {name}: {len(found)} test rows: max |diff| {d}, differing {n} "
            f"({time.perf_counter() - started:.0f} s)",
            flush=True,
        )
        ok = ok and n == 0
    return "pass" if ok else "FAIL"


class Adapter:
    """A repository model with the scikit-learn interface the original's ``cv.run`` expects."""

    def __init__(self, model: Any) -> None:
        self.model = model

    def fit(self, x: pd.DataFrame, y: np.ndarray) -> Adapter:
        self.model.fit(x, y)
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        p = self.model.predict_proba(x)
        return np.column_stack([1 - p, p])


def stage_cv() -> str:
    build = repository("models").build
    cv, ensemble = original("cv"), original("ensemble")
    x, _, y, groups, _ = dumped_matrix()
    folds = cv.make_folds(y, groups)
    ok, oofs = True, {}
    for name, ours in MEMBERS.items():
        started = time.perf_counter()
        _, oofs[name] = cv.run(lambda ours=ours: Adapter(build(ours)), x, y, folds, log=False)
        ok = report_oof(name, oofs[name], started) and ok
    weights = json.loads((ALIGN / "original/configs/blends.json").read_text())["blend_v11"]
    stacked = np.stack([oofs[m] for m in weights["members"]])
    w = np.array(list(weights["members"].values()))
    blend = np.stack([ensemble.blend(stacked[:, r], w) for r in range(stacked.shape[1])])
    return "pass" if report_oof("blend_v11", blend, time.perf_counter()) and ok else "FAIL"


def report_oof(name: str, found: np.ndarray, started: float) -> bool:
    d, n = max_diff(np.load(DUMP / "cv" / f"original_{name}.npy"), found)
    print(
        f"cv {name}: out of fold {found.shape}: max |diff| {d}, differing {n} "
        f"({time.perf_counter() - started:.0f} s)",
        flush=True,
    )
    return n == 0


def stage_e2e(proba: Path) -> str:
    if not proba.exists():
        raise Skip(f"no probabilities in {proba}")
    found = pd.read_csv(proba)
    ids = original_matrices()["X_test"].index.to_numpy()
    assert (found["battle_id"].to_numpy() == ids).all(), "test order"
    p, ref = found["p_player_won"].to_numpy(), np.load(ORIGINAL / "blend_v11.npy")
    d, _ = max_diff(ref, p)
    labels = (p > 0.5).astype(int)
    submitted = pd.read_csv(REFERENCE_SUBMISSION).set_index("battle_id")["player_won"]
    print(
        f"e2e {proba.name}: {len(p)} test battles, max |diff| {d:.4f}, mean |diff| "
        f"{np.abs(ref - p).mean():.4f}, labels apart from the dump "
        f"{int((labels != (ref > 0.5)).sum())}, from sub20_blend_v11.csv "
        f"{int((labels != submitted.reindex(found['battle_id']).to_numpy()).sum())}"
    )
    return "reported"


def check(stages: Iterable[str], artifact: Path, proba: Path) -> int:
    runs: dict[str, Callable[[], str]] = {
        "domain": stage_domain,
        "detector": lambda: stage_detector(artifact),
        "tables": lambda: stage_tables(artifact),
        "features": stage_features,
        "models": stage_models,
        "cv": stage_cv,
        "e2e": lambda: stage_e2e(proba),
    }
    results: dict[str, str] = {}
    for stage in stages:
        started = time.perf_counter()
        try:
            results[stage] = runs[stage]()
        except Skip as reason:
            results[stage] = f"skipped ({reason})"
        print(f"[{stage}] {results[stage]} ({time.perf_counter() - started:.0f} s)\n", flush=True)
    failed = [s for s, r in results.items() if r == "FAIL"]
    for stage, result in results.items():
        print(f"{stage:9s} {'bitwise' if stage in BITWISE else 'noise':8s} {result}")
    print("V11 CHECK FAILED: " + ", ".join(failed) if failed else "v11 check passed")
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("export", help="the original's cut-level detector probabilities (once)")
    c = sub.add_parser("check", help="compare the stages with the dumps")
    c.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    c.add_argument("--artifact", type=Path, default=ARTIFACT, help="v11 detector artifact")
    c.add_argument("--proba", type=Path, default=PROBA, help="probabilities of a submission")
    args = parser.parse_args()
    if args.cmd == "export":
        return export()
    return check(args.stages, args.artifact, args.proba)


if __name__ == "__main__":
    raise SystemExit(main())
