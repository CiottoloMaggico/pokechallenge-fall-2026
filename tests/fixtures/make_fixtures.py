"""Rebuild ``battles.jsonl``: 200 battles of ``input/train.jsonl`` covering the cases the
domain tests need (P1 faint, double KO, voluntary switch, phazing, Explosion, Spikes, a
duplicated pair, a species without stats).

    python tests/fixtures/make_fixtures.py [--train input/train.jsonl]

The choice is deterministic: for each case the first battles (by file order) that match.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from collections.abc import Callable
from itertools import pairwise
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SIZE = 200
PER_CASE = 8
SEED = 0
PHAZING = {"roar", "whirlwind"}
SELFDESTRUCT = {"explosion", "selfdestruct"}


def fingerprint(record: dict) -> str:
    """sha1 of the record without id and label (same definition as the pipeline's)."""
    body = {k: v for k, v in record.items() if k not in ("battle_id", "player_won")}
    return hashlib.sha1(json.dumps(body, sort_keys=True).encode()).hexdigest()


def moves(record: dict, side: str) -> list[str]:
    """Names of the moves a side used in the timeline."""
    return [
        t[f"{side}_move_details"]["name"]
        for t in record["battle_timeline"]
        if t.get(f"{side}_move_details")
    ]


def active(record: dict, side: str) -> list[str]:
    """Name of the active Pokémon at each turn."""
    return [t[f"{side}_pokemon_state"]["name"] for t in record["battle_timeline"]]


def p1_faint(record: dict) -> bool:
    return any(t["p1_pokemon_state"]["status"] == "fnt" for t in record["battle_timeline"])


def double_ko(record: dict) -> bool:
    return any(
        t["p1_pokemon_state"]["status"] == "fnt" and t["p2_pokemon_state"]["status"] == "fnt"
        for t in record["battle_timeline"]
    )


def voluntary_switch(record: dict) -> bool:
    """P1 changes active Pokémon although the previous one was not fainted."""
    turns = record["battle_timeline"]
    return any(
        a["p1_pokemon_state"]["name"] != b["p1_pokemon_state"]["name"]
        and a["p1_pokemon_state"]["status"] != "fnt"
        for a, b in pairwise(turns)
    )


def phazing(record: dict) -> bool:
    return bool(PHAZING & set(moves(record, "p1") + moves(record, "p2")))


def explosion(record: dict) -> bool:
    return bool(SELFDESTRUCT & set(moves(record, "p1") + moves(record, "p2")))


def spikes(record: dict) -> bool:
    return "spikes" in moves(record, "p1") + moves(record, "p2")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=ROOT / "input" / "train.jsonl")
    parser.add_argument("--out", type=Path, default=Path(__file__).with_name("battles.jsonl"))
    args = parser.parse_args()

    records = [json.loads(line) for line in args.train.open()]
    with_stats = {p["name"] for r in records for p in (*r["p1_team_details"], r["p2_lead_details"])}
    groups: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(records):
        groups[fingerprint(r)].append(i)

    def unknown_species(record: dict) -> bool:
        return any(name not in with_stats for name in active(record, "p2"))

    cases: dict[str, Callable[[dict], bool]] = {
        "p1_faint": p1_faint,
        "double_ko": double_ko,
        "voluntary_switch": voluntary_switch,
        "phazing": phazing,
        "explosion": explosion,
        "spikes": spikes,
        "unknown_species": unknown_species,
    }
    chosen: dict[int, None] = {}
    for name, predicate in cases.items():
        hits = [i for i, r in enumerate(records) if predicate(r)]
        print(f"{name}: {len(hits)} battles in train")
        for i in hits[:PER_CASE]:
            chosen[i] = None
    pairs = [g for g in groups.values() if len(g) > 1]
    print(f"duplicated groups: {len(pairs)}")
    for i in pairs[0] if pairs else []:
        chosen[i] = None

    rest = [i for i in range(len(records)) if i not in chosen]
    random.Random(SEED).shuffle(rest)
    for i in rest[: SIZE - len(chosen)]:
        chosen[i] = None

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as handle:
        for i in sorted(chosen):
            handle.write(json.dumps(records[i], separators=(",", ":")) + "\n")
    print(f"wrote {len(chosen)} battles to {args.out}")


if __name__ == "__main__":
    main()
