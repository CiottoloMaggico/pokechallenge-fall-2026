"""Cut every battle of a file to its first turns (the input of the accuracy-by-turn experiment).

    python docs/evaluation/truncate.py input/train.jsonl --turn 10 --out train_t10.jsonl

Everything downstream (knockout repair, features, model) then sees only those turns, as if the
battle had been recorded up to there.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def truncate(source: Path, turn: int, out: Path) -> None:
    """Write ``source`` with every ``battle_timeline`` cut to its first ``turn`` entries."""
    out.parent.mkdir(parents=True, exist_ok=True)
    with source.open() as lines, out.open("w") as target:
        for line in lines:
            record = json.loads(line)
            record["battle_timeline"] = record["battle_timeline"][:turn]
            target.write(json.dumps(record, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="cut every battle to its first turns")
    parser.add_argument("source", type=Path)
    parser.add_argument("--turn", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    truncate(args.source, args.turn, args.out)
