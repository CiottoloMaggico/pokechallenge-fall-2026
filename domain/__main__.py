"""Command line of the domain.

python -m domain build-pokedex input/train.jsonl --out results/cache/species.json
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections.abc import Sequence
from pathlib import Path

from domain.pokedex import PokedexBuilder, species_without_stats
from domain.records import read_records


def build_pokedex(train: Path, out: Path) -> int:
    """Write the Pokédex of ``train`` and list the species left without stats."""
    with train.open("rb") as fh:
        digest = hashlib.file_digest(fh, "sha256").hexdigest()
    pokedex = PokedexBuilder().add_records(read_records(train)).build(digest)
    pokedex.to_json(out)
    print(f"{len(pokedex)} species written to {out} (source sha256 {digest})")
    for name, battles in species_without_stats(read_records(train), pokedex).items():
        print(f"no stats in any record: {name} (in {battles} battle(s))")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Parse the arguments and run one command."""
    parser = argparse.ArgumentParser(prog="python -m domain", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-pokedex", help="build the Pokédex from the training file")
    build.add_argument("train", type=Path)
    build.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    return build_pokedex(args.train, args.out)


if __name__ == "__main__":
    sys.exit(main())
