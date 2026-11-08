"""Command line of the domain.

python -m domain build-pokedex input/train.jsonl --out results/cache/species.json
python -m domain replay input/train.jsonl --battle 42
python -m domain check input/train.jsonl input/test.jsonl

Only ``train`` builds a Pokédex: ``check`` builds it from its first file and
``replay`` loads the one written by ``build-pokedex``. Replays are not repaired:
the repairs live outside the domain (``python main.py replay --repair ...``).
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections.abc import Sequence
from pathlib import Path

from domain.battle import BattleReplay, SideState
from domain.errors import DomainError, ReplayError
from domain.pokedex import Missing, Pokedex, PokedexBuilder, species_without_stats
from domain.records import read_records
from domain.repair import NO_REPAIR, TimelineRepair

DEFAULT_POKEDEX = Path("results/cache/species.json")


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


def replay(
    path: Path, battle_id: int, pokedex_path: Path, repair: TimelineRepair = NO_REPAIR
) -> int:
    """Print the state of battle ``battle_id`` at the end of every turn, after ``repair``."""
    record = next((r for r in read_records(path) if r["battle_id"] == battle_id), None)
    if record is None:
        raise ReplayError(f"battle {battle_id} is not in {path}")
    pokedex = Pokedex.from_json(pokedex_path)
    context = BattleReplay(pokedex, repair, Missing()).replay(record)
    for state in context.history:
        print(f"turn {state.turn:>2}  p1 {_side(state.p1)}  |  p2 {_side(state.p2)}")
    return 0


def _side(side: SideState) -> str:
    move = side.move.name if side.move is not None else "-"
    return (
        f"{side.active} {side.hp_pct:.0%} {side.status} move={move} "
        f"alive={side.n_alive} hp_total={side.hp_total:.2f}"
    )


def check(train: Path, others: Sequence[Path]) -> int:
    """Validate every row and list the species without stats in each file.

    Those species are expected (the replay gives them NaN), so they do not
    fail the check; an invalid row stops it with exit code 2.
    """
    pokedex = PokedexBuilder().add_records(read_records(train)).build()
    print(f"pokedex of {train}: {len(pokedex)} species")
    for path in (train, *others):
        battles = sum(1 for _ in read_records(path))
        missing = species_without_stats(read_records(path), pokedex)
        print(f"{path}: {battles} valid battles, species without stats: {missing or 'none'}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Parse the arguments and run one command; a domain error gives exit code 2."""
    parser = argparse.ArgumentParser(prog="python -m domain", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-pokedex", help="build the Pokédex from the training file")
    build.add_argument("train", type=Path)
    build.add_argument("--out", type=Path, required=True)
    show = commands.add_parser("replay", help="print the states of one battle")
    show.add_argument("path", type=Path)
    show.add_argument("--battle", type=int, required=True)
    show.add_argument("--pokedex", type=Path, default=DEFAULT_POKEDEX)
    validate = commands.add_parser("check", help="invalid rows and species without stats")
    validate.add_argument("train", type=Path)
    validate.add_argument("others", type=Path, nargs="*")
    args = parser.parse_args(argv)
    try:
        if args.command == "build-pokedex":
            return build_pokedex(args.train, args.out)
        if args.command == "replay":
            return replay(args.path, args.battle, args.pokedex)
        return check(args.train, args.others)
    except DomainError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
