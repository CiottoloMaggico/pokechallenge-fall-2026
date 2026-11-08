"""One-off: move the outputs of the old layout to ``results/`` (REFACTORING.md section 2).

    python tools/move_old_outputs.py            # prints the moves, changes nothing
    python tools/move_old_outputs.py --apply

    runs/tuning/*            -> results/tuning/
    runs/optuna.db           -> results/tuning/optuna.db
    runs/<anything else>     -> results/runs/<same name>
    results/<evaluation>/    -> results/evaluations/<same name>

``runs/folds`` goes to ``results/runs/folds``, not to ``results/cache/folds``, on purpose: its
tables were built before the Pokédex of step 2 and ``tune`` would read them without a warning.
``results/reference_runs``, ``golden``, ``cache``, ``submissions`` and ``v11_*`` stay where they
are. Nothing is overwritten: if a target exists the script stops before moving anything.
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def planned_moves(root: Path) -> list[tuple[Path, Path]]:
    """``(source, target)`` of everything still in the old layout."""
    runs, results = root / "runs", root / "results"
    moves: list[tuple[Path, Path]] = []
    if runs.is_dir():
        for path in sorted(runs.iterdir()):
            if path.name == "tuning":
                moves += [(p, results / "tuning" / p.name) for p in sorted(path.iterdir())]
            elif path.name == "optuna.db":
                moves.append((path, results / "tuning" / path.name))
            else:
                moves.append((path, results / "runs" / path.name))
    if results.is_dir():
        for path in sorted(results.iterdir()):
            if (path / "folds.csv").exists():
                moves.append((path, results / "evaluations" / path.name))
    return moves


def main() -> int:
    parser = argparse.ArgumentParser(description="Move old outputs to results/")
    parser.add_argument("--apply", action="store_true", help="move (default: only print)")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    moves = planned_moves(args.root)
    taken = [target for _, target in moves if target.exists()]
    if taken:
        raise SystemExit(f"targets already exist, nothing moved: {taken}")
    for source, target in moves:
        print(f"{source.relative_to(args.root)} -> {target.relative_to(args.root)}")
        if args.apply:
            target.parent.mkdir(parents=True, exist_ok=True)
            source.rename(target)
    if args.apply:
        for empty in (args.root / "runs" / "tuning", args.root / "runs"):
            if empty.is_dir() and not any(empty.iterdir()):
                empty.rmdir()
    print(f"{len(moves)} moves {'done' if args.apply else 'planned (use --apply)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
