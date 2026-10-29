"""Build the Pokédex JSON from the training data plus manual entries.

Usage::

    python build_pokedex.py input/train.jsonl --manual assets/species_manual.json \\
        --out assets/species.json --check input/test.jsonl

1. Every species with data in the training records is extracted (the
   values come from the dataset itself, nothing is typed by hand).
2. The manual file adds the species that never appear with data (for
   example P2 Pokémon revealed after the lead). A manual entry for a
   species already extracted must match it exactly.
3. ``--check`` files are replayed against the result: every species they
   mention must be in the Pokédex, otherwise they are listed (and would
   be replaced by random species at load time).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

from data import load_records, species_from_records, species_names
from domain.pokedex import STATS, Pokedex, PokemonGen2

DESCRIPTION = "Build the Pokédex JSON from the training data plus manual entries."


def build(train_paths: Sequence[Path], manual_path: Path | None) -> list[PokemonGen2]:
    """Species from the training records, completed by the manual file."""
    species = {
        s.name: s
        for path in train_paths
        for s in species_from_records(load_records(path))
    }
    extracted = len(species)
    for entry in _manual_entries(manual_path):
        known = species.get(entry.name)
        if known is not None and known != entry:
            raise SystemExit(f"manual entry {entry} contradicts the data: {known}")
        species[entry.name] = entry
    print(
        f"{extracted} species from the training data, "
        f"{len(species) - extracted} added from {manual_path}"
    )
    return [species[name] for name in sorted(species)]


def missing_species(pokedex: Pokedex, paths: Iterable[Path]) -> dict[str, int]:
    """Species mentioned in ``paths`` but absent from the Pokédex -> battles."""
    missing: dict[str, int] = {}
    for path in paths:
        for record in load_records(path):
            for name in species_names(record):
                if name not in pokedex:
                    missing[name] = missing.get(name, 0) + 1
    return dict(sorted(missing.items()))


def to_json(species: Sequence[PokemonGen2]) -> list[dict[str, object]]:
    return [
        {
            "name": s.name,
            "types": list(s.types),
            **{f"base_{stat}": s.base(stat) for stat in ("hp", *STATS)},
        }
        for s in species
    ]


def _manual_entries(path: Path | None) -> list[PokemonGen2]:
    if path is None:
        return []
    with path.open(encoding="utf-8") as fh:
        return [PokemonGen2.from_dict(entry) for entry in json.load(fh)]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("train", nargs="+", type=Path, help="training .jsonl files")
    parser.add_argument("--manual", type=Path, help="JSON list of extra species")
    parser.add_argument("--out", type=Path, default=Path("assets/species.json"))
    parser.add_argument("--check", nargs="*", type=Path, default=[])
    args = parser.parse_args(argv)

    species = build(args.train, args.manual)
    pokedex = Pokedex(species)  # validates names, types and stats
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(to_json(species), indent=1) + "\n")
    print(f"{len(pokedex)} species written to {args.out}")

    missing = missing_species(pokedex, [*args.train, *args.check])
    for name, battles in missing.items():
        print(f"not in the Pokédex: {name} (in {battles} battle(s), replaced at load)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
