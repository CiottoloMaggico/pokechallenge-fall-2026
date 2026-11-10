"""v11's symmetric bags (Player 1 minus Player 2) of species, alive HP and revealed moves, over
vocabularies built from the training battles. A copy of the original's ``features/bags.py``."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

from features.core.v11_fs.battle import Battle
from features.core.v11_fs.state import Track


class Vocab:
    """Species and move vocabularies built from training battles only (sorted names)."""

    def __init__(self, battles: Sequence[Battle], min_species: int = 20, min_moves: int = 30):
        sp: Counter[str] = Counter()
        mv: Counter[str] = Counter()
        for b in battles:
            sp.update(p["name"] for p in b.team)
            sp.update(b.sides["p2"].revealed.keys())
            for s in ("p1", "p2"):
                mv.update({m for moves in b.sides[s].revealed.values() for m in moves})
        self.species = sorted(k for k, v in sp.items() if v >= min_species)
        self.moves = sorted(k for k, v in mv.items() if v >= min_moves)


def bag_features(b: Battle, vocab: Vocab, t1: Track, t2: Track) -> dict[str, Any]:
    """The ``bs_`` (species), ``bh_`` (alive HP) and ``bm_`` (moves) columns of a battle."""
    f: dict[str, Any] = {}
    team = {p["name"] for p in b.team}
    rev2 = set(b.sides["p2"].revealed)
    for s in vocab.species:
        f[f"bs_{s}"] = float(s in team) - float(s in rev2)
        h1 = (1 - t1["pf"][s]) * t1["hp"][s] if s in team else 0.0
        h2 = (1 - t2["pf"][s]) * t2["hp"][s] if s in rev2 else 0.0
        f[f"bh_{s}"] = h1 - h2
    m1 = {m for mv in b.sides["p1"].revealed.values() for m in mv}
    m2 = {m for mv in b.sides["p2"].revealed.values() for m in mv}
    for m in vocab.moves:
        f[f"bm_{m}"] = float(m in m1) - float(m in m2)
    return f
