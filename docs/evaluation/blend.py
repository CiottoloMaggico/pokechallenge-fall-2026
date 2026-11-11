"""The third submission: the 50/50 blend of two models' probabilities, averaged as logits.

    python docs/evaluation/blend.py compact_proba.csv v11_proba.csv --out submission_blend.csv

``expit(w logit(p_a) + (1 - w) logit(p_b))`` with the probabilities clipped to [1e-6, 1 - 1e-6],
the formula of v11's own blend of its members. The inputs are the ``--proba`` files of
``main.py predict`` (``battle_id``, ``player_won``, ``p_player_won``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit

CLIP = 1e-6
WEIGHT = 0.5


def blend(p_a: np.ndarray, p_b: np.ndarray, weight: float = WEIGHT) -> np.ndarray:
    """``weight`` on the logit of ``p_a``, the rest on the logit of ``p_b``."""
    z_a, z_b = (logit(np.clip(p, CLIP, 1 - CLIP)) for p in (p_a, p_b))
    return expit(weight * z_a + (1 - weight) * z_b)


def blend_files(first: Path, second: Path, out: Path, proba: Path | None = None) -> None:
    """Write the submission (and optionally the probabilities) of the blend of two models."""
    a, b = pd.read_csv(first), pd.read_csv(second)
    if not a["battle_id"].equals(b["battle_id"]):
        raise SystemExit("the two files do not list the same battles in the same order")
    p = blend(a["p_player_won"].to_numpy(), b["p_player_won"].to_numpy())
    table = pd.DataFrame({"battle_id": a["battle_id"], "player_won": (p >= 0.5).astype(int)})
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    if proba is not None:
        table.assign(p_player_won=p).to_csv(proba, index=False)
    print(f"{len(table)} predictions written to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="50/50 logit blend of two probability files")
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--proba", type=Path, help="also write the blended probabilities")
    args = parser.parse_args()
    blend_files(args.first, args.second, args.out, args.proba)
