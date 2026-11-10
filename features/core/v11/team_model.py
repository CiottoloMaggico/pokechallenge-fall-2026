"""v11's expected composition of the unrevealed part of Player 2's team, from team co-occurrences
in the training battles. A copy of ``TeamModel`` of the original's ``features/hidden.py`` (its
``hidden_features`` are not ported)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from features.core.v11.battle import Battle


class TeamModel:
    """Naive-Bayes co-occurrence model of complete teams, fitted on the P1 teams of training
    battles; gives the expected composition of player 2's unrevealed slots."""

    def __init__(
        self, battles: Sequence[Battle], alpha: float = 1.0, temperature: float = 0.5
    ) -> None:
        names = sorted({p["name"] for b in battles for p in b.team})
        self.index = {n: i for i, n in enumerate(names)}
        T = np.zeros((len(battles), len(names)))
        for k, b in enumerate(battles):
            for p in b.team:
                T[k, self.index[p["name"]]] = 1
        self.names = names
        cnt = T.sum(0)
        self.log_prior = np.log((cnt + alpha) / (len(battles) + 2 * alpha))
        co = T.T @ T
        cond = (co + alpha) / (cnt[:, None] + 2 * alpha)
        marg = (cnt + alpha) / (len(battles) + 2 * alpha)
        self.log_lift = np.log(cond / marg[None, :])
        self.temperature = temperature

    def expected_unrevealed(self, revealed: Sequence[str], n_hidden: int) -> dict[str, float]:
        """Expected presence of each species in the ``n_hidden`` slots (sums to ``n_hidden``
        before the cap at 1; entries under 1e-3 dropped)."""
        if n_hidden <= 0:
            return {}
        idx = [self.index[r] for r in revealed if r in self.index]
        score = self.log_prior + self.temperature * self.log_lift[:, idx].sum(1)
        score[idx] = -np.inf
        q = np.exp(score - score[np.isfinite(score)].max())
        q /= q.sum()
        e = 1 - (1 - q) ** n_hidden
        e *= n_hidden / e.sum()
        e = np.minimum(e, 1.0)
        return {self.names[i]: float(e[i]) for i in np.flatnonzero(e > 1e-3)}
