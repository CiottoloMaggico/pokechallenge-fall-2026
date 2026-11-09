"""v11's team effect: a proxy for the player behind a team.

A copy of ``TeamEffect`` of the original's ``models.py`` restricted to what ``blend_v11`` runs
(``joint``, ``split``, ``lam`` and ``lam_pair`` dropped); same constants and operation order.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from models.core.v11_stacking import SEED

State = tuple[Any, ...]


class TeamEffect(BaseEstimator, TransformerMixin):
    """Shrunk mean residual (w.r.t. the cross-fitted logit ``base``) of the training battles
    played with the same team.

    A battle counts for its P1 team and, with opposite sign, for its P2 team when known; with
    ``link``, a P2 team with five revealed species is matched to the only training team
    containing them; with ``pair``, also per (P1 team, P2 team) pairing. With ``soft``, a P2
    team with ``soft_min`` to five revealed species is matched to all the training teams that
    contain them, weighted by how often each was played: such battles spread their residual over
    these candidates and get their weighted effect (``tm_soft``, also added to ``tm`` when the P2
    team is not known otherwise). With ``raw``, the same matching also gives the raw rating of
    both teams, the shrunk win rate minus one half (``tm_raw1``, ``tm_raw2``, shrinkage
    ``raw_a``). The ``k_*`` columns are replaced by the ``tm*`` ones. ``fit_transform`` is
    cross-fitted, ``transform`` uses all training rows.
    """

    def __init__(
        self,
        base: str = "stack_logit_full",
        a: float = 5.0,
        p2: bool = True,
        link: bool = False,
        pair: bool = False,
        n_splits: int = 5,
        seed: int = SEED,
        drop_base: bool = False,
        soft: bool = False,
        soft_min: int = 3,
        raw: bool = False,
        raw_a: float = 2.0,
    ) -> None:
        self.base, self.a, self.p2, self.link, self.pair = base, a, p2, link, pair
        self.n_splits, self.seed, self.drop_base = n_splits, seed, drop_base
        self.soft, self.soft_min, self.raw, self.raw_a = soft, soft_min, raw, raw_a

    def _links(self, X: pd.DataFrame) -> defaultdict[Any, set[Any]]:
        links: defaultdict[Any, set[Any]] = defaultdict(set)
        if self.link:
            for side in ("1", "2"):
                subs = X[[f"k_p{side}sub{k}" for k in range(6)]].to_numpy()
                for team, row in zip(X[f"k_p{side}team"].to_numpy(), subs, strict=True):
                    if not np.isnan(team):
                        for sub in row:
                            links[sub].add(team)
        return links

    def _keys(self, X: pd.DataFrame, links: dict[Any, set[Any]]) -> tuple[np.ndarray, np.ndarray]:
        k1, k2 = X["k_p1team"].to_numpy(), X["k_p2team"].to_numpy().copy()
        if self.link:
            part = X["k_p2part"].to_numpy()
            for i in np.flatnonzero(~np.isnan(part)):
                cands = links.get(part[i], set()) - {k1[i]}
                if len(cands) == 1:
                    k2[i] = cands.pop()
        return k1, k2

    def _fit(self, X: pd.DataFrame, y: np.ndarray) -> State:
        links = self._links(X)
        k1, k2 = self._keys(X, links)
        base = X[self.base].to_numpy()
        r = y - expit(base)
        s: defaultdict[Any, float] = defaultdict(float)
        c: defaultdict[Any, int] = defaultdict(int)
        for a, b, ri in zip(k1, k2, r, strict=True):
            s[a] += ri
            c[a] += 1
            if not np.isnan(b):
                if self.p2:
                    s[b] -= ri
                    c[b] += 1
                if self.pair:
                    s[a, b] += ri
                    c[a, b] += 1
                    s[b, a] -= ri
                    c[b, a] += 1
        soft = self._soft_fit(X, r) if self.soft else None
        return s, c, links, soft, self._soft_fit(X, y - 0.5) if self.raw else None

    @staticmethod
    def _species(X: pd.DataFrame) -> tuple[list[frozenset[Any]], list[frozenset[Any]]]:
        p1 = X[[f"k_p1sp{k}" for k in range(6)]].to_numpy()
        p2 = X[[f"k_p2sp{k}" for k in range(6)]].to_numpy()
        return [frozenset(row) for row in p1], [
            frozenset(v for v in row if not np.isnan(v)) for row in p2
        ]

    @staticmethod
    def _candidates(
        index: dict[Any, set[frozenset[Any]]], revealed: frozenset[Any], own: frozenset[Any]
    ) -> list[frozenset[Any]]:
        cands: set[frozenset[Any]] | None = None
        for sp in sorted(revealed):
            cands = set(index.get(sp, ())) if cands is None else cands & index.get(sp, set())
            if not cands:
                return []
        return sorted((cands or set()) - {own}, key=sorted)

    def _partial(self, revealed: frozenset[Any]) -> bool:
        return self.soft_min <= len(revealed) < 6

    def _soft_fit(self, X: pd.DataFrame, r: np.ndarray) -> State:
        """Team strengths over sets of species: full teams as usual, then every partially
        revealed P2 team shares its residual among the candidates in proportion to their
        counts."""
        sp1, sp2 = self._species(X)
        s: defaultdict[frozenset[Any], float] = defaultdict(float)
        c: defaultdict[frozenset[Any], float] = defaultdict(float)
        for t1, t2, ri in zip(sp1, sp2, r, strict=True):
            s[t1] += ri
            c[t1] += 1
            if len(t2) == 6:
                s[t2] -= ri
                c[t2] += 1
        index: defaultdict[Any, set[frozenset[Any]]] = defaultdict(set)
        for team in c:
            for sp in team:
                index[sp].add(team)
        add_s: defaultdict[frozenset[Any], float] = defaultdict(float)
        add_c: defaultdict[frozenset[Any], float] = defaultdict(float)
        for t1, t2, ri in zip(sp1, sp2, r, strict=True):
            cands = self._candidates(index, t2, t1) if self._partial(t2) else []
            if cands:
                w = np.array([c[t] for t in cands])
                for team, wt in zip(cands, w / w.sum(), strict=True):
                    add_s[team] -= wt * ri
                    add_c[team] += wt
        for team in add_s:
            s[team] += add_s[team]
            c[team] += add_c[team]
        return s, c, index

    def _soft_encode(self, X: pd.DataFrame, state: State) -> tuple[np.ndarray, np.ndarray]:
        s, c, index = state
        sp1, sp2 = self._species(X)
        effect, count = np.zeros(len(X)), np.zeros(len(X))
        for i, (t1, t2) in enumerate(zip(sp1, sp2, strict=True)):
            cands = self._candidates(index, t2, t1) if self._partial(t2) else []
            if cands:
                w = np.array([c[t] for t in cands])
                e = np.array([s[t] / (c[t] + self.a) for t in cands])
                effect[i], count[i] = -(w @ e) / w.sum(), w.sum()
        return effect, count

    def _raw_encode(self, X: pd.DataFrame, state: State) -> tuple[np.ndarray, np.ndarray]:
        s, c, index = state
        sp1, sp2 = self._species(X)

        def rate(team: frozenset[Any]) -> float:
            return s.get(team, 0.0) / (c.get(team, 0.0) + self.raw_a)

        p1, p2 = np.array([rate(t) for t in sp1]), np.zeros(len(X))
        for i, (t1, t2) in enumerate(zip(sp1, sp2, strict=True)):
            if len(t2) == 6:
                p2[i] = rate(t2)
            elif self._partial(t2):
                cands = self._candidates(index, t2, t1)
                if cands:
                    w = np.array([c[t] for t in cands])
                    p2[i] = w @ np.array([rate(t) for t in cands]) / w.sum()
        return p1, p2

    def _encode(self, X: pd.DataFrame, state: State) -> dict[str, np.ndarray]:
        s, c, links, soft, raw = state
        k1, k2 = self._keys(X, links)
        known = ~np.isnan(k2)
        n1 = np.array([c.get(k, 0) for k in k1], dtype=float)
        n2 = np.array([c.get(k, 0) if ok else 0 for k, ok in zip(k2, known, strict=True)], float)
        e1 = np.array([s.get(k, 0.0) for k in k1]) / (n1 + self.a)
        s2 = np.array([s.get(k, 0.0) if ok else 0.0 for k, ok in zip(k2, known, strict=True)])
        e2 = s2 / (n2 + self.a)
        out = {"tm": e1 - (e2 if self.p2 else 0.0), "tm_n1": n1, "tm_n2": n2}
        if self.pair:
            pairs = list(zip(k1, k2, strict=True))
            n = np.array(
                [c.get(p, 0) if ok else 0 for p, ok in zip(pairs, known, strict=True)], float
            )
            out["tm_pair"] = np.array(
                [s.get(p, 0.0) if ok else 0.0 for p, ok in zip(pairs, known, strict=True)]
            ) / (n + self.a)
            out["tm_npair"] = n
        if self.soft:
            out["tm_soft"], out["tm_nsoft"] = self._soft_encode(X, soft)
            out["tm"] = out["tm"] + np.where(n2 == 0, out["tm_soft"], 0.0)
        if self.raw:
            out["tm_raw1"], out["tm_raw2"] = self._raw_encode(X, raw)
        return out

    def fit(self, X: pd.DataFrame, y: Any) -> TeamEffect:
        """Fit on all rows."""
        self.state_ = self._fit(X, np.asarray(y))
        return self

    def fit_transform(self, X: pd.DataFrame, y: Any = None, **kw: Any) -> pd.DataFrame:
        """Attach the cross-fitted team columns, then fit on all rows."""
        y = np.asarray(y)
        cols: dict[str, np.ndarray] = {}
        if X.index.is_unique:
            splits = StratifiedKFold(self.n_splits, shuffle=True, random_state=self.seed).split(
                X, y
            )
        else:
            splits = StratifiedGroupKFold(
                self.n_splits, shuffle=True, random_state=self.seed
            ).split(X, y, X.index)
        for tr, va in splits:
            for name, v in self._encode(X.iloc[va], self._fit(X.iloc[tr], y[tr])).items():
                cols.setdefault(name, np.zeros(len(X)))[va] = v
        self.state_ = self._fit(X, y)
        return self._attach(X, cols)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Attach the team columns of the model fitted on all rows."""
        return self._attach(X, self._encode(X, self.state_))

    def _attach(self, X: pd.DataFrame, cols: dict[str, np.ndarray]) -> pd.DataFrame:
        keys = [c for c in X.columns if c.startswith("k_")]
        drop = keys + ([self.base] if self.drop_base else [])
        return pd.concat([X.drop(columns=drop), pd.DataFrame(cols, index=X.index)], axis=1)
