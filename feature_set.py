"""Registry of battle features.

A feature can be written in two ways:

* a **function** ``battle -> (name, value)``, for simple features;
* a subclass of :class:`Feature`, for anything more complex: several
  outputs at once, parameters, output names known in advance, or state
  learned from data with :meth:`Feature.fit` (for example statistics
  computed over all battles).

Both are collected in a :class:`FeatureSet`, which runs them, validates
their output and resolves name conflicts. The module knows nothing about
what a battle is: the battle type is a type parameter (in this project
it is :class:`battle.BattleContext`).

Example::

    features = FeatureSet[BattleContext]("base")

    @features.register
    def alive_delta(c: BattleContext) -> FeatureValue:
        return "d_alive_lr", c.last.p1.n_alive - c.last.p2.n_alive

    class HpTrend(Feature[BattleContext]):
        def __init__(self, window: int) -> None:
            super().__init__(feature_id=f"hp_trend_{window}")
            self.window = window

        @property
        def output_names(self) -> tuple[str, ...]:
            return (f"p1_hp_trend_{self.window}", f"p2_hp_trend_{self.window}")

        def compute(self, c: BattleContext) -> dict[str, float]:
            turns = c.window(self.window)
            return {
                f"{s}_hp_trend_{self.window}":
                    turns[-1].side(s).hp_total - turns[0].side(s).hp_total
                for s in SIDES
            }

    features.register(HpTrend(window=10))
    features.fit(all_battles)        # only needed by features with state
    table = features.to_frame(battles)

Leakage rules
-------------
* A stateful feature learns in :meth:`Feature.fit`, never in ``compute``.
* Features that use labels declare ``uses_labels = True``; they are fitted
  only on labelled training battles and never see ``unlabeled`` data.
* Label-free stateful features may also learn from ``unlabeled`` battles
  (for example the competition test set), never from validation data.
* The gym fits a fresh :meth:`FeatureSet.clone` inside every fold, on
  the training part only, so validation battles never influence a
  feature's state.
* Training rows of label-aware features are computed *out of fold*
  (:meth:`FeatureSet.fit_transform`): a battle's value never comes from
  a state fitted on its own label, so the model sees on training rows
  the same kind of values it will see on new battles.
"""

from __future__ import annotations

import copy
import math
import random
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, TypeVar

from interfaces import Feature, FeatureError, FeatureNotFittedError, FeatureValue, Row

BattleT = TypeVar("BattleT")

FeatureFn = Callable[[BattleT], FeatureValue]

SIDES = ("p1", "p2")
DEFAULT_INNER_FOLDS = 5


class ConflictPolicy(Enum):
    """What to do when two features use the same name."""

    RAISE = "raise"
    KEEP_FIRST = "keep_first"
    OVERWRITE = "overwrite"
    RENAME = "rename"


class ErrorPolicy(Enum):
    """What to do when a feature fails on a battle."""

    RAISE = "raise"
    NAN = "nan"


class FeatureConflictError(KeyError):
    """Two features claimed the same name under ConflictPolicy.RAISE."""


class FunctionFeature(Feature[BattleT]):
    """Adapter that turns a ``battle -> (name, value)`` function into a Feature."""

    def __init__(self, fn: FeatureFn, feature_id: str | None = None) -> None:
        if not callable(fn):
            raise TypeError(f"expected a callable, got {type(fn).__name__}")
        super().__init__(feature_id or getattr(fn, "__name__", repr(fn)))
        self._fn = fn

    def compute(self, battle: BattleT) -> Mapping[str, float]:
        result = self._fn(battle)
        if not (isinstance(result, tuple) and len(result) == 2):
            raise FeatureError(self.feature_id, "must return a (name, value) tuple")
        name, value = result
        return {name: value}


# ----------------------------------------------------------------------
# feature set
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class _Entry[BattleT]:
    """A registered feature and the prefix added to its outputs."""

    feature: Feature[BattleT]
    prefix: str = ""

    @property
    def feature_id(self) -> str:
        return self.feature.feature_id

    @property
    def output_names(self) -> tuple[str, ...] | None:
        names = self.feature.output_names
        return None if names is None else tuple(self.prefix + n for n in names)


class FeatureSet[BattleT]:
    """Ordered collection of features applied to one battle.

    Conflicts can happen at two moments:

    * registration: two features with the same id, or (when output names
      are declared) two features producing the same column;
    * extraction: two features return the same name.

    Both follow ``on_conflict``. With ``ConflictPolicy.RENAME`` the
    later name gets a numeric suffix (``name__2``, ``name__3``, ...).

    ``name`` identifies the set in experiments and saved models.
    """

    def __init__(
        self,
        name: str,
        on_conflict: ConflictPolicy = ConflictPolicy.RAISE,
        on_error: ErrorPolicy = ErrorPolicy.RAISE,
    ) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError("a feature set needs a non-empty name")
        self._name = name
        self._entries: dict[str, _Entry[BattleT]] = {}
        self._on_conflict = on_conflict
        self._on_error = on_error

    # ------------------------------------------------------------------
    # registration
    # ------------------------------------------------------------------
    def register(
        self,
        feature: FeatureFn | Feature[BattleT] | None = None,
        *,
        feature_id: str | None = None,
        prefix: str = "",
    ) -> Any:
        """Register a function or a :class:`Feature`; usable as a decorator.

        ``feature_id`` overrides the registry id (default: the function
        name, or the Feature's own id). ``prefix`` is prepended to every
        output name, to namespace groups of features. Returns the
        registered object unchanged.
        """
        if feature is None:
            return lambda f: self.register(f, feature_id=feature_id, prefix=prefix)
        wrapped = self._as_feature(feature, feature_id)
        entry = _Entry(wrapped, prefix)
        fid = entry.feature_id
        if fid in self._entries:
            resolved = self._resolve_id_conflict(fid)
            if resolved is None:
                return feature
            fid = resolved
        self._check_declared_outputs(entry, fid)
        self._entries[fid] = entry
        return feature

    def register_all(
        self, features: Iterable[FeatureFn | Feature[BattleT]], prefix: str = ""
    ) -> None:
        """Register several features with a shared prefix."""
        for feature in features:
            self.register(feature, prefix=prefix)

    # ------------------------------------------------------------------
    # fitting (see "Leakage rules" in the module docstring)
    # ------------------------------------------------------------------
    def fit(
        self,
        battles: Iterable[BattleT],
        labels: Sequence[int] | None = None,
        unlabeled: Iterable[BattleT] = (),
    ) -> FeatureSet[BattleT]:
        """Fit every stateful feature; return ``self``.

        ``battles`` and ``labels`` are the *training* data only. Features
        with ``uses_labels`` learn from them alone; label-free features
        learn from ``battles`` plus ``unlabeled``.
        """
        battles = list(battles)
        extra = list(unlabeled)
        if labels is not None and len(labels) != len(battles):
            raise ValueError("one label per training battle is required")
        for fid, entry in self._entries.items():
            feature = entry.feature
            if not feature.requires_fit:
                continue
            if feature.uses_labels:
                if labels is None:
                    raise ValueError(f"feature {fid!r} needs the training labels")
                feature.fit(battles, labels)
            else:
                feature.fit(battles + extra)
        return self

    def fit_transform(
        self,
        battles: Iterable[BattleT],
        labels: Sequence[int] | None = None,
        unlabeled: Iterable[BattleT] = (),
        groups: Sequence[Any] | None = None,
        inner_folds: int = DEFAULT_INNER_FOLDS,
        seed: int = 0,
    ) -> Any:
        """Fit on the training data and return its feature table.

        The set is fitted exactly as by :meth:`fit` (that state is used
        later on validation and test battles). Columns of label-aware
        features are then recomputed out of fold: battles are split in
        ``inner_folds`` folds (by ``groups`` if given, so duplicates stay
        together) and each fold's values come from a copy of the feature
        fitted on the other folds only. Other columns are unchanged.
        """
        import pandas as pd

        battles = list(battles)
        self.fit(battles, labels, unlabeled)
        label_aware = [
            f for f, e in self._entries.items() if e.feature.uses_labels and e.feature.inner_oof
        ]
        if not label_aware or labels is None:
            return pd.DataFrame(self.extract_many(battles))
        self._prepare(battles)
        folds = _inner_fold_ids(len(battles), groups, inner_folds, seed)
        rows: list[Row | None] = [None] * len(battles)
        for fold in sorted(set(folds)):
            inside = [i for i, f in enumerate(folds) if f == fold]
            outside = [i for i, f in enumerate(folds) if f != fold]
            entries = dict(self._entries)
            for fid in label_aware:
                twin = copy.deepcopy(self._entries[fid].feature)
                twin.reset()
                twin.fit([battles[i] for i in outside], [labels[i] for i in outside])
                entries[fid] = _Entry(twin, self._entries[fid].prefix)
            for i in inside:
                rows[i] = self._extract_with(entries, battles[i])
        return pd.DataFrame(rows)

    @property
    def requires_fit(self) -> bool:
        """True if some feature still needs :meth:`fit`."""
        return any(not e.feature.is_fitted for e in self._entries.values())

    @property
    def is_stateful(self) -> bool:
        """True if some feature learns state with :meth:`fit`."""
        return any(e.feature.requires_fit for e in self._entries.values())

    @property
    def uses_labels(self) -> bool:
        """True if some feature learns from the labels."""
        return any(e.feature.uses_labels for e in self._entries.values())

    def clone(self) -> FeatureSet[BattleT]:
        """Independent, unfitted copy (same name, features and prefixes)."""
        twin = self._empty_copy(self._name)
        for fid, entry in self._entries.items():
            feature = copy.deepcopy(entry.feature)
            feature.reset()
            twin._entries[fid] = _Entry(feature, entry.prefix)
        return twin

    def get_state(self) -> dict[str, dict[str, Any]]:
        """Learned state of every stateful feature, keyed by feature id."""
        if self.requires_fit:
            raise FeatureNotFittedError("fit() the feature set before saving it")
        return {
            fid: e.feature.get_state() for fid, e in self._entries.items() if e.feature.requires_fit
        }

    def set_state(self, state: dict[str, dict[str, Any]]) -> None:
        """Restore the state saved by :meth:`get_state`."""
        expected = {f for f, e in self._entries.items() if e.feature.requires_fit}
        if set(state) != expected:
            raise ValueError(
                f"saved state is for features {sorted(state)}, "
                f"this set has stateful features {sorted(expected)}"
            )
        for fid, feature_state in state.items():
            self._entries[fid].feature.set_state(feature_state)

    # ------------------------------------------------------------------
    # extraction
    # ------------------------------------------------------------------
    def extract(self, battle: BattleT) -> Row:
        """Run every feature on one battle and return ``{name: value}``."""
        return self._extract_with(self._entries, battle)

    def extract_many(self, battles: Iterable[BattleT]) -> list[Row]:
        """Run every feature on each battle, preserving order."""
        if self.requires_fit:
            unfitted = [fid for fid, e in self._entries.items() if not e.feature.is_fitted]
            raise FeatureNotFittedError(f"call fit() first; unfitted: {unfitted}")
        battles = list(battles)
        self._prepare(battles)
        return [self.extract(battle) for battle in battles]

    def _prepare(self, battles: list[BattleT]) -> None:
        """Let batch features compute all the battles at once."""
        for entry in self._entries.values():
            entry.feature.prepare(battles)

    def to_frame(
        self,
        battles: Iterable[BattleT],
        index: Callable[[BattleT], Any] | None = None,
    ) -> Any:
        """Return a ``pandas.DataFrame`` with one row per battle.

        ``index`` optionally maps a battle to its row label, for
        example its ``battle_id``.
        """
        import pandas as pd

        battles = list(battles)
        rows = self.extract_many(battles)
        labels = [index(b) for b in battles] if index else None
        return pd.DataFrame(rows, index=labels)

    # ------------------------------------------------------------------
    # composition
    # ------------------------------------------------------------------
    def select(self, feature_ids: Iterable[str], name: str | None = None) -> FeatureSet[BattleT]:
        """Return a new set with only the given registry ids, in order.

        The new set is called ``name`` (default ``"<name>[subset]"``).
        Features are shared, so their fitted state is kept.
        """
        subset = self._empty_copy(name or f"{self._name}[subset]")
        for fid in feature_ids:
            if fid not in self._entries:
                raise KeyError(f"no feature registered as {fid!r}")
            subset._entries[fid] = self._entries[fid]
        return subset

    def without(self, feature_ids: Iterable[str], name: str | None = None) -> FeatureSet[BattleT]:
        """Return a new set without the given registry ids, keeping the order.

        The new set is called ``name`` (default ``"<name>[subset]"``).
        Features are shared, so their fitted state is kept.
        """
        drops = set(feature_ids)
        missing = drops - set(self._entries)
        if missing:
            raise KeyError(f"{self._name}: no features {sorted(missing)} to drop")
        return self.select([f for f in self._entries if f not in drops], name)

    def merge(self, other: FeatureSet[BattleT], name: str | None = None) -> FeatureSet[BattleT]:
        """Return a new set with this set's features, then ``other``'s.

        The new set is called ``name`` (default ``"<name>+<other>"``).
        """
        merged = self._empty_copy(name or f"{self._name}+{other.name}")
        for fid, entry in [*self._entries.items(), *other._entries.items()]:
            merged.register(entry.feature, feature_id=fid, prefix=entry.prefix)
        return merged

    # ------------------------------------------------------------------
    # container protocol
    # ------------------------------------------------------------------
    @property
    def name(self) -> str:
        """Name of the set, used to tag results and saved models."""
        return self._name

    @property
    def feature_ids(self) -> list[str]:
        """Registry ids, in registration order."""
        return list(self._entries)

    @property
    def declared_columns(self) -> list[str] | None:
        """All output columns, if every feature declares them; else None."""
        columns: list[str] = []
        for entry in self._entries.values():
            names = entry.output_names
            if names is None:
                return None
            columns.extend(names)
        return columns

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, feature_id: object) -> bool:
        return feature_id in self._entries

    def __iter__(self) -> Iterator[str]:
        return iter(self._entries)

    def __repr__(self) -> str:
        return (
            f"FeatureSet({self._name!r}, {len(self)} features, "
            f"on_conflict={self._on_conflict.value}, "
            f"on_error={self._on_error.value})"
        )

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _empty_copy(self, name: str) -> FeatureSet[BattleT]:
        return FeatureSet(name, self._on_conflict, self._on_error)

    @staticmethod
    def _as_feature(
        feature: FeatureFn | Feature[BattleT], feature_id: str | None
    ) -> Feature[BattleT]:
        if isinstance(feature, Feature):
            if feature_id is not None and feature_id != feature.feature_id:
                return _Renamed(feature, feature_id)
            return feature
        if callable(feature):
            return FunctionFeature(feature, feature_id)
        raise TypeError(f"expected a function or a Feature, got {type(feature).__name__}")

    def _extract_with(self, entries: Mapping[str, _Entry[BattleT]], battle: BattleT) -> Row:
        row: Row = {}
        for fid, entry in entries.items():
            for name, value in self._run(entry, fid, battle).items():
                self._store(row, name, value, fid)
        return row

    def _run(self, entry: _Entry[BattleT], fid: str, battle: BattleT) -> Row:
        try:
            values = entry.feature(battle)
        except (FeatureNotFittedError, FeatureError):
            raise  # configuration or validation bugs: never turned into NaN
        except Exception as exc:
            if self._on_error is ErrorPolicy.NAN:
                names = entry.output_names or (entry.prefix + fid,)
                return {n: math.nan for n in names}
            raise FeatureError(fid, repr(exc)) from exc
        return {entry.prefix + name: value for name, value in values.items()}

    def _store(self, row: Row, name: str, value: float, fid: str) -> None:
        if name not in row:
            row[name] = value
            return
        policy = self._on_conflict
        if policy is ConflictPolicy.RAISE:
            raise FeatureConflictError(
                f"feature {fid!r} returned {name!r}, already set by an earlier feature"
            )
        if policy is ConflictPolicy.OVERWRITE:
            row[name] = value
        elif policy is ConflictPolicy.RENAME:
            row[_free_name(name, row)] = value

    def _resolve_id_conflict(self, fid: str) -> str | None:
        policy = self._on_conflict
        if policy is ConflictPolicy.RAISE:
            raise FeatureConflictError(f"{fid!r} is already registered")
        if policy is ConflictPolicy.KEEP_FIRST:
            return None
        if policy is ConflictPolicy.OVERWRITE:
            return fid
        return _free_name(fid, self._entries)

    def _check_declared_outputs(self, entry: _Entry[BattleT], fid: str) -> None:
        """Under RAISE, reject declared columns already declared by others."""
        names = entry.output_names
        if names is None or self._on_conflict is not ConflictPolicy.RAISE:
            return
        for other_id, other in self._entries.items():
            if other_id == fid or other.output_names is None:
                continue
            clash = set(names) & set(other.output_names)
            if clash:
                raise FeatureConflictError(
                    f"feature {fid!r} declares {sorted(clash)}, already declared by {other_id!r}"
                )


class _Renamed(Feature[BattleT]):
    """A Feature registered under a different id (shares the original)."""

    def __init__(self, inner: Feature[BattleT], feature_id: str) -> None:
        super().__init__(feature_id)
        self._inner = inner
        self.requires_fit = inner.requires_fit
        self.uses_labels = inner.uses_labels
        self.inner_oof = inner.inner_oof

    @property
    def output_names(self) -> tuple[str, ...] | None:
        return self._inner.output_names

    @property
    def is_fitted(self) -> bool:
        return self._inner.is_fitted

    def fit(
        self, battles: Sequence[BattleT], labels: Sequence[int] | None = None
    ) -> Feature[BattleT]:
        self._inner.fit(battles, labels)
        return self

    def prepare(self, battles: Sequence[BattleT]) -> None:
        self._inner.prepare(battles)

    def reset(self) -> None:
        super().reset()
        self._inner.reset()

    def get_state(self) -> dict[str, Any]:
        return self._inner.get_state()

    def set_state(self, state: dict[str, Any]) -> None:
        self._inner.set_state(state)

    def compute(self, battle: BattleT) -> Mapping[str, float]:
        return self._inner.compute(battle)


def _inner_fold_ids(n: int, groups: Sequence[Any] | None, k: int, seed: int) -> list[int]:
    """Assign each of ``n`` rows to one of ``k`` folds, whole groups at a time."""
    keys = list(range(n)) if groups is None else list(groups)
    if len(keys) != n:
        raise ValueError("one group per training battle is required")
    unique = list(dict.fromkeys(keys))
    if k < 2 or len(unique) < k:
        raise ValueError(f"cannot split {len(unique)} groups into {k} inner folds")
    random.Random(seed).shuffle(unique)
    fold_of = {key: i % k for i, key in enumerate(unique)}
    return [fold_of[key] for key in keys]


def _free_name(name: str, taken: Iterable[str]) -> str:
    """Return ``name__k`` with the smallest k >= 2 not in ``taken``."""
    taken = set(taken)
    k = 2
    while f"{name}__{k}" in taken:
        k += 1
    return f"{name}__{k}"


# ----------------------------------------------------------------------
# helpers to build common feature shapes
# ----------------------------------------------------------------------
