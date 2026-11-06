"""What the application, the features and the models agree on.

:class:`Feature` is what every concrete feature extends. :class:`FeatureSource` and
:class:`FittableFeatureSource` are what the gym and the saved runs need from a table of features;
:class:`feature_set.FeatureSet` is the implementation. :class:`Model` is the binary classifier the
gym fits and compares, whatever library is behind it.
"""
from __future__ import annotations

import copy
import pickle
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from numbers import Real
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    import pandas as pd  # pyright: ignore[reportMissingTypeStubs]

FeatureValue = tuple[str, float]
Row = dict[str, float]


class FeatureError(RuntimeError):
    """A feature failed or returned an invalid value."""

    def __init__(self, feature_id: str, message: str) -> None:
        super().__init__(f"feature {feature_id!r}: {message}")
        self.feature_id = feature_id


class FeatureNotFittedError(RuntimeError):
    """A feature that needs ``fit`` was used before being fitted.

    This is a configuration error, so it is raised whatever the
    ErrorPolicy.
    """


class Feature[BattleT](ABC):
    """Base class for features that need more than a plain function.

    Subclasses implement :meth:`compute`, returning ``{name: value}``
    with one or more outputs. Optionally they:

    * override :attr:`output_names` to declare their columns, which lets
      the FeatureSet detect name conflicts at registration and fill the
      right columns with NaN under ``ErrorPolicy.NAN``;
    * set ``requires_fit = True`` and implement :meth:`_fit` to learn
      state from battles before computing;
    * also set ``uses_labels = True`` if :meth:`_fit` needs the labels;
    * set ``inner_oof = False`` if they never output in-sample values for
      their own training battles (e.g. NaN), so they skip the inner folds;
    * implement :meth:`prepare` to compute a whole batch at once.

    The learned state is saved with :meth:`get_state` (by default every
    instance attribute, so keep it picklable) and restored with
    :meth:`set_state`.
    """

    requires_fit: bool = False
    uses_labels: bool = False
    inner_oof: bool = True

    def __init__(self, feature_id: str | None = None) -> None:
        self._feature_id = feature_id or type(self).__name__
        self._fitted = False

    @property
    def feature_id(self) -> str:
        """Registry id of this feature (defaults to the class name)."""
        return self._feature_id

    @property
    def output_names(self) -> tuple[str, ...] | None:
        """Names of the produced columns, or None if known only at runtime."""
        return None

    @property
    def is_fitted(self) -> bool:
        """True when the feature can compute: fitted, or nothing to fit."""
        return self._fitted or not self.requires_fit

    def fit(
        self, battles: Sequence[BattleT], labels: Sequence[int] | None = None
    ) -> Feature[BattleT]:
        """Learn the feature's state; return ``self``.

        ``labels`` must be given (aligned with ``battles``) if and only
        if the feature declares ``uses_labels``.
        """
        if self.uses_labels:
            if labels is None or len(labels) != len(battles):
                raise ValueError(f"feature {self.feature_id!r} needs one label per battle")
        elif labels is not None:
            raise ValueError(f"feature {self.feature_id!r} does not use labels: do not pass them")
        self._fit(battles, labels)
        self._fitted = True
        return self

    def _fit(self, battles: Sequence[BattleT], labels: Sequence[int] | None) -> None:  # noqa: B027
        """Hook for stateful features; stateless ones do nothing."""

    def prepare(self, battles: Sequence[BattleT]) -> None:  # noqa: B027
        """Called with every battle before the row-by-row extraction (e.g. a whole
        external pipeline run once on the batch, then read back by ``compute``)."""

    def reset(self) -> None:
        """Forget the fitted flag (used on clones before refitting)."""
        self._fitted = False

    def get_state(self) -> dict[str, Any]:
        """Learned state, for saving a trained model (default: attributes)."""
        return copy.deepcopy(vars(self))

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore a state returned by :meth:`get_state`."""
        vars(self).update(copy.deepcopy(state))

    @abstractmethod
    def compute(self, battle: BattleT) -> Mapping[str, float]:
        """Return ``{name: value}`` for one battle."""

    def __call__(self, battle: BattleT) -> Row:
        if not self.is_fitted:
            raise FeatureNotFittedError(f"feature {self.feature_id!r} needs fit() before use")
        return _validate(self.feature_id, self.compute(battle))

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.feature_id!r})"


def _validate(feature_id: str, values: object) -> Row:
    """Check that a feature returned ``{non-empty str: number}``."""
    if not isinstance(values, Mapping) or not values:
        raise FeatureError(feature_id, "must return a non-empty {name: value} map")
    row: Row = {}
    for name, value in values.items():  # pyright: ignore[reportUnknownVariableType]
        if not isinstance(name, str) or not name:
            raise FeatureError(feature_id, "names must be non-empty strings")
        if not isinstance(value, Real):
            raise FeatureError(
                feature_id,
                f"value of {name!r} must be a number, got {type(value).__name__}",  # pyright: ignore[reportUnknownArgumentType]
            )
        row[name] = float(value)
    return row


@runtime_checkable
class FeatureSource(Protocol):
    """Anything that turns battles into a named feature table.

    :class:`feature_set.FeatureSet` satisfies this protocol.
    """

    @property
    def name(self) -> str: ...

    def to_frame(self, battles: Iterable[Any], /) -> pd.DataFrame: ...


@runtime_checkable
class FittableFeatureSource(Protocol):
    """A feature source with learned state (a FeatureSet with ``fit``)."""

    @property
    def name(self) -> str: ...

    @property
    def is_stateful(self) -> bool: ...

    def to_frame(self, battles: Iterable[Any], /) -> pd.DataFrame: ...

    def clone(self) -> Any: ...

    def fit_transform(
        self,
        battles: Iterable[Any],
        labels: Sequence[int] | None = None,
        unlabeled: Iterable[Any] = (),
        groups: Sequence[Any] | None = None,
        /,
    ) -> pd.DataFrame: ...

    def get_state(self) -> dict[str, Any]: ...

    def set_state(self, state: dict[str, Any], /) -> None: ...


def is_stateful(source: FeatureSource) -> bool:
    """True if ``source`` learns something and must be fitted inside each fold."""
    return isinstance(source, FittableFeatureSource) and source.is_stateful


class NotFittedError(RuntimeError):
    """``predict`` or ``predict_proba`` called before ``fit``."""


class Model(ABC):
    """A binary classifier: positive class = Player 1 wins (label 1)."""

    def __init__(self, name: str) -> None:
        if not name:
            raise ValueError("a model needs a non-empty name")
        self._name = name
        self._fitted = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit(self, x: pd.DataFrame, y: Any) -> Model:
        """Train on ``x`` and binary labels ``y``; return ``self``."""
        labels = np.asarray(y)
        _check_binary(labels)
        self._fit(x, labels)
        self._fitted = True
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        """Return P(label = 1) for each row, as a 1-D array."""
        self._require_fitted()
        proba = np.asarray(self._predict_proba(x), dtype=float).ravel()
        if proba.size and (proba.min() < 0 or proba.max() > 1):
            raise ValueError(f"{self.name}: probabilities outside [0, 1]")
        return proba

    def predict(self, x: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Return hard 0/1 predictions (1 when P(label = 1) > threshold)."""
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        return (self.predict_proba(x) > threshold).astype(int)

    @abstractmethod
    def clone(self) -> Model:
        """Return an unfitted copy with the same configuration."""

    @abstractmethod
    def get_params(self) -> dict[str, Any]:
        """Return the configuration, for logging and reports."""

    def save(self, path: str | Path) -> None:
        """Serialize the model (fitted or not) with pickle."""
        with Path(path).open("wb") as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(path: str | Path) -> Model:
        """Load a model saved with :meth:`save` (only trusted files)."""
        with Path(path).open("rb") as fh:
            model = pickle.load(fh)
        if not isinstance(model, Model):
            raise TypeError(f"{path} does not contain a Model")
        return model

    def __repr__(self) -> str:
        state = "fitted" if self._fitted else "unfitted"
        return f"{type(self).__name__}({self.name!r}, {state})"

    @abstractmethod
    def _fit(self, x: pd.DataFrame, y: np.ndarray) -> None:
        """Train the underlying learner."""

    @abstractmethod
    def _predict_proba(self, x: pd.DataFrame) -> Any:
        """Return P(label = 1) for each row."""

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise NotFittedError(f"{self.name}: call fit() first")


def _check_binary(y: np.ndarray) -> None:
    values = set(np.unique(y).tolist())
    if not values <= {0, 1} or len(values) < 2:
        raise ValueError(f"labels must contain both 0 and 1, got {values}")
