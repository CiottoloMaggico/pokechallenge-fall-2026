"""What the application and the features agree on: a feature, and a source of feature tables.

:class:`Feature` is what every concrete feature extends. :class:`FeatureSource` and
:class:`FittableFeatureSource` are what the gym and the saved runs need from a table of features;
:class:`feature_set.FeatureSet` is the implementation.
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from numbers import Real
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

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
