"""FeatureSet composition, batch hook and out-of-fold values of label-aware features."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import pytest

from feature_set import FeatureSet
from interfaces import Feature, FeatureError, FeatureNotFittedError


class Plain(Feature[int]):
    """The battle itself, as a column named after the feature."""

    def compute(self, battle: int) -> Mapping[str, float]:
        return {self.feature_id: float(battle)}


class Memory(Feature[int]):
    """The label seen in fit for this battle: NaN if the state never saw that battle."""

    requires_fit = True
    uses_labels = True

    def __init__(self, feature_id: str = "memory") -> None:
        super().__init__(feature_id)
        self.seen: dict[int, int] = {}

    def _fit(self, battles: Sequence[int], labels: Sequence[int] | None) -> None:
        assert labels is not None
        self.seen = dict(zip(battles, labels, strict=True))

    def compute(self, battle: int) -> Mapping[str, float]:
        return {self.feature_id: float(self.seen.get(battle, math.nan))}


class NoInnerFolds(Memory):
    inner_oof = False


class Batch(Feature[int]):
    """Records what the batch hook received."""

    def __init__(self) -> None:
        super().__init__("batch")
        self.prepared: list[int] = []

    def prepare(self, battles: Sequence[int]) -> None:
        self.prepared = list(battles)

    def compute(self, battle: int) -> Mapping[str, float]:
        return {"batch": float(battle)}


def three_plain() -> FeatureSet[int]:
    features = FeatureSet[int]("base")
    features.register_all([Plain("a"), Plain("b"), Plain("c")])
    return features


def test_without_keeps_the_order_and_shares_the_features() -> None:
    features = three_plain()
    subset = features.without(["b"], name="subset")
    assert subset.name == "subset"
    assert subset.feature_ids == ["a", "c"]
    assert list(subset.to_frame([1]).columns) == ["a", "c"]


def test_without_refuses_an_unknown_feature() -> None:
    with pytest.raises(KeyError, match="no features"):
        three_plain().without(["nope"])


def test_prepare_sees_the_whole_batch_before_extraction() -> None:
    feature = Batch()
    features = FeatureSet[int]("batch")
    features.register(feature)
    features.to_frame([1, 2, 3])
    assert feature.prepared == [1, 2, 3]


def test_training_rows_of_a_label_aware_feature_are_out_of_fold() -> None:
    features = FeatureSet[int]("memory")
    features.register(Memory())
    battles = list(range(20))
    table = features.fit_transform(battles, [b % 2 for b in battles])
    assert table["memory"].isna().all()
    assert features.to_frame([3])["memory"].tolist() == [1.0]


def test_duplicates_stay_in_the_same_inner_fold() -> None:
    features = FeatureSet[int]("memory")
    features.register(Memory())
    battles = [b // 2 for b in range(20)]
    table = features.fit_transform(battles, [0] * 20, groups=battles)
    assert table["memory"].isna().all()


def test_inner_oof_false_skips_the_inner_folds() -> None:
    features = FeatureSet[int]("memory")
    features.register(NoInnerFolds())
    table = features.fit_transform([1, 2, 3], [0, 1, 0])
    assert table["memory"].tolist() == [0.0, 1.0, 0.0]


def test_a_registered_alias_keeps_the_interface_of_the_feature() -> None:
    feature = NoInnerFolds()
    features = FeatureSet[int]("alias")
    features.register(feature, feature_id="other")
    assert features.fit_transform([1, 2, 3], [0, 1, 0])["memory"].tolist() == [0.0, 1.0, 0.0]
    batch = Batch()
    features.register(batch, feature_id="other_batch")
    features.to_frame([7])
    assert batch.prepared == [7]


def test_a_stateful_feature_must_be_fitted() -> None:
    features = FeatureSet[int]("memory")
    features.register(Memory())
    with pytest.raises(FeatureNotFittedError):
        features.to_frame([1])


def test_a_feature_cannot_return_a_non_number() -> None:
    class Bad(Feature[int]):
        def compute(self, battle: int) -> Mapping[str, float]:
            return {"bad": "x"}  # type: ignore[dict-item]

    features = FeatureSet[int]("bad")
    features.register(Bad())
    with pytest.raises(FeatureError, match="must be a number"):
        features.to_frame([1])
