"""blend_v11: logit blend of a catalog model (default ensemble_from_lr) with ``v11_proba``.

The probability column comes from the ``*_v11`` feature sets (``features/v11_bridge.py``);
``w`` comes from ``tools/v11_blend_weight.py``. See ``models/core/logit_blend.py``.
"""

from __future__ import annotations

from typing import Any

from interfaces import Model
from models.core.logit_blend import LogitBlend
from registry import model


@model("blend_v11")
def build(name: str = "blend_v11", **params: Any) -> Model:
    return LogitBlend(name=name, **params)
