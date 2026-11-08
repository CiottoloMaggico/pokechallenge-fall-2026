"""Files of tuned parameters: written by ``tune``, read by ``--params``.

A file is a JSON object with at least ``model`` and ``params``; ``tune`` also writes the
feature set, the metric and its value, the accuracy, the number of trials and the date.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_params(path: Path, **info: Any) -> Path:
    """Write ``info`` (``model``, ``params`` and any other JSON fields) to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info, indent=2) + "\n")
    return path


def load_params(path: Path) -> tuple[str, dict[str, Any]]:
    """Model name and parameters of a file written by :func:`save_params`."""
    info = json.loads(path.read_text())
    return str(info["model"]), dict(info["params"])
