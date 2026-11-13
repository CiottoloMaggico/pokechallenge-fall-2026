"""The tables and the figure of the notebook's section on hidden knockouts, from ``data/``.

Only pandas, numpy and matplotlib, like ``draw.py``. The tables come from ``knockouts.py``
(``knockout_counts``, ``_agreement``, ``_votes``, ``_situations``, ``_outcome``) and from
``knockout_swap.py`` (``knockout_swap``, ``knockout_reverse``).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plots import BLUE, GRAY, INDIGO, save, table_style  # noqa: E402

GREEN, ORANGE = "#34C759", "#FF9500"
NAMES = {
    "none": "no hidden knockout", "v3": "ours: detector of compact_v5",
    "v11": "ours: detector of v11", "rule": "ours: earlier hand rule",
    "second": "2nd place: model", "spikes": "9th place: hand rule",
}  # fmt: skip
COLORS = {"v3": INDIGO, "v11": BLUE, "rule": GRAY, "second": GREEN, "spikes": ORANGE}
MODELS = {"logreg": "logistic regression", "ensemble_from_lr": "ensemble"}
SECOND = {"five_view": "five views", "team_similarity": "five views + teams",
          "tactical": "tactical view", "tactical_team_spikes": "tactical + teams"}  # fmt: skip
METRICS = (("accuracy", "Accuracy", "p "), ("log_loss", "Log loss", "p  "),
           ("roc_auc", "ROC AUC", "p   "))  # fmt: skip


def counts_table(counts: pd.DataFrame) -> Any:
    """Per detector: knockouts counted among P2's last exits and the sum of its scores."""
    table = pd.DataFrame({
        "Detector": counts["team"].replace(NAMES),
        "Knockouts counted": counts["knockouts"],
        "Share of the exits": counts["knockouts"] / counts["exits"],
        "Sum of the scores": counts["sum_of_scores"],
    })  # fmt: skip
    formats = {"Knockouts counted": "{:,}", "Share of the exits": "{:.1%}",
               "Sum of the scores": "{:,.0f}"}  # fmt: skip
    return table_style(table).format(formats)


def agreement_table(agreement: pd.DataFrame, detectors: list[str]) -> Any:
    """Per pair of ``detectors``: same call, knockouts of both and of one only, kappa."""
    pairs: Any = agreement[agreement["a"].isin(detectors) & agreement["b"].isin(detectors)]
    table = pd.DataFrame({
        "Detectors": [f"{NAMES[a]} / {NAMES[b]}"
                      for a, b in zip(pairs["a"], pairs["b"], strict=True)],
        "Same call": pairs["same"],
        "Knockout for both": pairs["both"],
        "Only the first": pairs["only_a"],
        "Only the second": pairs["only_b"],
        "Kappa": pairs["kappa"],
    })  # fmt: skip
    formats = {"Same call": "{:.1%}", "Knockout for both": "{:,}", "Only the first": "{:,}",
               "Only the second": "{:,}", "Kappa": "{:.2f}"}  # fmt: skip
    return table_style(table).format(formats)


def plot_situations(situations: pd.DataFrame, folder: Path | None = None) -> Figure:
    """Share of P2's last exits each detector calls a knockout, by what the record shows."""
    panels = list(dict.fromkeys(situations["situation"]))
    widths = [(situations["situation"] == name).sum() for name in panels]
    fig, axes = plt.subplots(1, len(panels), figsize=(12, 4.2), sharey=True, width_ratios=widths)
    for ax, name in zip(axes, panels, strict=True):
        part = situations[situations["situation"] == name]
        x = np.arange(len(part))
        for k, team in enumerate(COLORS):
            ax.bar(x + (k - 2) * 0.16, part[team], width=0.16, color=COLORS[team],
                   label=NAMES[team])  # fmt: skip
        labels = [f"{value}\n({n:,})" for value, n in zip(part["value"], part["exits"],
                                                         strict=True)]  # fmt: skip
        ax.set_xticks(x, [label.replace(", no", ",\nno") for label in labels], fontsize=8)
        ax.set_title(name, fontweight="normal", fontsize=10)
        ax.grid(axis="x", visible=False)
        ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    axes[0].set_ylabel("Exits called a knockout")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Hidden knockouts by situation (exits in brackets)", fontsize=11,
                 fontweight="bold")  # fmt: skip
    return save(fig, folder, "knockout_situations")


def outcome_table(outcome: pd.DataFrame) -> Any:
    """ROC AUC of "P2's knockouts minus P1's" for the winner, per detector."""
    wide: Any = outcome.pivot(index="team", columns="knockouts")
    order = [team for team in NAMES if team in wide.index]
    rows = []
    for team in order:
        row: dict[str, Any] = {"Hidden knockouts": NAMES[team]}
        for kind, label in (("counted", "counted"), ("scores", "sum of the scores")):
            value = wide.loc[team, ("roc_auc", kind)]
            low, high = (wide.loc[team, (f"minus_v3_{side}", kind)] for side in ("low", "high"))
            row[f"AUC, {label}"] = value
            reference = team == "v3" or np.isnan(low)
            row[f"minus ours ({label})"] = "" if reference else f"{low:+.3f} to {high:+.3f}"
        rows.append(row)
    table = pd.DataFrame(rows)
    formats = {column: "{:.3f}" for column in table if column.startswith("AUC")}
    return table_style(table).format(formats, na_rep="")


def swap_table(swap: pd.DataFrame) -> Any:
    """Our models with each detector's knockouts: metrics and difference from our detector."""
    rows = []
    for arm in NAMES:
        for model, label in MODELS.items():
            part = swap[(swap["arm"] == arm) & (swap["model"] == model)].set_index("metric")
            row: dict[str, Any] = {"Hidden knockouts": NAMES[arm], "Model": label}
            for metric, name, p in METRICS[:2]:
                row[name] = part.loc[metric, "mean"]
                row[f"Δ {name.lower()}"] = part.loc[metric, "delta"] if arm != "v3" else np.nan
                row[p] = part.loc[metric, "p"]
            rows.append(row)
    formats = {"Accuracy": "{:.4f}", "Log loss": "{:.4f}", "Δ accuracy": "{:+.4f}",
               "Δ log loss": "{:+.4f}", "p ": "{:.3f}", "p  ": "{:.3f}"}  # fmt: skip
    return table_style(pd.DataFrame(rows)).format(formats, na_rep="")


def reverse_table(reverse: pd.DataFrame) -> Any:
    """The 2nd place's models with their own knockout scores and with our detector's."""
    rows = []
    for model, label in SECOND.items():
        part = reverse[reverse["model"] == model].set_index("metric")
        row: dict[str, Any] = {"2nd place's model": label}
        for metric, name, p in METRICS:
            row[f"{name}, theirs"] = part.loc[metric, "theirs"]
            row[f"{name}, ours"] = part.loc[metric, "ours"]
            row[p] = part.loc[metric, "p"]
        rows.append(row)
    table = pd.DataFrame(rows)
    formats = {column: "{:.3f}" if column.startswith("p") else "{:.4f}"
               for column in table.columns[1:]}  # fmt: skip
    return table_style(table).format(formats)
