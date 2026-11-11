"""The figures and tables of the notebook, drawn from the tables of ``summarize.py``.

Only pandas, numpy and matplotlib: the notebook's kernel imports this file without the
project's pinned environment. Colours are Apple's system colours (Human Interface Guidelines,
light appearance), on a white background with a light grid and no frame.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.axes import Axes
from matplotlib.figure import Figure

BLUE, ORANGE, PURPLE = "#007AFF", "#FF9500", "#AF52DE"
GREEN, RED, GRAY = "#34C759", "#FF3B30", "#8E8E93"
LABEL, SECONDARY, GRID, FILL = "#1C1C1E", "#636366", "#E5E5EA", "#F2F2F7"
MODELS = {"compact_v5": "compact_v5", "v11": "v11", "blend": "blend 50/50"}
COLORS = {"compact_v5": BLUE, "v11": ORANGE, "blend": PURPLE}


FONTS = ("SF Pro Display", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans")


def apple_style() -> None:
    """Matplotlib defaults of every figure (with the first of ``FONTS`` that is installed)."""
    installed = {font.name for font in font_manager.fontManager.ttflist}
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 200,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.family": [name for name in FONTS if name in installed],
            "font.size": 11,
            "text.color": LABEL,
            "axes.labelcolor": SECONDARY,
            "axes.edgecolor": GRID,
            "axes.titlesize": 14,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.titlepad": 14,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "xtick.color": SECONDARY,
            "ytick.color": SECONDARY,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "lines.linewidth": 2.2,
            "lines.solid_capstyle": "round",
            "legend.frameon": False,
        }
    )


def save(fig: Figure, folder: Path | None, name: str) -> Figure:
    """Tighten ``fig`` and, with a folder, write ``<name>.png`` there."""
    fig.tight_layout()
    if folder is not None:
        folder.mkdir(parents=True, exist_ok=True)
        fig.savefig(folder / f"{name}.png", bbox_inches="tight")
    return fig


def roc_points(y: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """False and true positive rates at every threshold (ties kept together)."""
    order = np.argsort(-p, kind="stable")
    y, p = y[order], p[order]
    last = np.r_[np.flatnonzero(np.diff(p)), len(p) - 1]
    tpr = np.r_[0.0, np.cumsum(y)[last] / y.sum()]
    fpr = np.r_[0.0, np.cumsum(1 - y)[last] / (1 - y).sum()]
    return fpr, tpr


def plot_roc(oof: pd.DataFrame, metrics: pd.DataFrame, folder: Path | None = None) -> Figure:
    """ROC of the out-of-fold probabilities of the three models, with a zoom on the knee."""
    auc = metrics.set_index("model")["roc_auc"]
    y = oof["y_true"].to_numpy()
    fig, ax = plt.subplots(figsize=(6.6, 6.2))
    zoom = ax.inset_axes((0.50, 0.10, 0.44, 0.44))
    for model, name in MODELS.items():
        fpr, tpr = roc_points(y, oof[model].to_numpy())
        ax.plot(fpr, tpr, color=COLORS[model], label=f"{name}  ·  AUC {auc[model]:.4f}")
        zoom.plot(fpr, tpr, color=COLORS[model], linewidth=1.8)
    ax.plot([0, 1], [0, 1], color=GRAY, linestyle=(0, (2, 3)), linewidth=1.2, label="chance")
    zoom.set(xlim=(0.15, 0.35), ylim=(0.60, 0.80))
    zoom.tick_params(labelsize=8)
    zoom.set_facecolor(FILL)
    ax.indicate_inset_zoom(zoom, edgecolor=GRAY)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1))
    ax.set_title("ROC curves, out-of-fold predictions")
    ax.set_aspect("equal")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=2)
    return save(fig, folder, "roc_curves")


def band(ax: Axes, table: Any, color: str, label: str, style: Any = "-") -> None:
    """Mean line of ``table`` (index x; columns mean, std) with a ± 1 std band."""
    ax.plot(table.index, table["mean"], linestyle=style, color=color, label=label)
    low, high = table["mean"] - table["std"], table["mean"] + table["std"]
    ax.fill_between(table.index, low, high, color=color, alpha=0.12, linewidth=0)


def plot_accuracy_by_turn(turns: pd.DataFrame, folder: Path | None = None) -> Figure:
    """Accuracy of compact_v5 when the battles are cut to their first N turns."""
    grouped: Any = turns.groupby("turn")["accuracy"]
    table = grouped.agg(["mean", "std"])
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    band(ax, table, BLUE, "compact_v5 (mean ± std over the folds)")
    ax.scatter(table.index, table["mean"], color=BLUE, s=26, zorder=3)
    ax.axhline(0.5, color=GRAY, linestyle=(0, (2, 3)), linewidth=1.2, label="chance")
    for turn in (table.index[0], 10, 20, table.index[-1]):
        if turn in table.index:
            value = table.loc[turn, "mean"]
            ax.annotate(f"{value:.1%}", (turn, value), xytext=(0, 10), ha="center",
                        textcoords="offset points", color=LABEL, fontsize=10)  # fmt: skip
    ax.set(xlabel="Last turn of the battle shown to the model", ylabel="Accuracy")
    ax.set_title("The later the cut, the more accurate the prediction")
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
    ax.legend(loc="lower right")
    return save(fig, folder, "accuracy_by_turn")


def curve_stats(curve: pd.DataFrame, model: str, split: str) -> Any:
    """Mean and std over the folds of the log loss of ``model`` by training size."""
    part: Any = curve[(curve["model"] == model) & (curve["split"] == split)]
    table = part.groupby("size").agg(
        n=("n_train", "mean"), mean=("log_loss", "mean"), std=("log_loss", "std")
    )
    return table.set_index("n").fillna(0.0)


def plot_loss_convergence(curve: pd.DataFrame, folder: Path | None = None) -> Figure:
    """Validation log loss of the three models as the training battles grow."""
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for model, name in MODELS.items():
        table = curve_stats(curve, model, "valid")
        band(ax, table, COLORS[model], name)
        ax.scatter(table.index, table["mean"], color=COLORS[model], s=14, zorder=3)
        ax.annotate(f"{table['mean'].iloc[-1]:.3f}", (table.index[-1], table["mean"].iloc[-1]),
                    xytext=(8, 0), textcoords="offset points", va="center", fontsize=10,
                    color=COLORS[model])  # fmt: skip
    ax.set(xlabel="Training battles", ylabel="Validation log loss")
    ax.set_title("Convergence of the validation log loss")
    ax.xaxis.set_major_formatter(lambda value, _: f"{value:,.0f}")
    ax.set_xlim(0, 8700)
    ax.legend(loc="upper right")
    return save(fig, folder, "loss_convergence")


def plot_generalization_gap(curve: pd.DataFrame, folder: Path | None = None) -> Figure:
    """Training against validation log loss by training size, one panel per model."""
    fig, axes = plt.subplots(1, len(MODELS), figsize=(13, 4.4), sharey=True)
    for ax, (model, name) in zip(axes, MODELS.items(), strict=True):
        train, valid = curve_stats(curve, model, "train"), curve_stats(curve, model, "valid")
        band(ax, valid, COLORS[model], "validation")
        band(ax, train, COLORS[model], "training", style=(0, (4, 2)))
        ax.fill_between(train.index, train["mean"], valid["mean"], color=FILL, linewidth=0)
        gap = valid["mean"].iloc[-1] - train["mean"].iloc[-1]
        ax.set_title(f"{name}  ·  gap {gap:.3f}", fontsize=12)
        ax.set_xlabel("Training battles")
        ax.xaxis.set_major_formatter(lambda value, _: f"{value:,.0f}")
        ax.legend(loc="center right")
    axes[0].set_ylabel("Log loss")
    fig.suptitle("Training and validation log loss", x=0.01, ha="left", fontsize=14,
                 fontweight="semibold")  # fmt: skip
    return save(fig, folder, "generalization_gap")


def table_style(table: pd.DataFrame) -> Any:
    """A plain table with Apple's label colours and hairline separators."""
    cell = [("padding", "7px 12px"), ("border-bottom", f"1px solid {GRID}"), ("color", LABEL)]
    head = [*cell, ("color", SECONDARY), ("font-weight", "600"), ("text-align", "left")]
    styles: Any = [{"selector": "th", "props": head}, {"selector": "td", "props": cell}]
    font = "-apple-system, 'SF Pro Text', 'Helvetica Neue', Arial, sans-serif"
    table_props = [("border-collapse", "collapse"), ("font-family", font), ("font-size", "13px")]
    styles.append({"selector": "", "props": table_props})
    return table.style.hide(axis="index").set_table_styles(styles)


def metrics_table(metrics: pd.DataFrame) -> Any:
    """The metrics of the three models, the best value of each column in bold."""
    table: Any = metrics.assign(model=metrics["model"].replace(MODELS))
    table = table.rename(columns={
        "model": "Model", "accuracy": "Accuracy", "log_loss": "Log loss", "roc_auc": "ROC AUC",
        "brier": "Brier", "accuracy_fold_std": "± fold", "accuracy_seed_std": "± seed",
        "folds": "Folds",
    })  # fmt: skip
    shown = ["Model", "Accuracy", "± fold", "± seed", "Log loss", "ROC AUC", "Brier", "Folds"]
    table = table[shown].dropna(axis="columns", how="all")  # no "± seed" with a single seed
    bold = f"font-weight: 600; color: {BLUE}"
    styled = table_style(table).format(precision=4)
    styled = styled.highlight_max(subset=["Accuracy", "ROC AUC"], props=bold)
    return styled.highlight_min(subset=["Log loss", "Brier"], props=bold)


def paired_table(paired: pd.DataFrame) -> Any:
    """Paired differences ``a - b`` per metric as ``delta (p)``, one row per pair of models."""
    shown = paired.assign(
        pair=paired["a"].replace(MODELS) + "  −  " + paired["b"].replace(MODELS),
        value=[f"{d:+.4f}  (p {p:.3f})" for d, p in zip(paired["delta"], paired["p"], strict=True)],
    )
    wide: Any = shown.pivot(index="pair", columns="metric", values="value")
    pairs = list(dict.fromkeys(shown["pair"]))
    wide = wide.loc[pairs, ["accuracy", "log_loss", "roc_auc", "brier"]]
    wide.columns = ["Δ accuracy", "Δ log loss", "Δ ROC AUC", "Δ Brier"]
    return table_style(wide.rename_axis("Difference").reset_index())


def mapping_table(mapping: pd.DataFrame) -> Any:
    """The v11 groups next to the compact_v5 features that measure the same thing; the groups
    with no counterpart are greyed."""
    alone = {k for k, cell in mapping["compact_v5 features"].items() if str(cell) == "—"}

    def shade(row: pd.Series) -> list[str]:
        return [f"color: {GRAY}" if row.name in alone else ""] * len(row)

    styled = table_style(mapping).apply(shade, axis=1)
    return styled.set_properties(subset=["compact_v5 features"], **{"font-weight": "600"})
