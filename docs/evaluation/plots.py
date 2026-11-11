"""The figures and tables of the notebook, drawn from the tables of ``summarize.py``.

Only pandas, numpy and matplotlib: the notebook's kernel imports this file without the
project's pinned environment. Serif text, Apple's system colours, a light grid and no frame.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# Apple system colours
BLUE, INDIGO, RED = "#007AFF", "#5856D6", "#FF3B30"
GRAY, LABEL, SECONDARY, GRID, FILL = "#8E8E93", "#1C1C1E", "#636366", "#E5E5EA", "#F2F2F7"
MODELS = {"compact_v5": "compact_v5", "v11": "v11", "blend": "blend 50/50"}
COLORS = {"compact_v5": BLUE, "v11": RED, "blend": INDIGO}
MIN_BIN = 50  # calibration bins with fewer battles are not drawn


def apple_style() -> None:
    """Matplotlib defaults of every figure: serif text, centred bold titles, light grid."""
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 200,
            "font.family": "serif",
            "font.serif": ["STIXGeneral"],
            "mathtext.fontset": "stix",
            "font.size": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "center",
            "axes.linewidth": 0.6,
            "grid.linewidth": 0.4,
            "legend.fontsize": 8,
            "text.color": LABEL,
            "axes.labelcolor": SECONDARY,
            "axes.edgecolor": SECONDARY,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "axes.axisbelow": True,
            "xtick.color": SECONDARY,
            "ytick.color": SECONDARY,
            "lines.linewidth": 1.6,
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
    """ROC of the out-of-fold probabilities and, beside it, the true positive rate of v11 and
    of the blend minus compact_v5's (moving average of width 0.2 over the false positive rate):
    the curves are too close to tell apart, their difference is not."""
    auc = metrics.set_index("model")["roc_auc"]
    y = oof["y_true"].to_numpy()
    grid = np.linspace(0, 1, 2001)
    tpr = {}
    for model in MODELS:
        fpr, rate = roc_points(y, oof[model].to_numpy())
        tpr[model] = np.interp(grid, fpr, rate)
    fig, (roc, diff) = plt.subplots(1, 2, figsize=(10, 4.4))
    for model, name in MODELS.items():
        roc.plot(grid, tpr[model], color=COLORS[model], label=f"{name} (AUC {auc[model]:.3f})")
        if model != "compact_v5":
            delta = pd.Series(tpr[model] - tpr["compact_v5"])
            smooth = delta.rolling(401, center=True, min_periods=1).mean()
            diff.plot(grid, smooth, color=COLORS[model], label=name)
    roc.plot([0, 1], [0, 1], color=GRAY, linestyle=(0, (2, 3)), linewidth=1)
    diff.axhline(0, color=GRAY, linestyle=(0, (2, 3)), linewidth=1)
    roc.set(xlabel="False positive rate", ylabel="True positive rate", aspect="equal")
    roc.set_title("ROC, out-of-fold")
    roc.legend(loc="lower right")
    diff.set(xlabel="False positive rate", ylabel="TPR − TPR of compact_v5")
    diff.set_title("Difference from compact_v5")
    diff.legend(loc="upper right")
    return save(fig, folder, "roc_curves")


def save_table(table: pd.DataFrame, folder: Path | None, name: str) -> None:
    """With a folder, write the numbers behind a figure as ``<name>.csv`` there."""
    if folder is not None:
        folder.mkdir(parents=True, exist_ok=True)
        table.to_csv(folder / f"{name}.csv", index=False)


def calibration_table(
    oof: pd.DataFrame, folder: Path | None = None, bins: int = 10
) -> pd.DataFrame:
    """Per model and equal-width bin of p over [0, 1]: battles, mean p, observed share of P1
    wins, ``few`` (fewer than ``MIN_BIN`` battles: kept here, not drawn) and the model's ECE;
    with a folder, also written there as ``calibration.csv``."""
    edges = np.linspace(0, 1, bins + 1)
    y = oof["y_true"].to_numpy()
    rows = []
    for model in MODELS:
        p = oof[model].to_numpy()
        index = np.digitize(p, edges[1:-1])  # 0 .. bins - 1, p = 1 in the last bin
        for b in range(bins):
            inside = index == b
            n = int(inside.sum())
            mean_p, observed = (p[inside].mean(), y[inside].mean()) if n else (np.nan, np.nan)
            rows.append({"model": model, "bin": b, "low": edges[b], "high": edges[b + 1],
                         "n": n, "mean_p": mean_p, "observed": observed})  # fmt: skip
    table = pd.DataFrame(rows)
    table["few"] = table["n"] < MIN_BIN
    weighted: Any = (table["mean_p"] - table["observed"]).abs() * table["n"]
    battles: Any = table.groupby("model")["n"].sum()
    ece: Any = weighted.groupby(table["model"]).sum() / battles
    table = table.assign(ece=table["model"].map(ece))
    save_table(table, folder, "calibration")
    return table


def plot_calibration(table: pd.DataFrame, folder: Path | None = None) -> Figure:
    """Reliability curves of ``calibration_table`` over the diagonal, battles per bin below."""
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(6, 6.4), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    top.plot([0, 1], [0, 1], color=GRAY, linestyle=(0, (2, 3)), linewidth=1, label="ideal")
    width = 0.1 / (len(MODELS) + 1)
    for k, (model, name) in enumerate(MODELS.items()):
        part: Any = table[table["model"] == model]
        shown = part[~part["few"]]
        label = f"{name} (ECE {part['ece'].iloc[0]:.4f})"
        top.plot(shown["mean_p"], shown["observed"], color=COLORS[model], marker="o",
                 markersize=3.5, label=label)  # fmt: skip
        centres = (part["low"] + part["high"]) / 2 + (k - 1) * width
        bottom.bar(centres, part["n"], width=width, color=COLORS[model], label=name)
    top.set(ylabel="Observed share of P1 wins", xlim=(0, 1), ylim=(0, 1), aspect="equal")
    top.set_title("Calibration, out-of-fold")
    top.legend(loc="upper left")
    top.tick_params(labelbottom=True)
    bottom.set(xlabel="Mean predicted probability p", ylabel="Battles")
    bottom.set_xticks(np.linspace(0, 1, 11))
    return save(fig, folder, "calibration")


def confidence_table(
    oof: pd.DataFrame, folder: Path | None = None, step: float = 0.005, least: int = 200
) -> pd.DataFrame:
    """Per model and threshold t from 0.5: battles with confidence max(p, 1 - p) >= t, their
    share (coverage) and accuracy, until fewer than ``least`` battles remain; with a folder,
    also written there as ``accuracy_vs_confidence.csv``."""
    y = oof["y_true"].to_numpy()
    rows = []
    for model in MODELS:
        p = oof[model].to_numpy()
        confidence, right = np.maximum(p, 1 - p), (p > 0.5) == y
        for t in np.round(np.arange(0.5, 1, step), 4):
            kept = confidence >= t
            if kept.sum() < least:
                break
            rows.append({"model": model, "threshold": t, "n": int(kept.sum()),
                         "coverage": kept.mean(), "accuracy": right[kept].mean()})  # fmt: skip
    table = pd.DataFrame(rows)
    save_table(table, folder, "accuracy_vs_confidence")
    return table


def plot_accuracy_vs_confidence(table: pd.DataFrame, folder: Path | None = None) -> Figure:
    """Accuracy above each confidence threshold (top) and the coverage left (bottom), with the
    overall accuracy (t = 0.5) marked."""
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(7.4, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    for model, name in MODELS.items():
        part: Any = table[table["model"] == model]
        overall = part["accuracy"].iloc[0]
        top.plot(part["threshold"], part["accuracy"], color=COLORS[model],
                 label=f"{name} ({overall:.1%} at t = 0.5)")  # fmt: skip
        top.scatter([0.5], [overall], color=COLORS[model], s=18, zorder=3)
        bottom.plot(part["threshold"], part["coverage"], color=COLORS[model])
    top.set(ylabel="Accuracy (confidence ≥ t)")
    top.set_title("Accuracy vs confidence threshold")
    top.legend(loc="upper left")
    top.tick_params(labelbottom=True)
    bottom.set(xlabel="Confidence threshold t", ylabel="Coverage", ylim=(0, 1.02))
    for ax in (top, bottom):
        ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    return save(fig, folder, "accuracy_vs_confidence")


def disagreement_table(
    oof: pd.DataFrame, folder: Path | None = None, resamples: int = 2000, seed: int = 0
) -> pd.DataFrame:
    """Accuracy of the three models on the battles where compact_v5 and v11 predict the same
    class (agreement) and on the others (disagreement), with a 95 % percentile bootstrap
    interval over the battles of each group (the same resamples for the three models); with a
    folder, also written there as ``disagreement.csv``."""
    y = oof["y_true"].to_numpy()
    right = {model: (oof[model].to_numpy() > 0.5) == y for model in MODELS}
    same = (oof["compact_v5"] > 0.5).to_numpy() == (oof["v11"] > 0.5).to_numpy()
    rng = np.random.default_rng(seed)
    rows = []
    for group, members in (("agreement", same), ("disagreement", ~same)):
        battles = np.flatnonzero(members)
        samples = rng.integers(0, len(battles), (resamples, len(battles)), dtype=np.int32)
        for model in MODELS:
            hits = right[model][battles]
            low, high = np.quantile(hits[samples].mean(axis=1), [0.025, 0.975])
            rows.append({"group": group, "model": model, "n": len(battles),
                         "share": len(battles) / len(y), "accuracy": hits.mean(),
                         "ci_low": low, "ci_high": high})  # fmt: skip
    table = pd.DataFrame(rows)
    save_table(table, folder, "disagreement")
    return table


def plot_disagreement(table: pd.DataFrame, folder: Path | None = None) -> Figure:
    """Accuracy of the three models where compact_v5 and v11 agree and disagree (95 % CI)."""
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    groups = list(dict.fromkeys(table["group"]))
    width = 0.8 / len(MODELS)
    bars = {"elinewidth": 0.8, "capsize": 2.5, "ecolor": LABEL}
    for k, (model, name) in enumerate(MODELS.items()):
        part = table[table["model"] == model].set_index("group").loc[groups]
        x = np.arange(len(groups)) + (k - 1) * width
        error = [part["accuracy"] - part["ci_low"], part["ci_high"] - part["accuracy"]]
        ax.bar(x, part["accuracy"], width=width * 0.92, color=COLORS[model], label=name,
               yerr=error, error_kw=bars)  # fmt: skip
        for xi, value, top in zip(x, part["accuracy"], part["ci_high"], strict=True):
            ax.annotate(f"{value:.1%}", (xi, top), xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=8)  # fmt: skip
    sizes = table.drop_duplicates("group").set_index("group")["n"]
    ax.set_xticks(np.arange(len(groups)), [f"{g}\n({sizes[g]:,} battles)" for g in groups])
    ax.axhline(0.5, color=GRAY, linestyle=(0, (2, 3)), linewidth=1)
    ax.set(ylabel="Accuracy", ylim=(0, 1))
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.set_title("Accuracy by agreement of compact_v5 and v11")
    ax.legend(loc="upper right", ncol=3)
    return save(fig, folder, "disagreement")


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
    ax.scatter(table.index, table["mean"], color=BLUE, s=16, zorder=3)
    ax.axhline(0.5, color=GRAY, linestyle=(0, (2, 3)), linewidth=1, label="chance")
    for turn in (table.index[0], 10, 20, table.index[-1]):
        if turn in table.index:
            value = table.loc[turn, "mean"]
            ax.annotate(f"{value:.1%}", (turn, value), xytext=(0, 8), ha="center",
                        textcoords="offset points", color=LABEL, fontsize=8)  # fmt: skip
    ax.set(xlabel="Last turn of the battle shown to the model", ylabel="Accuracy")
    ax.set_title("Accuracy by last turn seen")
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
    ax.legend(loc="upper left")
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
    last = {}
    for model, name in MODELS.items():
        table = curve_stats(curve, model, "valid")
        band(ax, table, COLORS[model], name)
        ax.scatter(table.index, table["mean"], color=COLORS[model], s=10, zorder=3)
        last[model] = (table.index[-1], table["mean"].iloc[-1])
    # final values, spread so that close ones stay readable
    for rank, model in enumerate(sorted(last, key=lambda m: -last[m][1])):
        x, value = last[model]
        ax.annotate(f"{value:.4f}", (x, value), xytext=(10, 10 - 10 * rank), va="center",
                    textcoords="offset points", fontsize=8, color=COLORS[model])  # fmt: skip
    ax.set(xlabel="Training battles", ylabel="Validation log loss")
    ax.set_title("Validation log loss by training size")
    ax.xaxis.set_major_formatter(lambda value, _: f"{value:,.0f}")
    ax.set_xlim(0, 8900)
    ax.legend(loc="upper right")
    return save(fig, folder, "loss_convergence")


def plot_generalization_gap(curve: pd.DataFrame, folder: Path | None = None) -> Figure:
    """Training against validation log loss by training size, one panel per model."""
    fig, axes = plt.subplots(1, len(MODELS), figsize=(12, 4), sharey=True)
    for ax, (model, name) in zip(axes, MODELS.items(), strict=True):
        train, valid = curve_stats(curve, model, "train"), curve_stats(curve, model, "valid")
        band(ax, valid, COLORS[model], "validation")
        band(ax, train, COLORS[model], "training", style=(0, (4, 2)))
        ax.fill_between(train.index, train["mean"], valid["mean"], color=FILL, linewidth=0)
        ax.set_title(name, fontweight="normal")
        ax.tick_params(labelleft=True)  # numbered y axis on every panel
        ax.set_xlabel("Training battles")
        ax.xaxis.set_major_formatter(lambda value, _: f"{value:,.0f}")
    axes[0].legend(loc="lower right")
    axes[0].set_ylabel("Log loss")
    fig.suptitle("Training vs validation log loss", fontsize=11, fontweight="bold")
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
        "accuracy_fold_std": "± fold",
    })  # fmt: skip
    table = table[["Model", "Accuracy", "± fold", "Log loss", "ROC AUC"]]
    bold = f"font-weight: 600; color: {BLUE}"
    styled = table_style(table).format(precision=4)
    styled = styled.highlight_max(subset=["Accuracy", "ROC AUC"], props=bold)
    return styled.highlight_min(subset=["Log loss"], props=bold)
