"""The figures and tables of the comparison notebook, from the tables of ``data/``.

    python docs/evaluation/comparison/draw.py [--data D] [--figures F]

Only pandas, numpy and matplotlib, with the style and helpers of the notebook's evaluation
(``docs/evaluation/plots.py``). The figures compare one model per team: ``blend`` (ours), the
2nd-placed team's and the 8th-placed team's (Fantastic Four) best private submissions.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import plots  # noqa: E402
from plots import GRAY, GRID, INDIGO, LABEL, apple_style, roc_points, save, save_table  # noqa: E402

GREEN, ORANGE = "#34C759", "#FF9500"
MODELS = {"blend": "blend", "second": "2nd place", "third": "8th place"}
COLORS = {"blend": INDIGO, "second": GREEN, "third": ORANGE}
TEAMS = {"ours": "ours", "second": "2nd place", "third": "8th place"}
TEAM_COLORS = {"ours": INDIGO, "second": GREEN, "third": ORANGE}
NAMES = {  # every model of the tables
    "compact_v5": "compact_v5", "v11": "v11", "blend": "blend",
    "second_five_views": "2nd: five views", "second_teams": "2nd: five views + teams",
    "second": "2nd: tactical + teams",
    "third_ensemble": "8th: boosters + CatBoost + LR", "third_catboost": "8th: CatBoost",
    "third": "8th: boosters + LR",
}  # fmt: skip
DASHED = (0, (2, 3))


@contextmanager
def compared(models: dict[str, str] = MODELS, colors: dict[str, str] = COLORS) -> Iterator[None]:
    """The figures of ``plots.py`` drawn for ``models`` instead of our three submissions."""
    saved = plots.MODELS, plots.COLORS
    plots.MODELS, plots.COLORS = models, colors
    try:
        yield
    finally:
        plots.MODELS, plots.COLORS = saved


def plot_roc(oof: pd.DataFrame, metrics: pd.DataFrame, folder: Path | None = None) -> Figure:
    """ROC of the three teams and, beside it, the true positive rate of the other two minus
    ``blend``'s (moving average of width 0.2 over the false positive rate)."""
    auc = metrics.set_index("model")["roc_auc"]
    y, grid = oof["y_true"].to_numpy(), np.linspace(0, 1, 2001)
    tpr = {m: np.interp(grid, *roc_points(y, oof[m].to_numpy())) for m in MODELS}
    fig, (roc, diff) = plt.subplots(1, 2, figsize=(10, 4.4))
    for model, name in MODELS.items():
        roc.plot(grid, tpr[model], color=COLORS[model], label=f"{name} (AUC {auc[model]:.3f})")
        if model != "blend":
            smooth = pd.Series(tpr[model] - tpr["blend"]).rolling(401, center=True, min_periods=1)
            diff.plot(grid, smooth.mean(), color=COLORS[model], label=name)
    roc.plot([0, 1], [0, 1], color=GRAY, linestyle=DASHED, linewidth=1)
    diff.axhline(0, color=GRAY, linestyle=DASHED, linewidth=1)
    roc.set(xlabel="False positive rate", ylabel="True positive rate", aspect="equal")
    roc.set_title("ROC, out-of-fold")
    roc.legend(loc="lower right")
    diff.set(xlabel="False positive rate", ylabel="TPR − TPR of blend")
    diff.set_title("Difference from blend")
    diff.legend(loc="lower right")
    return save(fig, folder, "roc_curves")


def plot_calibration(oof: pd.DataFrame, data: Path | None, folder: Path | None = None) -> Figure:
    """``plots.plot_calibration`` for the three teams (table written to ``data``)."""
    with compared():
        return plots.plot_calibration(plots.calibration_table(oof, data), folder)


def plot_confidence(oof: pd.DataFrame, data: Path | None, folder: Path | None = None) -> Figure:
    """``plots.plot_accuracy_vs_confidence`` for the three teams (table written to ``data``)."""
    with compared():
        return plots.plot_accuracy_vs_confidence(plots.confidence_table(oof, data), folder)


def disagreement_table(
    oof: pd.DataFrame, folder: Path | None = None, resamples: int = 2000, seed: int = 0
) -> pd.DataFrame:
    """For each other team: accuracy of ``blend`` and of its model on the battles where the two
    predict the same winner and on the others, with a 95 % bootstrap interval over battles."""
    y = oof["y_true"].to_numpy()
    right = {model: (oof[model].to_numpy() > 0.5) == y for model in MODELS}
    rng = np.random.default_rng(seed)
    rows = []
    for other in ("second", "third"):
        same = (oof["blend"] > 0.5).to_numpy() == (oof[other] > 0.5).to_numpy()
        for group, members in (("agreement", same), ("disagreement", ~same)):
            battles = np.flatnonzero(members)
            samples = rng.integers(0, len(battles), (resamples, len(battles)), dtype=np.int32)
            for model in ("blend", other):
                hits = right[model][battles]
                low, high = np.quantile(hits[samples].mean(axis=1), [0.025, 0.975])
                rows.append({"other": other, "group": group, "model": model, "n": len(battles),
                             "share": len(battles) / len(y), "accuracy": hits.mean(),
                             "ci_low": low, "ci_high": high})  # fmt: skip
    table = pd.DataFrame(rows)
    save_table(table, folder, "disagreement")
    return table


def plot_disagreement(table: pd.DataFrame, folder: Path | None = None) -> Figure:
    """Accuracy where ``blend`` and another team agree and disagree, one panel per team."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    groups = ["agreement", "disagreement"]
    for ax, other in zip(axes, ("second", "third"), strict=True):
        part: Any = table[table["other"] == other]
        sizes = part.drop_duplicates("group").set_index("group")["n"]
        for k, model in enumerate(("blend", other)):
            bars = part[part["model"] == model].set_index("group").loc[groups]
            x = np.arange(2) + (k - 0.5) * 0.38
            error = [bars["accuracy"] - bars["ci_low"], bars["ci_high"] - bars["accuracy"]]
            whiskers = {"elinewidth": 0.8, "capsize": 2.5, "ecolor": LABEL}
            ax.bar(x, bars["accuracy"], width=0.35, color=COLORS[model], label=MODELS[model],
                   yerr=error, error_kw=whiskers)  # fmt: skip
            for xi, value, top in zip(x, bars["accuracy"], bars["ci_high"], strict=True):
                ax.annotate(f"{value:.1%}", (xi, top), xytext=(0, 3), ha="center", fontsize=8,
                            textcoords="offset points")  # fmt: skip
        ax.set_xticks(np.arange(2), [f"{g}\n({sizes[g]:,} battles)" for g in groups])
        ax.axhline(0.5, color=GRAY, linestyle=DASHED, linewidth=1)
        ax.set_title(f"blend and {MODELS[other]}", fontweight="normal")
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
        ax.legend(loc="upper right", ncol=2)
    axes[0].set_ylabel("Accuracy")
    fig.suptitle("Accuracy by agreement with blend", fontsize=11, fontweight="bold")
    return save(fig, folder, "disagreement")


def plot_loss_convergence(curve: pd.DataFrame, folder: Path | None = None) -> Figure:
    """``plots.plot_loss_convergence`` for the three teams (one validation fold)."""
    names = {**MODELS, "second": "2nd place (tactical view)"}
    with compared(names):
        return plots.plot_loss_convergence(curve, folder)


def plot_leaderboard(
    board: pd.DataFrame, metrics: pd.DataFrame, folder: Path | None = None
) -> Figure:
    """Per submission: accuracy in cross-validation on our folds, and on the public and private
    leaderboards (as reported by the teams; a missing score is not drawn)."""
    cv = metrics.set_index("model")["accuracy"]
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    labels, y = [], 0.0
    for team in dict.fromkeys(board["team"]):
        color = TEAM_COLORS[team]
        part = board[board["team"] == team]
        for model, public, private in zip(part["model"], part["public"], part["private"],
                                          strict=True):  # fmt: skip
            ax.plot([public, private], [y, y], color=color, linewidth=1, zorder=1)
            ax.scatter([cv[model]], [y], marker="D", s=30, facecolor="white", edgecolor=color)
            ax.scatter([public], [y], marker="o", s=34, facecolor="white", edgecolor=color)
            ax.scatter([private], [y], marker="o", s=34, color=color)
            labels.append((y, NAMES[model]))
            y += 1
        y += 0.6
    ax.set_yticks([position for position, _ in labels], [name for _, name in labels])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    for marker, face, name in (("D", "white", "cross-validation (our folds)"),
                               ("o", "white", "public"), ("o", GRAY, "private")):  # fmt: skip
        ax.scatter([], [], marker=marker, s=30, facecolor=face, edgecolor=GRAY, label=name)
    ax.xaxis.set_major_formatter(lambda value, _: f"{value:.1%}")
    ax.set_xlabel("Accuracy")
    ax.set_title("Cross-validation, public and private accuracy")
    ax.legend(loc="lower right")
    return save(fig, folder, "leaderboard")


def plot_swing(sim: pd.DataFrame, board: pd.DataFrame, folder: Path | None = None) -> Figure:
    """Private minus public accuracy over the simulated leaderboards (``blend``; the other
    teams' histograms are the same), with the swings reported on the real leaderboard."""
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    part = sim[sim["team"] == "blend"]
    swing = (part["private"] - part["public"]) * 100
    ax.hist(swing, bins=np.linspace(-5, 5, 41).tolist(), color=GRID, edgecolor="white",
            label=f"by chance (std {swing.std():.1f})")  # fmt: skip
    top = ax.get_ylim()[1]
    real: Any = board.dropna(subset=["public", "private"])
    for k, team in enumerate(dict.fromkeys(real["team"])):
        rows = real[real["team"] == team]
        values = (rows["private"] - rows["public"]) * 100
        height = np.full(len(values), top * (0.06 + 0.07 * k))
        ax.scatter(values, height, marker="v", s=36, color=TEAM_COLORS[team], zorder=3,
                   label=f"{TEAMS[team]}, real")  # fmt: skip
    ax.axvline(0, color=GRAY, linestyle=DASHED, linewidth=1)
    ax.set(xlabel="Private − public accuracy (points)", ylabel="Simulated leaderboards")
    ax.set_title("Private − public accuracy")
    ax.legend(loc="upper left")
    return save(fig, folder, "swing")


def gap_table(board: pd.DataFrame, metrics: pd.DataFrame, sim: pd.DataFrame) -> Any:
    """Gap between two teams (points of accuracy) in cross-validation and on the two
    leaderboards, for each team's best private submission, next to the standard deviation of
    such a gap over the simulated leaderboards."""
    cv = metrics.set_index("model")["accuracy"]
    best: Any = board.loc[board.groupby("team")["private"].idxmax()].set_index("team")
    simulated = {"ours": "blend", "second": "second", "third": "third"}
    wide = {p: sim.pivot(index="draw", columns="team", values=p) for p in ("public", "private")}
    rows = []
    for a, b in (("ours", "second"), ("ours", "third"), ("second", "third")):
        noise = {p: (wide[p][simulated[a]] - wide[p][simulated[b]]).std() for p in wide}
        rows.append({
            "Gap": f"{NAMES[best.loc[a, 'model']]} − {NAMES[best.loc[b, 'model']]}",
            "Cross-validation": (cv[best.loc[a, "model"]] - cv[best.loc[b, "model"]]) * 100,
            "Public": (best.loc[a, "public"] - best.loc[b, "public"]) * 100,
            "± chance": noise["public"] * 100,
            "Private": (best.loc[a, "private"] - best.loc[b, "private"]) * 100,
            "± chance ": noise["private"] * 100,
        })  # fmt: skip
    formats = {"Cross-validation": "{:+.2f}", "Public": "{:+.2f}", "Private": "{:+.2f}",
               "± chance": "{:.1f}", "± chance ": "{:.1f}"}  # fmt: skip
    return plots.table_style(pd.DataFrame(rows)).format(formats, na_rep="not available")


def metrics_table(metrics: pd.DataFrame, models: list[str]) -> Any:
    """Fold-mean metrics of ``models``, the best value of each column in bold."""
    with compared(NAMES):
        rows: Any = metrics.set_index("model").loc[models].reset_index()
        return plots.metrics_table(rows)


def paired_table(paired: pd.DataFrame, pairs: list[tuple[str, str]]) -> Any:
    """Paired differences a − b of ``pairs`` (points of accuracy, log loss, AUC) with p-values."""
    rows = []
    for a, b in pairs:
        part = paired[(paired["a"] == a) & (paired["b"] == b)].set_index("metric")
        rows.append({
            "Comparison": f"{NAMES[a]} − {NAMES[b]}",
            "Δ accuracy (points)": part.loc["accuracy", "delta"] * 100,
            "p ": part.loc["accuracy", "p"],
            "Δ log loss": part.loc["log_loss", "delta"], "p  ": part.loc["log_loss", "p"],
            "Δ ROC AUC": part.loc["roc_auc", "delta"], "p   ": part.loc["roc_auc", "p"],
        })  # fmt: skip
    formats = {"Δ accuracy (points)": "{:+.2f}", "Δ log loss": "{:+.4f}", "Δ ROC AUC": "{:+.4f}",
               "p ": "{:.3f}", "p  ": "{:.3f}", "p   ": "{:.3f}"}  # fmt: skip
    return plots.table_style(pd.DataFrame(rows)).format(formats)


def estimates_table(estimates: pd.DataFrame) -> Any:
    """Accuracy of each model under the three validation protocols."""
    wide: Any = estimates.pivot(index="model", columns="protocol", values="accuracy")
    order = list(dict.fromkeys(estimates["protocol"]))
    wide = wide.loc[list(dict.fromkeys(estimates["model"])), order].reset_index()
    wide["model"] = wide["model"].replace(NAMES)
    return plots.table_style(wide.rename(columns={"model": "Model"})).format(precision=4)


def leaderboard_table(board: pd.DataFrame, metrics: pd.DataFrame) -> Any:
    """Cross-validation accuracy on our folds next to the reported leaderboard scores."""
    cv = metrics.set_index("model")["accuracy"]
    table = pd.DataFrame({
        "Team": [TEAMS[team] for team in board["team"]],
        "Submission": [NAMES[model] for model in board["model"]],
        "Cross-validation": [cv[model] for model in board["model"]],
        "Public": board["public"],
        "Private": board["private"],
        "Private − public (points)": (board["private"] - board["public"]) * 100,
    })  # fmt: skip
    formats = {"Cross-validation": "{:.4f}", "Public": "{:.4f}", "Private": "{:.4f}",
               "Private − public (points)": "{:+.2f}"}  # fmt: skip
    return plots.table_style(table).format(formats, na_rep="not available")


def draw(data: Path, figures: Path) -> None:
    """Every figure of the comparison (and the tables computed from ``oof.csv`` alone)."""
    apple_style()
    oof, metrics = pd.read_csv(data / "oof.csv"), pd.read_csv(data / "metrics.csv")
    board = pd.read_csv(data / "leaderboard.csv")
    plot_roc(oof, metrics, figures)
    plot_confidence(oof, data, figures)
    plot_calibration(oof, data, figures)
    plot_disagreement(disagreement_table(oof, data), figures)
    plot_loss_convergence(pd.read_csv(data / "loss_curve.csv"), figures)
    plot_leaderboard(board, metrics, figures)
    plot_swing(pd.read_csv(data / "split_simulation.csv"), board, figures)
    plt.close("all")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="figures of the comparison with other teams")
    parser.add_argument("--data", type=Path, default=HERE / "data")
    parser.add_argument("--figures", type=Path, default=HERE / "figures")
    arguments = parser.parse_args()
    draw(arguments.data, arguments.figures)
