"""The closing animation of the comparison notebook: a stick figure leaves home and touches grass.

Only numpy and matplotlib (the GIF is written by Pillow, which matplotlib installs).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.axes import Axes
from matplotlib.patches import Circle, Polygon, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plots import GRAY, INDIGO, LABEL  # noqa: E402

GREEN, ORANGE = "#34C759", "#FF9500"
DOOR, STOP = 2.35, 7.55  # where the figure starts and where it stops, in front of the grass
WAIT, WALK, CROUCH, HOLD = 5, 40, 12, 20  # frames of each part
BLADES = np.linspace(8.5, 10.6, 15)


def scenery(ax: Axes, door_open: bool, touched: float) -> None:
    """Ground, sun, house (the door dark once open) and grass (it sways when ``touched`` > 0)."""
    ax.plot([0, 11.5], [0, 0], color=GRAY, linewidth=1)
    ax.add_patch(Circle((10.4, 4.3), 0.38, color=ORANGE))
    ax.add_patch(Polygon([(0.5, 0), (0.5, 3.2), (3.2, 3.2), (3.2, 0)], fill=False,
                             edgecolor=LABEL, linewidth=1.6))  # fmt: skip
    ax.plot([0.25, 1.85, 3.45], [3.2, 4.4, 3.2], color=LABEL, linewidth=1.6)
    ax.add_patch(Rectangle((1.9, 0), 0.9, 2.85, edgecolor=LABEL, linewidth=1.2,
                               facecolor=LABEL if door_open else "white"))  # fmt: skip
    ax.add_patch(Rectangle((0.8, 1.5), 0.7, 0.7, fill=False, edgecolor=LABEL, linewidth=1.2))
    for k, x in enumerate(BLADES):
        height = 0.32 + 0.1 * np.sin(3 * k)
        lean = 0.1 * np.sin(k) + 0.12 * touched * np.exp(-((x - 8.9) ** 2))
        ax.plot([x, x + lean], [0, height], color=GREEN, linewidth=1.6)


def stick_figure(ax: Axes, x: float, swing: float, crouch: float) -> None:
    """The figure with its hip above ``x``: legs and arms at angle ``swing`` (walking), body
    bent forward and front hand towards the grass as ``crouch`` goes from 0 to 1."""
    bend = 1.1 * crouch
    hip = np.array([x, 1.0 - 0.4 * crouch])
    along = np.array([np.sin(bend), np.cos(bend)])
    shoulder, neck = hip + 0.9 * along, hip + 1.1 * along
    for side in (-1, 1):
        foot = np.array([x + side * (np.sin(swing) * (1 - crouch) + 0.22 * crouch), 0.0])
        knee = (hip + foot) / 2 + np.array([0.3 * crouch, 0.0])
        ax.plot(*zip(hip, knee, foot, strict=True), color=INDIGO, linewidth=2)
        hanging = shoulder + 0.75 * np.array([-side * np.sin(swing), -np.cos(swing)])
        reach = np.array([8.9, 0.36]) - shoulder
        towards = shoulder + 0.85 * reach / np.linalg.norm(reach)
        hand = hanging + (towards - hanging) * (crouch if side == 1 else 0.3 * crouch)
        ax.plot(*zip(shoulder, hand, strict=True), color=INDIGO, linewidth=2)
    ax.plot(*zip(hip, neck, strict=True), color=INDIGO, linewidth=2)
    ax.add_patch(Circle(tuple(neck + 0.3 * along), 0.28, facecolor="white", edgecolor=INDIGO,
                            linewidth=2, zorder=3))  # fmt: skip


def draw_frame(ax: Axes, frame: int) -> None:
    """One frame: the door opens, the figure walks to the grass, crouches and touches it."""
    step = frame - WAIT
    walked = min(step / WALK, 1.0)
    crouch = float(np.clip((step - WALK) / CROUCH, 0, 1))
    held = max(step - WALK - CROUCH, 0)
    ax.clear()
    ax.set(xlim=(0, 11.5), ylim=(-0.3, 5.0), aspect="equal")
    ax.axis("off")
    scenery(ax, step >= 0, np.sin(held * 0.9) if held else 0.0)
    if step >= 0:
        swing = 0.45 * np.sin(step * 0.75) if walked < 1 else 0.0
        stick_figure(ax, DOOR + (STOP - DOOR) * walked, swing, crouch)
    if held > 3:
        ax.text(6.3, 4.1, "touching grass", ha="center", fontsize=12, color=LABEL)


def touch_grass(folder: Path) -> Path:
    """Write ``touch_grass.gif`` in ``folder`` and return its path."""
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    fig.subplots_adjust(0, 0, 1, 1)
    animation = FuncAnimation(fig, lambda frame: draw_frame(ax, frame),
                              frames=WAIT + WALK + CROUCH + HOLD)  # fmt: skip
    folder.mkdir(parents=True, exist_ok=True)
    # the notebook's style saves figures at a high resolution: far too heavy for 77 frames
    animation.save(folder / "touch_grass.gif", writer=PillowWriter(fps=14), dpi=80)
    plt.close(fig)
    return folder / "touch_grass.gif"
