"""The closing animation of the comparison notebook: a stick figure leaves home, touches grass,
lies down on it and is flattened by a Snorlax.

Only numpy and matplotlib (the GIF is written by Pillow, which matplotlib installs).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.axes import Axes
from matplotlib.patches import Circle, Ellipse, Polygon, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plots import GRAY, INDIGO, LABEL  # noqa: E402

GREEN, ORANGE = "#34C759", "#FF9500"
TEAL, CREAM, DARK = "#2F6F84", "#F1E4C3", "#1D3B47"  # Snorlax
DOOR, STOP = 2.35, 7.55  # where the figure starts and where it stops, in front of the grass
WAIT, WALK, CROUCH, HOLD = 5, 40, 12, 20  # frames of each part
LIE, REST, FALL, LANDED = 12, 8, 8, 30
BLADES = np.linspace(8.5, 10.6, 15)
SNORLAX_X, SNORLAX_TOP = 9.4, 5.3  # where it lands and the height it falls from
Joints = dict[str, np.ndarray]


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


def standing(x: float, swing: float, crouch: float) -> Joints:
    """The joints with the hip above ``x``: legs and arms at angle ``swing`` (walking), body
    bent forward and front hand towards the grass as ``crouch`` goes from 0 to 1."""
    bend = 1.1 * crouch
    hip = np.array([x, 1.0 - 0.4 * crouch])
    along = np.array([np.sin(bend), np.cos(bend)])
    shoulder, neck = hip + 0.9 * along, hip + 1.1 * along
    joints = {"hip": hip, "shoulder": shoulder, "neck": neck, "head": neck + 0.3 * along}
    for side in (-1, 1):
        foot = np.array([x + side * (np.sin(swing) * (1 - crouch) + 0.22 * crouch), 0.0])
        joints[f"foot{side}"] = foot
        joints[f"knee{side}"] = (hip + foot) / 2 + np.array([0.3 * crouch, 0.0])
        hanging = shoulder + 0.75 * np.array([-side * np.sin(swing), -np.cos(swing)])
        reach = np.array([8.9, 0.36]) - shoulder
        towards = shoulder + 0.85 * reach / np.linalg.norm(reach)
        share = crouch if side == 1 else 0.3 * crouch
        joints[f"hand{side}"] = hanging + (towards - hanging) * share
    return joints


def lying(kick: float) -> Joints:
    """The joints lying on the back across the grass, head to the right, one hand behind the
    head; ``kick`` lifts the feet in turn (the legs left out from under the Snorlax)."""
    lift = 0.12 * kick
    joints = {"hip": [8.35, 0.3], "shoulder": [9.25, 0.3], "neck": [9.45, 0.3],
              "head": [9.75, 0.32], "knee-1": [7.8, 0.3 + lift], "foot-1": [7.25, 0.05 + lift],
              "knee1": [7.9, 0.6 - lift], "foot1": [7.4, 0.05 + 0.5 * lift],
              "hand-1": [8.85, 0.5], "hand1": [9.9, 0.75]}  # fmt: skip
    return {name: np.array(point, dtype=float) for name, point in joints.items()}


def stick_figure(ax: Axes, joints: Joints) -> None:
    """Draw the figure from its joints."""
    line = {"color": INDIGO, "linewidth": 2}
    for side in (-1, 1):
        ax.plot(*zip(joints["hip"], joints[f"knee{side}"], joints[f"foot{side}"], strict=True),
                **line)  # fmt: skip
        ax.plot(*zip(joints["shoulder"], joints[f"hand{side}"], strict=True), **line)
    ax.plot(*zip(joints["hip"], joints["neck"], strict=True), **line)
    ax.add_patch(Circle(tuple(joints["head"]), 0.28, facecolor="white", edgecolor=INDIGO,
                            linewidth=2, zorder=3))  # fmt: skip


def snorlax(ax: Axes, bottom: float, squash: float) -> None:
    """A sleeping Snorlax with its feet at height ``bottom``, flattened by ``squash`` < 1."""
    x, s = SNORLAX_X, squash

    def oval(cx: float, cy: float, w: float, h: float, color: str, z: float) -> None:
        ax.add_patch(Ellipse((x + cx, bottom + cy * s), w / s**0.5, h * s, color=color, zorder=z))

    for side in (-1, 1):
        ear = [(0.35, 2.55), (0.72, 3.0), (0.72, 2.4)]
        ax.add_patch(Polygon([(x + side * dx, bottom + dy * s) for dx, dy in ear], color=TEAL,
                             zorder=4))  # fmt: skip
        oval(side * 1.22, 1.15, 0.55, 0.95, TEAL, 4)
        oval(side * 0.8, 0.2, 0.62, 0.42, CREAM, 6)
        oval(side * 0.8, 0.2, 0.25, 0.18, "#9C7A54", 6.5)
        ax.plot([x + side * 0.18, x + side * 0.42], [bottom + 2.2 * s] * 2, color=DARK,
                linewidth=1.6, zorder=7)  # fmt: skip
    oval(0, 1.05, 2.6, 2.1, TEAL, 4.5)
    oval(0, 0.9, 1.95, 1.55, CREAM, 5)
    oval(0, 2.2, 1.55, 1.0, TEAL, 5.5)
    oval(0, 2.08, 1.15, 0.66, CREAM, 6)
    ax.plot([x - 0.18, x + 0.18], [bottom + 1.93 * s] * 2, color=DARK, linewidth=1.4, zorder=7)


def falling_snorlax(ax: Axes, since: int) -> None:
    """The Snorlax ``since`` frames after it starts falling: shadow, fall, bounce and Zzz."""
    drop = min(since / FALL, 1.0)
    ax.add_patch(Ellipse((SNORLAX_X, 0.0), 2.8 * (0.3 + 0.7 * drop), 0.2, color=GRAY,
                         alpha=0.5, zorder=2))  # fmt: skip
    landed = since - FALL
    squash = 1 - 0.25 * np.exp(-0.35 * landed) * np.cos(0.9 * landed) if landed >= 0 else 1.0
    snorlax(ax, SNORLAX_TOP * (1 - drop**2), squash)
    if landed > 10:
        for k, letter in enumerate("zzZ"):
            rise = (landed - 10 - 4 * k) * 0.06
            if rise > 0:
                ax.text(SNORLAX_X - 0.9 - 0.35 * k, 2.7 + rise, letter, fontsize=8 + 2 * k,
                        color=DARK, zorder=8)  # fmt: skip


def figure_joints(step: int) -> Joints:
    """Where the figure is ``step`` frames after the door opens."""
    walked = min(step / WALK, 1.0)
    crouch = float(np.clip((step - WALK) / CROUCH, 0, 1))
    swing = 0.45 * np.sin(step * 0.75) if walked < 1 else 0.0
    joints = standing(DOOR + (STOP - DOOR) * walked, swing, crouch)
    lie = float(np.clip((step - WALK - CROUCH - HOLD) / LIE, 0, 1))
    if lie == 0:
        return joints
    lie = lie * lie * (3 - 2 * lie)  # smooth start and stop
    landed = step - WALK - CROUCH - HOLD - LIE - REST - FALL
    kick = np.sin(1.6 * landed) if landed >= 0 else 0.0
    return {name: (1 - lie) * point + lie * lying(kick)[name] for name, point in joints.items()}


def draw_frame(ax: Axes, frame: int) -> None:
    """One frame: the door opens, the figure walks to the grass, touches it, lies down on it,
    and a Snorlax falls on it."""
    step = frame - WAIT
    held = max(step - WALK - CROUCH, 0)
    since = step - WALK - CROUCH - HOLD - LIE - REST
    ax.clear()
    ax.set(xlim=(0, 11.5), ylim=(-0.3, 5.0), aspect="equal")
    ax.axis("off")
    scenery(ax, step >= 0, np.sin(held * 0.9) if 0 < held < HOLD else 0.0)
    if step >= 0:
        stick_figure(ax, figure_joints(step))
    if since >= 0:
        falling_snorlax(ax, since)
    if since >= FALL:
        ax.text(5.2, 4.1, "Snorlax used Body Slam!", ha="center", fontsize=12, color=LABEL)
    elif held > 3:
        ax.text(6.3, 4.1, "touching grass", ha="center", fontsize=12, color=LABEL)


def touch_grass(folder: Path) -> Path:
    """Write ``touch_grass.gif`` in ``folder`` and return its path."""
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    fig.subplots_adjust(0, 0, 1, 1)
    frames = WAIT + WALK + CROUCH + HOLD + LIE + REST + FALL + LANDED
    animation = FuncAnimation(fig, lambda frame: draw_frame(ax, frame), frames=frames)
    folder.mkdir(parents=True, exist_ok=True)
    # the notebook's style saves figures at a high resolution: far too heavy for 135 frames
    animation.save(folder / "touch_grass.gif", writer=PillowWriter(fps=14), dpi=80)
    plt.close(fig)
    return folder / "touch_grass.gif"
