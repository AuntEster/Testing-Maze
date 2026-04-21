
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


_DOT_R = 3
_PATH_COLOR = (33, 150, 243)
_AGENT_COLOR = (255, 255, 255)
_GOAL_COLOR = (0, 230, 118)
_START_COLOR = (255, 140, 0)
_HAZARD_COLOR = {
    "death": (180, 0, 0),
    "death_rotating": (255, 140, 30),
    "confusion": (253, 216, 53),
    "teleport": (171, 71, 188),
    "teleport_dest": (171, 71, 188),
    "goal": (0, 230, 118),
}


def _cell_center(env, row: int, col: int) -> Tuple[int, int]:
    cell = env.CELL_SIZE
    x = min(col * cell + cell // 2, env.loader.w - 1)
    y = min(row * cell + cell // 2, env.loader.h - 1)
    return x, y


def _dot(draw: ImageDraw.ImageDraw, xy: Tuple[int, int], color: Tuple[int, int, int], r: int = _DOT_R):
    x, y = xy
    draw.ellipse([x - r, y - r, x + r, y + r], fill=color)


def _agent_dot(draw: ImageDraw.ImageDraw, xy: Tuple[int, int]):
    _dot(draw, xy, (0, 0, 0), r=6)
    _dot(draw, xy, _AGENT_COLOR, r=4)


def _draw_path(draw: ImageDraw.ImageDraw, env, path: List[Tuple[int, int]], color: Tuple[int, int, int] = _PATH_COLOR):
    if len(path) < 2:
        if path:
            _dot(draw, _cell_center(env, *path[0]), color)
        return
    points = [_cell_center(env, r, c) for r, c in path]
    draw.line(points, fill=color, width=2)


class LiveVisualizer:
    def __init__(self, env, title: str = "Fresh Maze Agent", update_every: int = 1, delay: float = 0.05):
        self.env = env
        self.update_every = update_every
        self.delay = delay
        self.step_no = 0
        self.painted = Image.fromarray(env.loader.img_rgb.copy())
        self.last_path_index = 0

        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.im = self.ax.imshow(np.array(self.painted), origin="upper")
        self.ax.axis("off")
        self.stats_text = self.ax.text(
            0.01,
            0.01,
            "",
            transform=self.ax.transAxes,
            fontsize=8.5,
            color="white",
            va="bottom",
            bbox=dict(boxstyle="round", facecolor="black", alpha=0.65),
        )
        self.ax.set_title(title)
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.05)

    def reset_episode(self):
        self.painted = Image.fromarray(self.env.loader.img_rgb.copy())
        self.last_path_index = 0

    def update(
        self,
        known_cells: Dict[Tuple[int, int], str],
        current_pos: Optional[Tuple[int, int]],
        path: List[Tuple[int, int]],
        episode: int,
        turn: int,
        start_pos: Optional[Tuple[int, int]],
        goal_pos: Optional[Tuple[int, int]],
        extra_stats: Optional[Dict[str, object]] = None,
    ):
        self.step_no += 1
        if self.step_no % self.update_every != 0:
            return

        draw = ImageDraw.Draw(self.painted)
        new_segment = path[max(0, self.last_path_index - 1):]
        _draw_path(draw, self.env, new_segment)
        self.last_path_index = len(path)

        for (r, c), label in known_cells.items():
            color = _HAZARD_COLOR.get(label)
            if color:
                _dot(draw, _cell_center(self.env, r, c), color, r=_DOT_R + 1)

        if start_pos:
            _dot(draw, _cell_center(self.env, *start_pos), _START_COLOR, r=5)
        if goal_pos:
            _dot(draw, _cell_center(self.env, *goal_pos), _GOAL_COLOR, r=6)

        display = self.painted.copy()
        disp_draw = ImageDraw.Draw(display)

        rotating = set()
        for cluster in self.env.fire_clusters:
            rotating.update(cluster)

        for r, c in self.env.death_pits:
            color = _HAZARD_COLOR["death_rotating"] if (r, c) in rotating else _HAZARD_COLOR["death"]
            _dot(disp_draw, _cell_center(self.env, r, c), color, r=_DOT_R + 2)

        if current_pos:
            _agent_dot(disp_draw, _cell_center(self.env, *current_pos))

        self.im.set_data(np.array(display))
        self.ax.set_title(f"Episode {episode} | Turn {turn}")
        if extra_stats:
            self.stats_text.set_text("\n".join(f"{k}: {v}" for k, v in extra_stats.items()))
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(self.delay)

    def close(self):
        plt.ioff()
        plt.close(self.fig)
