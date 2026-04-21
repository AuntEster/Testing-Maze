from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple, Set
from collections import defaultdict
import os

import cv2
import numpy as np


class Action(Enum):
    MOVE_UP = 0
    MOVE_DOWN = 1
    MOVE_LEFT = 2
    MOVE_RIGHT = 3
    WAIT = 4


@dataclass
class TurnResult:
    wall_hits: int = 0
    current_position: Tuple[int, int] = (0, 0)
    is_dead: bool = False
    is_confused: bool = False
    is_goal_reached: bool = False
    teleported: bool = False
    actions_executed: int = 0
    positions_visited: List[Tuple[int, int]] = field(default_factory=list)
    last_event: str = ""


class TemplateMatcher:
    """
    Uses OpenCV template matching on the 14x14 interior of each cell.
    Template filenames are expected to start with labels like:
      - death_pit
      - confusion
      - teleport_orange
      - teleport_green
      - teleport_purple
      - teleport_red
    """

    def __init__(self, templates_dir: str):
        self.templates_dir = templates_dir
        self.templates: Dict[str, List[np.ndarray]] = defaultdict(list)
        self._load_templates()

    def _load_templates(self) -> None:
        if not os.path.isdir(self.templates_dir):
            raise FileNotFoundError(f"Templates folder not found: {self.templates_dir}")

        for fname in os.listdir(self.templates_dir):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            path = os.path.join(self.templates_dir, fname)
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                continue

            label = fname.split("_r")[0]
            self.templates[label].append(img)

        if not self.templates:
            raise RuntimeError(f"No template images found in: {self.templates_dir}")

    def detect_hazards(
        self,
        img_bgr: np.ndarray,
        maze_height_cells: int,
        maze_width_cells: int,
        cell_size: int = 16,
        wall_thickness: int = 2,
        threshold: float = 0.60,
    ) -> Dict[str, Set[Tuple[int, int]]]:
        inner = cell_size - 2 * wall_thickness
        detected: Dict[str, Set[Tuple[int, int]]] = defaultdict(set)

        for r in range(maze_height_cells):
            for c in range(maze_width_cells):
                x0 = c * cell_size + wall_thickness
                y0 = r * cell_size + wall_thickness
                patch = img_bgr[y0:y0 + inner, x0:x0 + inner]

                if patch.shape[0] != inner or patch.shape[1] != inner:
                    continue

                best_label = None
                best_score = -1.0

                for label, templates in self.templates.items():
                    for tmpl in templates:
                        if tmpl.shape[:2] != patch.shape[:2]:
                            tmpl = cv2.resize(tmpl, (patch.shape[1], patch.shape[0]))

                        res = cv2.matchTemplate(patch, tmpl, cv2.TM_CCOEFF_NORMED)
                        score = float(res[0, 0])

                        if score > best_score:
                            best_score = score
                            best_label = label

                if best_label is not None and best_score >= threshold:
                    detected[best_label].add((r, c))

        return detected


class MazeLoader:
    CELL_SIZE = 16
    WALL_THICKNESS = 2
    GRID_SIZE = 64

    def __init__(self, maze_image_path: str, templates_dir: str):
        self.maze_image_path = maze_image_path
        self.templates_dir = templates_dir

        self.img_bgr = cv2.imread(maze_image_path, cv2.IMREAD_COLOR)
        if self.img_bgr is None:
            raise FileNotFoundError(f"Could not read maze image: {maze_image_path}")

        self.img_rgb = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2RGB)
        self.h, self.w = self.img_rgb.shape[:2]

        self.maze_height_cells = self.GRID_SIZE
        self.maze_width_cells = self.GRID_SIZE

        self.matcher = TemplateMatcher(templates_dir)

        # Passability image/grid
        self.maze_array = self._build_passability_array()
        self.grid = self._build_grid()

        # Project-specific known start/goal from border openings
        self.start_cell = (63, 31)
        self.goal_cell = (0, 31)
        self.start_pos = self.cell_to_pixel(self.start_cell[0], self.start_cell[1])
        self.goal_pos = self.cell_to_pixel(self.goal_cell[0], self.goal_cell[1])

        # Hazard containers
        self.hazard_cells_by_label: Dict[str, Set[Tuple[int, int]]] = {}
        self.death_pits: List[Tuple[int, int]] = []
        self.confusion_pads: List[Tuple[int, int]] = []
        self.teleport_orange: List[Tuple[int, int]] = []
        self.teleport_green: List[Tuple[int, int]] = []
        self.teleport_purple: List[Tuple[int, int]] = []
        self.teleport_red: List[Tuple[int, int]] = []

    def cell_to_pixel(self, row: int, col: int) -> Tuple[int, int]:
        y = row * self.CELL_SIZE + self.CELL_SIZE // 2
        x = col * self.CELL_SIZE + self.CELL_SIZE // 2
        y = min(y, self.h - 1)
        x = min(x, self.w - 1)
        return (x, y)

    def pixel_to_cell(self, x: int, y: int) -> Tuple[int, int]:
        return (y // self.CELL_SIZE, x // self.CELL_SIZE)

    def _build_passability_array(self) -> np.ndarray:
        """
        Build a pixel-level boolean array where True means open/passable.
        This assumes dark pixels are walls and bright pixels are corridors/interiors.
        """
        gray = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2GRAY)
        # In these mazes, corridors/cell interiors are bright while walls are dark.
        _, binary = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)
        return (binary > 0)

    def _build_grid(self) -> List[List[bool]]:
        grid = [[False] * self.maze_width_cells for _ in range(self.maze_height_cells)]
        for r in range(self.maze_height_cells):
            for c in range(self.maze_width_cells):
                x, y = self.cell_to_pixel(r, c)
                grid[r][c] = bool(self.maze_array[y, x])
        return grid

    def detect_hazards(self, threshold: float = 0.60) -> None:
        """
        Only detect hazards here.
        Start and goal are fixed and should not be inferred from the image.
        """
        self.hazard_cells_by_label = self.matcher.detect_hazards(
            self.img_bgr,
            self.maze_height_cells,
            self.maze_width_cells,
            cell_size=self.CELL_SIZE,
            wall_thickness=self.WALL_THICKNESS,
            threshold=threshold,
        )

        self.death_pits = sorted(self.hazard_cells_by_label.get("death_pit", []))
        self.confusion_pads = sorted(self.hazard_cells_by_label.get("confusion", []))
        self.teleport_orange = sorted(self.hazard_cells_by_label.get("teleport_orange", []))
        self.teleport_green = sorted(self.hazard_cells_by_label.get("teleport_green", []))
        self.teleport_purple = sorted(self.hazard_cells_by_label.get("teleport_purple", []))
        self.teleport_red = sorted(self.hazard_cells_by_label.get("teleport_red", []))


class MazeEnvironment:
    CELL_SIZE = 16

    DELTAS = {
        Action.MOVE_UP: (-1, 0),
        Action.MOVE_DOWN: (1, 0),
        Action.MOVE_LEFT: (0, -1),
        Action.MOVE_RIGHT: (0, 1),
        Action.WAIT: (0, 0),
    }

    INVERT = {
        Action.MOVE_UP: Action.MOVE_DOWN,
        Action.MOVE_DOWN: Action.MOVE_UP,
        Action.MOVE_LEFT: Action.MOVE_RIGHT,
        Action.MOVE_RIGHT: Action.MOVE_LEFT,
        Action.WAIT: Action.WAIT,
    }

    def __init__(self, maze_image_path: str, templates_dir: str = "templates", rotate_fire: bool = False):
        self.loader = MazeLoader(maze_image_path, templates_dir=templates_dir)
        self.loader.detect_hazards()

        self.grid = [row[:] for row in self.loader.grid]
        self.start_cell = self.loader.start_cell
        self.goal_cell = self.loader.goal_cell

        self.death_pits = set(map(tuple, self.loader.death_pits))
        self.initial_death_pits = set(self.death_pits)

        self.confusion_pads = set(map(tuple, self.loader.confusion_pads))

        self.fire_clusters: List[List[Tuple[int, int]]] = self.group_clusters(self.death_pits)
        self.initial_fire_clusters = [list(cluster) for cluster in self.fire_clusters]

        # mark hazards as passable cells (they are enterable, just dangerous)
        for r, c in self.death_pits | self.confusion_pads:
            self.grid[r][c] = True
        for pads in (
            self.loader.teleport_orange,
            self.loader.teleport_green,
            self.loader.teleport_purple,
            self.loader.teleport_red,
        ):
            for r, c in pads:
                self.grid[r][c] = True

        self.teleport_map: Dict[Tuple[int, int], Tuple[int, int]] = {}
        for group in (
            self.loader.teleport_orange,
            self.loader.teleport_green,
            self.loader.teleport_purple,
            self.loader.teleport_red,
        ):
            pads = [tuple(p) for p in group]
            for i in range(0, len(pads) - 1, 2):
                self.teleport_map[pads[i]] = pads[i + 1]
                self.teleport_map[pads[i + 1]] = pads[i]
            if len(pads) % 2 == 1:
                self.teleport_map[pads[-1]] = pads[-1]

        self.rotate_fire_enabled = rotate_fire
        self.freeze_fire = False

        self.agent_pos: Tuple[int, int] = self.start_cell
        self.turn_count = 0
        self.death_count = 0
        self.confused_count = 0
        self.teleport_count = 0
        self.confused_turns_left = 0
        self.cells_explored: Set[Tuple[int, int]] = set()
        self.episode_active = True

    def group_clusters(self, cells: Set[Tuple[int, int]], max_gap: int = 1) -> List[List[Tuple[int, int]]]:
        remaining = set(cells)
        clusters = []

        while remaining:
            seed = next(iter(remaining))
            cluster = []
            queue = [seed]
            remaining.discard(seed)

            while queue:
                cell = queue.pop()
                cluster.append(cell)

                for other in list(remaining):
                    if abs(other[0] - cell[0]) <= max_gap and abs(other[1] - cell[1]) <= max_gap:
                        remaining.discard(other)
                        queue.append(other)

            clusters.append(sorted(cluster))

        return clusters

    def cluster_orientation_and_pivot(self, cluster: List[Tuple[int, int]]) -> Tuple[str, Tuple[int, int]]:
        rows = [r for r, _ in cluster]
        cols = [c for _, c in cluster]

        min_r, max_r = min(rows), max(rows)
        min_c, max_c = min(cols), max(cols)

        row_counts = defaultdict(int)
        col_counts = defaultdict(int)
        for r, c in cluster:
            row_counts[r] += 1
            col_counts[c] += 1

        width = max_c - min_c + 1
        height = max_r - min_r + 1

        if width >= height:
            if row_counts[max_r] <= row_counts[min_r]:
                pivot_candidates = [cell for cell in cluster if cell[0] == max_r]
                pivot = min(pivot_candidates, key=lambda cell: abs(cell[1] - sum(cols) / len(cols)))
                return "v", pivot
            pivot_candidates = [cell for cell in cluster if cell[0] == min_r]
            pivot = min(pivot_candidates, key=lambda cell: abs(cell[1] - sum(cols) / len(cols)))
            return "^", pivot

        if col_counts[min_c] <= col_counts[max_c]:
            pivot_candidates = [cell for cell in cluster if cell[1] == min_c]
            pivot = min(pivot_candidates, key=lambda cell: abs(cell[0] - sum(rows) / len(rows)))
            return "<", pivot

        pivot_candidates = [cell for cell in cluster if cell[1] == max_c]
        pivot = min(pivot_candidates, key=lambda cell: abs(cell[0] - sum(rows) / len(rows)))
        return ">", pivot

    def rotate_fire_cluster(self, cluster: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        h = self.loader.maze_height_cells
        w = self.loader.maze_width_cells

        _, pivot = self.cluster_orientation_and_pivot(cluster)
        pr, pc = pivot

        rotated = []
        seen = set()

        for r, c in cluster:
            dr = r - pr
            dc = c - pc
            nr = pr + dc
            nc = pc - dr

            if 0 <= nr < h and 0 <= nc < w:
                cell = (nr, nc)
                if cell not in seen:
                    seen.add(cell)
                    rotated.append(cell)

        return sorted(rotated)

    def rotate_fire_clusters(self) -> None:
        if self.freeze_fire:
            return

        # restore old death pit cells back to original passability
        for r, c in self.death_pits:
            self.grid[r][c] = self.loader.grid[r][c]

        new_clusters = []
        new_death_pits = set()

        for cluster in self.fire_clusters:
            rotated = self.rotate_fire_cluster(cluster)
            new_clusters.append(rotated)
            new_death_pits.update(rotated)

        self.fire_clusters = new_clusters
        self.death_pits = new_death_pits

        for r, c in self.death_pits:
            self.grid[r][c] = True

    def reset(self) -> Tuple[int, int]:
        self.agent_pos = self.start_cell
        self.turn_count = 0
        self.death_count = 0
        self.confused_count = 0
        self.teleport_count = 0
        self.confused_turns_left = 0
        self.cells_explored = set()
        self.episode_active = True

        self.grid = [row[:] for row in self.loader.grid]
        self.death_pits = set(self.initial_death_pits)
        self.fire_clusters = [list(cluster) for cluster in self.initial_fire_clusters]

        for r, c in self.death_pits | self.confusion_pads:
            self.grid[r][c] = True
        for cell in self.teleport_map:
            self.grid[cell[0]][cell[1]] = True

        return self.agent_pos

    def is_passable(self, row: int, col: int) -> bool:
        if not (0 <= row < self.loader.maze_height_cells and 0 <= col < self.loader.maze_width_cells):
            return False
        return self.grid[row][col]

    def is_move_passable(
        self,
        from_r: int,
        from_c: int,
        to_r: int,
        to_c: int,
        from_hazard: bool = False,
        to_hazard: bool = False,
    ) -> bool:
        if not (0 <= to_r < self.loader.maze_height_cells and 0 <= to_c < self.loader.maze_width_cells):
            return False

        if not self.grid[from_r][from_c]:
            return False
        if not self.grid[to_r][to_c]:
            return False

        x1, y1 = self.loader.cell_to_pixel(from_r, from_c)
        x2, y2 = self.loader.cell_to_pixel(to_r, to_c)

        mx = (x1 + x2) // 2
        my = (y1 + y2) // 2

        return bool(self.loader.maze_array[my, mx])

    def step(self, actions: List[Action]) -> TurnResult:
        if not actions or len(actions) > 5:
            raise ValueError("Must submit 1-5 actions per turn.")

        result = TurnResult(current_position=self.agent_pos)
        currently_confused = self.confused_turns_left > 0

        for raw_action in actions:
            action = self.INVERT[raw_action] if currently_confused else raw_action

            if action == Action.WAIT:
                result.actions_executed += 1
                continue

            dr, dc = self.DELTAS[action]
            r, c = self.agent_pos
            nr, nc = r + dr, c + dc

            src_hazard = ((r, c) in self.death_pits or (r, c) in self.confusion_pads or (r, c) in self.teleport_map)
            dst_hazard = ((nr, nc) in self.death_pits or (nr, nc) in self.confusion_pads or (nr, nc) in self.teleport_map)

            if not self.is_move_passable(r, c, nr, nc, from_hazard=src_hazard, to_hazard=dst_hazard):
                result.wall_hits += 1
                result.actions_executed += 1
                continue

            self.agent_pos = (nr, nc)
            self.cells_explored.add(self.agent_pos)
            result.actions_executed += 1
            result.positions_visited.append(self.agent_pos)

            if self.agent_pos in self.teleport_map:
                src = self.agent_pos
                dest = self.teleport_map[self.agent_pos]
                self.agent_pos = dest
                result.teleported = True
                result.current_position = self.agent_pos
                result.last_event = f"TELEPORT {src}->{dest}"
                self.teleport_count += 1

            if self.agent_pos in self.confusion_pads:
                result.is_confused = True
                self.confused_turns_left = 2
                self.confused_count += 1
                currently_confused = True
                result.last_event = f"CONFUSED at {self.agent_pos}"

            if self.agent_pos in self.death_pits:
                result.is_dead = True
                self.death_count += 1
                result.current_position = self.agent_pos
                result.last_event = f"DEATH at {self.agent_pos}"
                self.agent_pos = self.start_cell
                break

            if self.agent_pos == self.goal_cell:
                result.is_goal_reached = True
                result.current_position = self.agent_pos
                self.episode_active = False
                break

        if self.confused_turns_left > 0:
            self.confused_turns_left -= 1

        self.turn_count += 1

        if self.rotate_fire_enabled and self.turn_count % 5 == 0:
            self.rotate_fire_clusters()

            if not result.is_dead and self.agent_pos in self.death_pits:
                result.is_dead = True
                self.death_count += 1
                result.current_position = self.agent_pos
                result.last_event = f"DEATH by rotating fire at {self.agent_pos}"
                self.agent_pos = self.start_cell

        if not result.is_dead:
            result.current_position = self.agent_pos

        return result

    def get_episode_stats(self) -> dict:
        return {
            "turns_taken": self.turn_count,
            "deaths": self.death_count,
            "confused": self.confused_count,
            "cells_explored": len(self.cells_explored),
            "goal_reached": not self.episode_active,
        }