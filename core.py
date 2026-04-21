
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from collections import defaultdict, deque

import cv2
import numpy as np


Cell = Tuple[int, int]


class Action(Enum):
    MOVE_UP = 0
    MOVE_DOWN = 1
    MOVE_LEFT = 2
    MOVE_RIGHT = 3
    WAIT = 4


@dataclass
class TurnResult:
    wall_hits: int = 0
    current_position: Cell = (0, 0)
    is_dead: bool = False
    is_confused: bool = False
    is_goal_reached: bool = False
    teleported: bool = False
    actions_executed: int = 0
    positions_visited: List[Cell] = field(default_factory=list)
    last_event: str = ""


class TemplateMatcher:
    """
    OpenCV template matcher over the interior of each 16x16 maze cell.
    Templates are expected to come from the existing template-cropping flow.
    """

    CELL_SIZE = 16
    WALL_BORDER = 2
    INNER = CELL_SIZE - 2 * WALL_BORDER

    def __init__(self, templates_dir: str | Path):
        self.templates_dir = Path(templates_dir)
        self.templates: Dict[str, List[np.ndarray]] = defaultdict(list)
        if self.templates_dir.exists():
            self._load_templates()

    def _load_templates(self) -> None:
        for path in sorted(self.templates_dir.glob("*.png")):
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            label = path.stem.split("_r")[0]
            self.templates[label].append(img)

    def has_templates(self) -> bool:
        return bool(self.templates)

    def crop_cell_interior(self, image_bgr: np.ndarray, row: int, col: int) -> np.ndarray:
        y0 = row * self.CELL_SIZE + self.WALL_BORDER
        x0 = col * self.CELL_SIZE + self.WALL_BORDER
        return image_bgr[y0:y0 + self.INNER, x0:x0 + self.INNER]

    def classify_cell(self, patch_bgr: np.ndarray) -> Tuple[Optional[str], float]:
        if not self.templates:
            return None, 0.0

        gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)
        best_label = None
        best_score = -1.0

        for label, templates in self.templates.items():
            label_best = -1.0
            for tmpl in templates:
                if gray.shape != tmpl.shape:
                    resized = cv2.resize(gray, (tmpl.shape[1], tmpl.shape[0]), interpolation=cv2.INTER_AREA)
                else:
                    resized = gray
                score = float(cv2.matchTemplate(resized, tmpl, cv2.TM_CCOEFF_NORMED)[0, 0])
                label_best = max(label_best, score)
            if label_best > best_score:
                best_score = label_best
                best_label = label

        return best_label, best_score

    def detect_hazards(
        self,
        image_bgr: np.ndarray,
        rows: int,
        cols: int,
        threshold: float = 0.60,
    ) -> Dict[str, List[Cell]]:
        found: Dict[str, List[Cell]] = defaultdict(list)
        for r in range(rows):
            for c in range(cols):
                patch = self.crop_cell_interior(image_bgr, r, c)
                label, score = self.classify_cell(patch)
                if label is not None and score >= threshold:
                    found[label].append((r, c))
        return dict(found)


class MazeLoader:
    CELL_SIZE = 16
    WALL_BORDER = 1

    def __init__(self, maze_image_path: str | Path, templates_dir: str | Path = "templates"):
        self.image_path = Path(maze_image_path)
        self.img_bgr = cv2.imread(str(self.image_path), cv2.IMREAD_COLOR)
        if self.img_bgr is None:
            raise FileNotFoundError(f"Could not read maze image: {self.image_path}")
        self.img_rgb = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2RGB)
        self.h, self.w = self.img_rgb.shape[:2]

        if self.h < 1024 or self.w < 1024:
            raise ValueError("Expected a 64x64 maze rendered at roughly 16 px per cell.")

        self.maze_height_cells = 64
        self.maze_width_cells = 64

        # True = passable / traversable.
        self.maze_array = self._build_passability_image()

        self.matcher = TemplateMatcher(templates_dir)
        self.hazard_cells_by_label: Dict[str, List[Cell]] = {}
        self.death_pits: List[Cell] = []
        self.confusion_pads: List[Cell] = []
        self.teleport_cells_by_label: Dict[str, List[Cell]] = {}
        self.start_cell: Optional[Cell] = None
        self.goal_cell: Optional[Cell] = None

    def _build_passability_image(self) -> np.ndarray:
        """
        Convert the raw image into a boolean walkability image.
        White-ish background is passable, black-ish wall lines are blocked.
        """
        gray = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2GRAY)
        passable = (gray > 50).astype(np.uint8)
        return passable

    def pixel_to_cell(self, x: int, y: int) -> Cell:
        return (y // self.CELL_SIZE, x // self.CELL_SIZE)

    def cell_center_px(self, row: int, col: int) -> Tuple[int, int]:
        x = min(col * self.CELL_SIZE + self.CELL_SIZE // 2, self.w - 1)
        y = min(row * self.CELL_SIZE + self.CELL_SIZE // 2, self.h - 1)
        return x, y

    def cell_patch(self, row: int, col: int) -> np.ndarray:
        return self.matcher.crop_cell_interior(self.img_bgr, row, col)

    def cell_is_traversable(self, row: int, col: int) -> bool:
        x, y = self.cell_center_px(row, col)
        return bool(self.maze_array[y, x])

    def _candidate_special_cells(self) -> List[Tuple[Cell, np.ndarray]]:
        """
        Return colored cells likely containing start/goal/special icons.
        Hazards are removed later; these are broad candidates.
        """
        out = []
        for r in range(self.maze_height_cells):
            for c in range(self.maze_width_cells):
                patch = self.cell_patch(r, c)
                mask = np.any(patch < 245, axis=2)
                count = int(mask.sum())
                if count >= 100:
                    out.append(((r, c), patch))
        return out

    def _mean_color(self, patch: np.ndarray) -> Tuple[float, float, float]:
        mask = np.any(patch < 245, axis=2)
        if not np.any(mask):
            return (255.0, 255.0, 255.0)
        vals = patch[mask][:, ::-1]  # BGR -> RGB
        mean = vals.mean(axis=0)
        return float(mean[0]), float(mean[1]), float(mean[2])

    def detect_hazards(self, threshold: float = 0.60) -> None:
        self.hazard_cells_by_label = self.matcher.detect_hazards(
            self.img_bgr,
            self.maze_height_cells,
            self.maze_width_cells,
            threshold=threshold,
        )
        self.death_pits = sorted(self.hazard_cells_by_label.get("death_pit", []))
        self.confusion_pads = sorted(self.hazard_cells_by_label.get("confusion", []))
        self.teleport_cells_by_label = {
            label: sorted(cells)
            for label, cells in self.hazard_cells_by_label.items()
            if label.startswith("teleport")
        }

        taken = set(self.death_pits) | set(self.confusion_pads)
        for cells in self.teleport_cells_by_label.values():
            taken.update(cells)

        candidates = [
            (cell, self._mean_color(patch))
            for cell, patch in self._candidate_special_cells()
            if cell not in taken
        ]

        # Goal: green-dominant special cell.
        goal_candidates = [
            (cell, rgb) for cell, rgb in candidates
            if rgb[1] > rgb[0] + 20 and rgb[1] > rgb[2] + 20
        ]
        if goal_candidates:
            self.goal_cell = max(goal_candidates, key=lambda item: item[1][1])[0]

        # Start: orange/yellow special cell that is not the goal.
        start_candidates = [
            (cell, rgb) for cell, rgb in candidates
            if rgb[0] > 160 and rgb[1] > 100 and rgb[2] < 120 and cell != self.goal_cell
        ]
        if start_candidates:
            # In these mazes the start marker is usually the warm round icon near the border.
            self.start_cell = max(start_candidates, key=lambda item: item[1][0] + item[1][1])[0]

        if self.start_cell is None or self.goal_cell is None:
            raise RuntimeError(
                f"Failed to infer start/goal from maze image. "
                f"start={self.start_cell}, goal={self.goal_cell}"
            )


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

    def __init__(
        self,
        maze_image_path: str | Path,
        templates_dir: str | Path = "templates",
        rotate_fire: bool = True,
    ):
        self.loader = MazeLoader(maze_image_path, templates_dir=templates_dir)
        self.loader.detect_hazards()

        self.grid = [
            [self.loader.cell_is_traversable(r, c) for c in range(self.loader.maze_width_cells)]
            for r in range(self.loader.maze_height_cells)
        ]

        self.start_cell = self.loader.start_cell
        self.goal_cell = self.loader.goal_cell

        self.death_pits = set(map(tuple, self.loader.death_pits))
        self.initial_death_pits = set(self.death_pits)
        self.fire_clusters = self.group_clusters(self.death_pits)
        self.initial_fire_clusters = [list(cluster) for cluster in self.fire_clusters]
        self.confusion_pads = set(map(tuple, self.loader.confusion_pads))

        self.teleport_cells_by_label = {
            label: [tuple(cell) for cell in cells]
            for label, cells in self.loader.teleport_cells_by_label.items()
        }
        self.teleport_map = self._build_teleport_map()

        # Hazards are traversable cells. Keep them walkable for movement tests.
        for r, c in self.death_pits | self.confusion_pads | set(self.teleport_map.keys()):
            self.grid[r][c] = True

        self.agent_pos: Cell = self.start_cell
        self.turn_count = 0
        self.death_count = 0
        self.confused_count = 0
        self.teleport_count = 0
        self.confused_turns_left = 0
        self.cells_explored: set[Cell] = set()
        self.episode_active = True
        self.freeze_fire = False
        self.rotate_fire_enabled = rotate_fire
        self.path_length = 0

    def _build_teleport_map(self) -> Dict[Cell, Cell]:
        teleport_map: Dict[Cell, Cell] = {}
        for _label, cells in self.teleport_cells_by_label.items():
            cells = sorted(cells)
            for i in range(0, len(cells) - 1, 2):
                a, b = cells[i], cells[i + 1]
                teleport_map[a] = b
                teleport_map[b] = a
            if len(cells) % 2 == 1:
                teleport_map[cells[-1]] = cells[-1]
        return teleport_map

    def group_clusters(self, cells: Iterable[Cell], max_gap: int = 1) -> List[List[Cell]]:
        remaining = set(cells)
        clusters: List[List[Cell]] = []
        while remaining:
            seed = next(iter(remaining))
            cluster = []
            stack = [seed]
            remaining.remove(seed)
            while stack:
                cur = stack.pop()
                cluster.append(cur)
                for other in list(remaining):
                    if abs(other[0] - cur[0]) <= max_gap and abs(other[1] - cur[1]) <= max_gap:
                        remaining.remove(other)
                        stack.append(other)
            clusters.append(sorted(cluster))
        return clusters

    def cluster_orientation_and_pivot(self, cluster: List[Cell]) -> Tuple[str, Cell]:
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
                candidates = [cell for cell in cluster if cell[0] == max_r]
                return "v", min(candidates, key=lambda cell: abs(cell[1] - sum(cols) / len(cols)))
            candidates = [cell for cell in cluster if cell[0] == min_r]
            return "^", min(candidates, key=lambda cell: abs(cell[1] - sum(cols) / len(cols)))

        if col_counts[min_c] <= col_counts[max_c]:
            candidates = [cell for cell in cluster if cell[1] == min_c]
            return "<", min(candidates, key=lambda cell: abs(cell[0] - sum(rows) / len(rows)))

        candidates = [cell for cell in cluster if cell[1] == max_c]
        return ">", min(candidates, key=lambda cell: abs(cell[0] - sum(rows) / len(rows)))

    def rotate_fire_cluster(self, cluster: List[Cell]) -> List[Cell]:
        h = self.loader.maze_height_cells
        w = self.loader.maze_width_cells
        _orientation, pivot = self.cluster_orientation_and_pivot(cluster)
        pr, pc = pivot

        rotated = []
        seen = set()
        for r, c in cluster:
            dr, dc = r - pr, c - pc
            nr, nc = pr + dc, pc - dr
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in seen:
                seen.add((nr, nc))
                rotated.append((nr, nc))
        return sorted(rotated)

    def rotate_fire_clusters(self) -> None:
        if self.freeze_fire:
            return

        # Restore old pit cells to their underlying traversability.
        for r, c in self.death_pits:
            self.grid[r][c] = self.loader.cell_is_traversable(r, c)

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

    def reset(self) -> Cell:
        self.agent_pos = self.start_cell
        self.turn_count = 0
        self.death_count = 0
        self.confused_count = 0
        self.teleport_count = 0
        self.confused_turns_left = 0
        self.cells_explored = {self.start_cell}
        self.episode_active = True
        self.path_length = 0

        self.death_pits = set(self.initial_death_pits)
        self.fire_clusters = [list(cluster) for cluster in self.initial_fire_clusters]
        for r in range(self.loader.maze_height_cells):
            for c in range(self.loader.maze_width_cells):
                self.grid[r][c] = self.loader.cell_is_traversable(r, c)
        for r, c in self.death_pits | self.confusion_pads | set(self.teleport_map.keys()):
            self.grid[r][c] = True

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

        ma = self.loader.maze_array
        cell = self.CELL_SIZE
        border = 1
        dr = to_r - from_r

        if dr == 0:
            right_c = max(from_c, to_c)
            right_is_hazard = to_hazard if to_c == right_c else from_hazard
            wall_x = right_c * cell + border
            y = min(from_r * cell + cell // 2 + border, ma.shape[0] - 1)

            if right_is_hazard:
                left_is_hazard = from_hazard if to_c == right_c else to_hazard
                if left_is_hazard:
                    return True
                x = min(wall_x - 2, ma.shape[1] - 1)
                return bool(ma[y, x])

            x_left = min(wall_x - 1, ma.shape[1] - 1)
            x_right = min(wall_x + 1, ma.shape[1] - 1)
            return bool(ma[y, x_left]) and bool(ma[y, x_right])

        bottom_r = max(from_r, to_r)
        bottom_is_hazard = to_hazard if to_r == bottom_r else from_hazard
        wall_y = bottom_r * cell + border
        x = min(from_c * cell + cell // 2 + border, ma.shape[1] - 1)

        if bottom_is_hazard:
            top_is_hazard = from_hazard if to_r == bottom_r else to_hazard
            if top_is_hazard:
                return True
            y = min(wall_y - 2, ma.shape[0] - 1)
            return bool(ma[y, x])

        y_top = min(wall_y - 1, ma.shape[0] - 1)
        y_bottom = min(wall_y + 1, ma.shape[0] - 1)
        return bool(ma[y_top, x]) and bool(ma[y_bottom, x])

    def step(self, actions: List[Action]) -> TurnResult:
        if not actions or len(actions) > 5:
            raise ValueError("Each turn must contain between 1 and 5 actions.")

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

            src_hazard = (r, c) in self.death_pits or (r, c) in self.confusion_pads or (r, c) in self.teleport_map
            dst_hazard = (nr, nc) in self.death_pits or (nr, nc) in self.confusion_pads or (nr, nc) in self.teleport_map

            if not self.is_move_passable(r, c, nr, nc, src_hazard, dst_hazard):
                result.wall_hits += 1
                result.actions_executed += 1
                continue

            self.agent_pos = (nr, nc)
            self.cells_explored.add(self.agent_pos)
            result.positions_visited.append(self.agent_pos)
            result.actions_executed += 1
            self.path_length += 1

            if self.agent_pos in self.teleport_map:
                src = self.agent_pos
                self.agent_pos = self.teleport_map[src]
                result.teleported = True
                result.last_event = f"TELEPORT {src}->{self.agent_pos}"
                result.current_position = self.agent_pos
                self.teleport_count += 1

            if self.agent_pos in self.confusion_pads:
                result.is_confused = True
                self.confused_turns_left = 2
                currently_confused = True
                self.confused_count += 1
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
            "path_length": self.path_length,
        }


class Evaluator:
    def evaluate_agent(
        self,
        agent,
        maze_image_path: str | Path,
        num_episodes: int = 5,
        templates_dir: str | Path = "templates",
        rotate_fire: bool = True,
        max_turns: int = 10_000,
    ) -> dict:
        env = MazeEnvironment(
            maze_image_path=maze_image_path,
            templates_dir=templates_dir,
            rotate_fire=rotate_fire,
        )

        episode_results = []
        for _ in range(num_episodes):
            agent.attach_environment(env)
            agent.reset_episode(env.reset())

            last_result = None
            for _turn in range(max_turns):
                actions = agent.plan_turn(last_result)
                last_result = env.step(actions)
                if last_result.is_goal_reached:
                    break

            episode_results.append(env.get_episode_stats())

        successes = [ep for ep in episode_results if ep["goal_reached"]]
        total_turns = sum(ep["turns_taken"] for ep in episode_results)
        total_deaths = sum(ep["deaths"] for ep in episode_results)
        total_paths = sum(ep["path_length"] for ep in episode_results)

        return {
            "success_rate": len(successes) / len(episode_results) if episode_results else 0.0,
            "avg_turns": (sum(ep["turns_taken"] for ep in successes) / len(successes)) if successes else None,
            "avg_deaths": total_deaths / len(episode_results) if episode_results else 0.0,
            "avg_path_length": (sum(ep["path_length"] for ep in successes) / len(successes)) if successes else None,
            "exploration_efficiency": (
                sum(ep["cells_explored"] for ep in episode_results) / total_paths if total_paths else 0.0
            ),
            "death_rate": total_deaths / total_turns if total_turns else 0.0,
            "episodes": episode_results,
        }
