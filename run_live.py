
from __future__ import annotations

import argparse
import time

from agent import MazeAgent
from core import MazeEnvironment
from visualize import LiveVisualizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--maze", default="/mnt/data/MAZE_1.png")
    parser.add_argument("--templates", default="templates")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-turns", type=int, default=10_000)
    parser.add_argument("--rotate-fire", action="store_true")
    parser.add_argument("--model", default="fresh_agent.pkl")
    parser.add_argument("--delay", type=float, default=0.05)
    args = parser.parse_args()

    env = MazeEnvironment(args.maze, templates_dir=args.templates, rotate_fire=args.rotate_fire)
    agent = MazeAgent()
    agent.attach_environment(env)
    agent.load(args.model)

    viz = LiveVisualizer(env, title="Fresh Maze Solver", delay=args.delay)
    for ep in range(1, args.episodes + 1):
        start = env.reset()
        agent.reset_episode(start)
        last_result = None
        path = [start]

        for turn in range(1, args.max_turns + 1):
            actions = agent.plan_turn(last_result)
            last_result = env.step(actions)

            if last_result.is_dead:
                path = [env.start_cell]
            else:
                for pos in last_result.positions_visited:
                    if not path or path[-1] != pos:
                        path.append(pos)

            viz.update(
                known_cells=agent.known_cells,
                current_pos=last_result.current_position,
                path=path,
                episode=ep,
                turn=turn,
                start_pos=env.start_cell,
                goal_pos=env.goal_cell,
                extra_stats={
                    "pos": last_result.current_position,
                    "deaths": env.death_count,
                    "teleports": env.teleport_count,
                    "confused": env.confused_count,
                    "explored": len(env.cells_explored),
                    "event": last_result.last_event or "—",
                },
            )

            if last_result.is_goal_reached:
                print(f"Episode {ep}: SUCCESS in {turn} turns")
                break
        else:
            print(f"Episode {ep}: timeout")

        time.sleep(1.0)
        viz.reset_episode()

    viz.close()


if __name__ == "__main__":
    main()
