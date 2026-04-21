
from __future__ import annotations

import argparse

from agent import FreshMazeAgent
from core import MazeEnvironment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--maze", default="/mnt/data/MAZE_1.png")
    parser.add_argument("--templates", default="templates")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--max-turns", type=int, default=10_000)
    parser.add_argument("--rotate-fire", action="store_true")
    parser.add_argument("--save", default="fresh_agent.pkl")
    args = parser.parse_args()

    env = MazeEnvironment(args.maze, templates_dir=args.templates, rotate_fire=args.rotate_fire)
    agent = FreshMazeAgent()
    agent.attach_environment(env)

    successes = 0
    for ep in range(1, args.episodes + 1):
        start = env.reset()
        agent.reset_episode(start)
        last_result = None

        for turn in range(1, args.max_turns + 1):
            actions = agent.plan_turn(last_result)
            last_result = env.step(actions)
            if last_result.is_goal_reached:
                successes += 1
                print(
                    f"Ep {ep:>3}: SUCCESS in {turn:>5} turns | "
                    f"known={len(agent.known_cells)} | wins={successes}"
                )
                break
        else:
            print(f"Ep {ep:>3}: timeout          | known={len(agent.known_cells)}")

    agent.save(args.save)
    print(f"Saved trained agent to {args.save}")


if __name__ == "__main__":
    main()
