
from __future__ import annotations

import argparse
from statistics import mean

from agent import FreshMazeAgent
from core import MazeEnvironment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--maze", default="/mnt/data/MAZE_1.png")
    parser.add_argument("--templates", default="templates")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-turns", type=int, default=10_000)
    parser.add_argument("--rotate-fire", action="store_true")
    parser.add_argument("--model", default="fresh_agent.pkl")
    args = parser.parse_args()

    env = MazeEnvironment(args.maze, templates_dir=args.templates, rotate_fire=args.rotate_fire)
    agent = FreshMazeAgent()
    agent.attach_environment(env)
    agent.load(args.model)

    runs = []
    for ep in range(1, args.episodes + 1):
        start = env.reset()
        agent.reset_episode(start)
        last_result = None

        for _turn in range(args.max_turns):
            actions = agent.plan_turn(last_result)
            last_result = env.step(actions)
            if last_result.is_goal_reached:
                break

        stats = env.get_episode_stats()
        runs.append(stats)
        status = "SUCCESS" if stats["goal_reached"] else "FAILED"
        print(
            f"Episode {ep}: {status} | turns={stats['turns_taken']} | "
            f"deaths={stats['deaths']} | explored={stats['cells_explored']} | "
            f"path={stats['path_length']}"
        )

    successes = [r for r in runs if r["goal_reached"]]
    success_rate = len(successes) / len(runs) if runs else 0.0
    avg_turns = mean(r["turns_taken"] for r in successes) if successes else None
    avg_path = mean(r["path_length"] for r in successes) if successes else None
    total_turns = sum(r["turns_taken"] for r in runs)
    total_deaths = sum(r["deaths"] for r in runs)
    total_path = sum(r["path_length"] for r in runs)
    total_unique = sum(r["cells_explored"] for r in runs)
    death_rate = total_deaths / total_turns if total_turns else 0.0
    exploration_efficiency = total_unique / total_path if total_path else 0.0

    print("\nFINAL METRICS")
    print("=" * 50)
    print(f"Success Rate            : {success_rate:.1%}")
    print(f"Avg Turns (successful)  : {avg_turns if avg_turns is not None else 'N/A'}")
    print(f"Avg Path (successful)   : {avg_path if avg_path is not None else 'N/A'}")
    print(f"Death Rate              : {death_rate:.4f}")
    print(f"Exploration Efficiency  : {exploration_efficiency:.3f}")


if __name__ == "__main__":
    main()
