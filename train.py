import argparse

from core import MazeEnvironment
from agent import MazeAgent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--maze", default="maze-alpha/MAZE_1.png")
    parser.add_argument("--templates", default="templates")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--max-turns", type=int, default=10_000)
    parser.add_argument("--rotate-fire", action="store_true")
    parser.add_argument("--save", default="agent.pkl")
    args = parser.parse_args()

    env = MazeEnvironment(
        args.maze,
        templates_dir=args.templates,
        rotate_fire=args.rotate_fire,
    )

    agent = MazeAgent()
    agent.env = env

    successes = 0

    for ep in range(args.episodes):
        pos = env.reset()
        agent.current_pos = pos
        agent.start_pos = pos
        agent.goal_pos = env.goal_cell
        agent.reset_episode()
        last_result = None

        for turn in range(args.max_turns):
            old_pos = agent.current_pos
            old_state = agent.state()

            actions = agent.plan_turn(last_result)
            result = env.step(actions)

            reward = agent.compute_reward(result, old_pos)
            new_state = agent.state()
            agent.update_q(old_state, actions[0], reward, new_state)

            last_result = result

            if result.is_goal_reached:
                successes += 1
                print(
                    f"Ep {ep+1:>3}: SUCCESS in {turn+1:>5} turns | "
                    f"ε={agent.epsilon:.3f} | "
                    f"known={len(agent.known)} cells | "
                    f"cumulative wins={successes}"
                )
                break

            if turn == args.max_turns - 1:
                print(
                    f"Ep {ep+1:>3}: timeout          | "
                    f"ε={agent.epsilon:.3f} | "
                    f"known={len(agent.known)} cells"
                )

    agent.save(args.save)
    print(f"\nTraining complete. Total successes: {successes}/{args.episodes}")


if __name__ == "__main__":
    main()