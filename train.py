from hazardDemo import MazeEnvironment
from agent import MazeAgent

MAX_EPISODES = 500
MAX_TURNS    = 10_000
MAZE_PATH    = "maze-alpha/MAZE_1.png"

env   = MazeEnvironment(MAZE_PATH)
agent = MazeAgent()

successes = 0

for ep in range(MAX_EPISODES):
    pos = env.reset()
    agent.current_pos = pos
    agent.start_pos   = pos
    agent.goal_pos    = env.goal_cell
    agent.reset_episode()
    last_result = None

    for turn in range(MAX_TURNS):
        old_pos   = agent.current_pos
        old_state = agent.state()

        actions = agent.plan_turn(last_result)
        result  = env.step(actions)

        reward    = agent.compute_reward(result, old_pos)
        new_state = agent.state()
        agent.update_q(old_state, actions[0], reward, new_state)

        last_result = result

        if result.is_goal_reached:
            agent.goal_pos = env.goal_cell
            successes += 1
            print(f"Ep {ep+1:>3}: SUCCESS in {turn+1:>5} turns | "
                  f"ε={agent.epsilon:.3f} | "
                  f"known={len(agent.known)} cells | "
                  f"cumulative wins={successes}")
            break

        if env.death_count > 200:
            print(f"Ep {ep+1:>3}: ABORT (deaths={env.death_count}) | "
                  f"turn={turn+1} | "
                  f"ε={agent.epsilon:.3f} | "
                  f"pos={agent.current_pos} | "
                  f"goal={'known' if agent.goal_pos else 'NOT found'} | "
                  f"explored={len(env.cells_explored)}")
            break

        if turn == MAX_TURNS - 1:
            goal_known = agent.goal_pos is not None
            if goal_known:
                dist = abs(agent.current_pos[0]-agent.goal_pos[0]) + abs(agent.current_pos[1]-agent.goal_pos[1])
                reason = f"goal known @ {agent.goal_pos}, dist={dist}, deaths={env.death_count}, explored={len(env.cells_explored)}"
            else:
                reason = f"goal NOT found, deaths={env.death_count}, explored={len(env.cells_explored)}"
            print(f"Ep {ep+1:>3}: TIMEOUT          | "
                  f"ε={agent.epsilon:.3f} | "
                  f"known={len(agent.known)} cells | {reason}")

agent.save("agent.pkl")
print(f"\nTraining complete.  Total successes: {successes}/{MAX_EPISODES}")