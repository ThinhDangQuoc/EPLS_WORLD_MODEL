"""RHEA/RMHC test suite on simple Gymnasium environments."""
import math
import time
import random
import gymnasium as gym
from planning.real.mcts import MCTS  # type: ignore[import]
from planning.real.rolling_horizon import RHEA  # type: ignore[import]
from planning.real.random_mutation_hill_climbing import RMHC  # type: ignore[import]

SEED = 0
is_seed = True
is_replay = False
MAX_TIME_STEPS = 200
games = ["FrozenLake-v1", "CartPole-v1", "Pendulum-v1"]

random_mutation_hill_climb_test_parameters = {
    "FrozenLake-v1": (25, 10, True,  False, 0.1) if is_seed else (25, 100, True, False, 0.2),
    "CartPole-v1":   (1,   5, False, True,  0.1) if is_seed else (2,   50, False, True),
    "Pendulum-v1":   (20, 30, True,  False, 0.1) if is_seed else (20,  30, True, False),
}

rolling_horizon_test_parameters = {
    "FrozenLake-v1": (4, 15, 6, True, False) if is_seed else (8, 20, 40, True, False, 0.2),
    "CartPole-v1":   (4,  3, 10, True, True)  if is_seed else (8,  2, 10, True, True),
    "Pendulum-v1":   (8, 20, 5, True, False)  if is_seed else (8, 20, 10, True, False),
}

monte_carlo_tree_search_parameters = {
    "FrozenLake-v1": (1, 10)             if is_seed else (1, 50),
    "CartPole-v1":   (math.sqrt(2), 3)   if is_seed else (1, 10),
    "Pendulum-v1":   (math.sqrt(2), 30)  if is_seed else (math.sqrt(2), 30),
}

print(f"\nSTARTING PLANNING TESTS on {games}\n")
for game in games:
    agents = [RMHC(*random_mutation_hill_climb_test_parameters[game]),
              RHEA(*rolling_horizon_test_parameters[game]),
              MCTS(*monte_carlo_tree_search_parameters[game])]

    for agent in agents:
        print(f"TESTING {agent} on {game}")
        make_kwargs = {"is_slippery": False} if game == "FrozenLake-v1" else {}
        env = gym.make(game, **make_kwargs)

        if is_seed:
            random.seed(SEED)
        obs, _ = env.reset(seed=SEED if is_seed else None)

        total_reward = 0.0
        actions = []
        start_time = time.time()

        for t in range(MAX_TIME_STEPS):
            action = agent.search(env)
            actions.append(action)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            done = terminated or truncated

            if done:
                if game == "FrozenLake-v1":
                    assert total_reward == 1
                elif game == "CartPole-v1":
                    print(total_reward)
                    assert total_reward == 200
                else:
                    print(total_reward)
                    expected_min_reward = -600 if isinstance(agent, MCTS) else -400
                    assert total_reward >= expected_min_reward
                print(f"Total Reward:{total_reward} | Reward: {reward} | Steps: {t} | Seconds: {round(time.time() - start_time, 2)} \n")
                break

        if is_replay:
            print(f"Replaying {agent} plan in {game} \n")
            if is_seed:
                random.seed(SEED)
            env.reset(seed=SEED if is_seed else None)
            for a in actions:
                env.step(a)
                env.render()

        env.close()
    print(f"SUCCESS: all agents passed {game} \n")

print(f"SUCCESS: all agents passed planning on {games} \n")
