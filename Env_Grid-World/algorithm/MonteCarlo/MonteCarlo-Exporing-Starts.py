import sys
import numpy as np
from collections import defaultdict

sys.path.append("../..")
from src.grid_world import GridWorld

gamma = 0.9


def init_random_policy(num_states, num_actions):
    policy = np.zeros((num_states, num_actions))
    for s in range(num_states):
        policy[s, np.random.choice(num_actions)] = 1
    return policy


def mc_exploring_starts(env, gamma, num_episodes=500):
    num_states = env.num_states
    num_actions = len(env.action_space)

    policy = init_random_policy(num_states, num_actions)
    print("Init Optimal Policy:")
    print(policy)

    returns_count = np.zeros((num_states, num_actions))  # Count of returns for each (state, action)
    returns_sum = np.zeros((num_states, num_actions))  # Sum of returns for each (state, action)

    for episode in range(num_episodes):
        # Step 1: Exploring Starts: Randomly choose a starting state and action
        start_state = np.random.choice(num_states)
        start_action = np.random.choice(num_actions)
        env.set_state((start_state % env.env_size[1], start_state // env.env_size[1]))
        action = env.action_space[start_action]

        # Generate an episode starting from (s0, a0)
        episode_data = []  # Store (state, action, reward) tuples
        state = start_state

        for i in range(200):
            next_state, reward, done, _ = env.step(action)
            episode_data.append((state, action, reward))

            state = next_state[1] * env.env_size[0] + next_state[0]

            action_prob = policy[state]
            action = env.action_space[np.random.choice(num_actions, p=action_prob)]

        # Step 2: Policy Evaluation and Improvement
        g = 0
        visited = set()

        for t in reversed(range(len(episode_data))):
            state, action, reward = episode_data[t]
            g = gamma * g + reward

            # First-visit method: Update Q only for first visits of (state, action)
            if (state, action) not in visited:
                visited.add((state, action))

                action_idx = env.action_space.index(action)
                returns_sum[state, action_idx] += g
                returns_count[state, action_idx] += 1

                policy[state, action_idx] = returns_sum[state, action_idx] / returns_count[state, action_idx]

                # Policy improvement: Make the policy greedy w.r.t. Q
                best_action_idx = np.argmax(policy[state])
                policy[state] = np.zeros(num_actions)
                policy[state, best_action_idx] = 1  # Update to greedy policy

    return policy


if __name__ == "__main__":
    env = GridWorld()

    optimal_policy = mc_exploring_starts(env, gamma, num_episodes=500)

    print("Optimal Policy:")
    print(optimal_policy)

    state = env.reset()
    for t in range(100):
        env.render(animation_interval=0.5)
        pos = env.agent_state[1] * env.env_size[0] + env.agent_state[0]
        action = env.action_space[np.argmax(optimal_policy[pos])]

        next_state, reward, done, info = env.step(action)
        print(
            f"Step: {t}, Action: {action}, Cur-state: {state}, Re-pos:{pos}, "
            f"Idx: {np.argmax(optimal_policy[pos])}, Next-state: {next_state}, Reward: {reward}, Done: {done}"
        )
        if done:
            break

    env.add_policy(optimal_policy)
    env.render(animation_interval=10)
