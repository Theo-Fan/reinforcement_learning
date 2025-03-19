"""
    A model-free variant of policy iteration
"""
import sys
import numpy as np

sys.path.append("../..")
from src.grid_world import GridWorld

gamma = 0.9
epsilon = 0.1


def init_random_policy(num_states, num_actions):
    # full random policy
    policy = np.full((num_states, num_actions), 0.2)
    return policy


def mc_exploring_starts(env, gamma, num_episodes=5000):
    num_states = env.num_states
    num_actions = len(env.action_space)

    policy = init_random_policy(num_states, num_actions)  # init random policy
    print("Init Optimal Policy:")
    print(policy)

    return_cnt = np.zeros((num_states, num_actions))  # Count of returns for each (state, action)
    # return_sum = np.zeros((num_states, num_actions))  # Sum of returns for each (state, action)

    for episode in range(num_episodes):
        # === Step 1: Exploring Starts: Randomly choose a starting state and action ===
        state = np.random.choice(num_states)
        env.set_state((state % env.env_size[1], state // env.env_size[1]))  # set agent position

        # pos = env.agent_state[1] * env.env_size[0] + env.agent_state[0]
        # action = env.action_space[np.argmax(policy[pos])]  # get action from policy when agent is in start state
        action = env.action_space[np.random.choice(5)]

        episode_data = []  # Store (state, action, reward) tuples, Generate an episode starting from (s0, a0)

        for i in range(200):
            next_state, reward, done, _ = env.step(action)
            episode_data.append((state, action, reward))  # store (state, action, reward)
            state = next_state[1] * env.env_size[0] + next_state[0]  # update state to next state
            # action = env.action_space[np.argmax(policy[state])]
            action_idx = np.argmax(policy[state])
            if np.random.random() < epsilon:
                tmp_action = [i for i in range(num_actions) if i != action_idx]
                action_idx = np.random.choice(tmp_action)
            action = env.action_space[action_idx]
            if done: break

        # ==== Step 2: Policy Evaluation and Improvement ====
        g = 0
        vis = set()

        for t in range(1, len(episode_data) + 1):
            state, action, reward = episode_data[-t]
            g = gamma * g + reward

            # ===== Every-visit method: Update Q for every visit of (state, action) =====
            action_idx = env.action_space.index(action)  # change action to index

            policy[state, action_idx] = g + policy[state, action_idx] * return_cnt[state, action_idx]
            return_cnt[state, action_idx] += 1
            policy[state, action_idx] /= return_cnt[state, action_idx]

            # Policy improvement: epsilon-greedy. We move this part to above (when agent choose action)

    return policy


def std_matrix(X):
    min_vals = np.min(X, axis=1, keepdims=True)
    max_vals = np.max(X, axis=1, keepdims=True)

    # 按行 Min-Max 归一化
    X_norm = (X - min_vals) / (max_vals - min_vals + 1e-8)  # 避免除以0
    return X_norm

def evaluate(env, optimal_policy):
    state = env.reset()
    for t in range(20):
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

    env.add_policy(std_matrix(optimal_policy))
    env.render(animation_interval=10)


def main():
    env = GridWorld()
    optimal_policy = mc_exploring_starts(env, gamma)
    evaluate(env, optimal_policy)

    print("Optimal Policy:")
    print(optimal_policy)


if __name__ == "__main__":
    main()
