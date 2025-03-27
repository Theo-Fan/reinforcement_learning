import sys

sys.path.append("../..")
from env.grid_world import GridWorld
import numpy as np

gamma = 0.9
num_episode = 1000
alpha = 0.1
epsilon = 0.1


def init_random_qtable(num_states, num_actions):
    matrix = np.full((num_states, num_actions), 0.2)
    return matrix


def epsilon_greedy(state, policy, epsilon, action_space):
    action_idx = np.argmax(policy[state])
    if np.random.uniform(0, 1) < epsilon:
        tmp_action = [i for i in range(len(action_space)) if i != action_idx]
        action_idx = np.random.choice(tmp_action)
    action = action_space[action_idx]
    return action


def train(env, q_table):
    for episode in range(num_episode):
        env.reset()
        # done = False
        # while not done:
        for t in range(3000):
            pos = env.agent_state[1] * env.env_size[0] + env.agent_state[0]
            action = epsilon_greedy(pos, q_table, epsilon, env.action_space)

            next_state, reward, done, info = env.step(action)

            ne_pos = next_state[1] * env.env_size[0] + next_state[0]
            action_idx = env.action_space.index(action)
            q_ne_max = np.max(q_table[ne_pos])

            # policy evaluation & improvement
            q_table[pos][action_idx] -= alpha * (
                    q_table[pos][action_idx] - (reward + gamma * q_ne_max)
            )

            if done:
                break
        # In this environment, we do not need to update state and action.
        # Because the environment achieve it.

        if episode % 100 == 0:
            print(f"Training process: {episode}/{num_episode}")


def std_matrix(X):
    min_vals = np.min(X, axis=1, keepdims=True)
    max_vals = np.max(X, axis=1, keepdims=True)

    # 按行 Min-Max 归一化
    X_norm = (X - min_vals) / (max_vals - min_vals + 1e-8)
    return X_norm


def test(env, q_table):
    env.reset()
    for t in range(1000):
        env.render(animation_interval=0.5)
        state = env.agent_state
        pos = env.agent_state[1] * env.env_size[0] + env.agent_state[0]
        action = env.action_space[np.argmax(q_table[pos])]

        next_state, reward, done, info = env.step(action)
        print(
            f"Step: {t}, Action: {action}, Cur-state: {state}, Re-pos: {pos}, "
            f"Idx: {np.argmax(q_table[pos])}, Next-state: {next_state}, Reward: {reward}, Done: {done}")
        if done:
            break


def main():
    env = GridWorld()
    num_state = env.num_states
    num_action = len(env.action_space)
    q_table = init_random_qtable(
        num_states=num_state,
        num_actions=num_action
    )

    train(env, q_table)

    test(env, q_table)

    env.add_policy(std_matrix(q_table))
    print("Final Q-table: \n", q_table)

    env.render(animation_interval=10)


if __name__ == "__main__":
    main()
