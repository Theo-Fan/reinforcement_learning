import random
import sys
import numpy as np

sys.path.append("../..")
from env.grid_world import GridWorld

gamma = 0.9
num_episode = 2000
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


def train_sarsa(env, q_table, num_episode, alpha, gamma, epsilon):
    for episode in range(num_episode):
        env.reset()
        pos = env.agent_state[1] * env.env_size[0] + env.agent_state[0]
        action = epsilon_greedy(pos, q_table, epsilon, env.action_space)
        for t in range(100):
            next_state, reward, done, info = env.step(action)

            # get next state & action
            ne_pos = next_state[1] * env.env_size[0] + next_state[0]
            next_action = epsilon_greedy(ne_pos, q_table, epsilon, env.action_space)

            # update q-value [policy improvement]
            q_table[pos][env.action_space.index(action)] -= alpha * (
                    q_table[pos][env.action_space.index(action)] -
                    (reward + gamma * q_table[ne_pos][env.action_space.index(next_action)])
            )

            # update state & action
            action = next_action
            pos = ne_pos

            if done:
                break

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{num_episode} completed.")



def test_policy(env, policy_matrix):
    env.reset()
    for t in range(15):
        env.render(animation_interval=0.5)
        state = env.agent_state
        pos = env.agent_state[1] * env.env_size[0] + env.agent_state[0]
        action = env.action_space[np.argmax(policy_matrix[pos])]

        next_state, reward, done, info = env.step(action)
        print(
            f"Step: {t}, Action: {action}, Cur-state: {state}, Re-pos: {pos}, "
            f"Idx: {np.argmax(policy_matrix[pos])}, Next-state: {next_state}, Reward: {reward}, Done: {done}")
        if done:
            break


def std_matrix(X):
    min_vals = np.min(X, axis=1, keepdims=True)
    max_vals = np.max(X, axis=1, keepdims=True)

    # 按行 Min-Max 归一化
    X_norm = (X - min_vals) / (max_vals - min_vals + 1e-8)  # 避免除以0
    return X_norm

if __name__ == "__main__":
    env = GridWorld()

    q_table = init_random_qtable(env.num_states, len(env.action_space))

    # train
    train_sarsa(env, q_table, num_episode, alpha, gamma, epsilon)

    # test
    test_policy(env, q_table)

    print("q-Table:", q_table)

    env.add_policy(std_matrix(q_table))
    env.render(animation_interval=10)
