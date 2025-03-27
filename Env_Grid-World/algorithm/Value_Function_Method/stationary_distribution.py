import sys
import random
import numpy as np
import matplotlib.pyplot as plt
"""
    Before running this code, you need modify the arguments.py file to keep the same environment setting.
"""
sys.path.append("../..")
from env.grid_world import GridWorld


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


def only_show_max(matrix):
    x, y = matrix.shape
    row_max = np.argmax(matrix, axis=1)
    new_matrix = np.full((x, y), 0.05)
    for i in range(x):
        new_matrix[i][row_max[i]] = 0.8
    print(new_matrix)
    return new_matrix




def visualize(env, n_ratio, num_steps, q_table):
    data = np.array(n_ratio)
    plt.figure()
    for i, row in enumerate(data):
        plt.plot(row, label=f"State {i + 1}")

    plt.xlabel("Episodes")
    plt.ylabel("State Visit Frequency")

    plt.xlim(0, num_steps)
    plt.ylim(0, 1)
    plt.legend()
    plt.show()

    env.reset()
    env.render()
    env.add_policy(only_show_max(q_table))

    env.render(animation_interval=10)


def train(num_episode, num_steps, alpha, gamma, epsilon, env, q_table, num_states):
    n_visits = None
    n_ratio = None

    for i in range(num_episode):
        n_visits = np.zeros(num_states)
        n_ratio = [[] for s in range(num_states)]

        s = np.random.randint(env.num_states)
        env.set_state((s % env.env_size[1], s // env.env_size[1]))
        pos = env.agent_state[1] * env.env_size[0] + env.agent_state[0]
        action = epsilon_greedy(pos, q_table, epsilon, env.action_space)

        for t in range(num_steps):
            n_visits[pos] += 1

            for it in range(num_states):
                if n_visits[it] == 0:
                    n_ratio[it].append(0)
                else:
                    n_ratio[it].append(n_visits[it] / (t + 1))

            next_state, reward, done, info = env.step(action)
            ne_pos = next_state[1] * env.env_size[0] + next_state[0]
            next_action = epsilon_greedy(ne_pos, q_table, epsilon, env.action_space)

            q_table[pos][env.action_space.index(action)] -= alpha * (
                    q_table[pos][env.action_space.index(action)] -
                    (reward + gamma * q_table[ne_pos][env.action_space.index(next_action)])
            )

            action = next_action
            pos = ne_pos

    return n_visits, n_ratio


def main():
    gamma = 0.9
    num_episode = 50
    num_steps = 1000
    alpha = 0.1
    epsilon = 0.5

    env = GridWorld()
    num_states = env.num_states
    num_actions = len(env.action_space)

    q_table = init_random_qtable(num_states, num_actions)

    n_visits, n_ratio = train(num_episode, num_steps, alpha, gamma, epsilon, env, q_table, num_states)

    print(q_table)
    visualize(env, n_ratio, num_steps, q_table)


if __name__ == "__main__":
    main()
