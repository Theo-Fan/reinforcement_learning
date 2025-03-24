import sys
import random
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("../..")
from src.grid_world import GridWorld  # 确保您已正确安装或导入该模块

def init_random_qtable(num_states, num_actions):
    matrix = np.ones((num_states, num_actions)) / num_actions
    return matrix

def epsilon_greedy_policy(state, policy_matrix, epsilon, action_space):
    if random.uniform(0, 1) < epsilon:
        action = random.choice(action_space)  # 随机探索
    else:
        action = action_space[np.argmax(policy_matrix[state])]  # 选择最优动作
    return action

if __name__ == "__main__":
    gamma = 0.9
    num_episode = 1000
    alpha = 0.1
    epsilon = 0.5

    env = GridWorld()
    num_states = env.num_states
    num_actions = len(env.action_space)

    n_visits = np.zeros(num_states)
    n_ratio = [ [] for s in range(num_states)]

    q_table = np.zeros((num_states, num_actions))
    policy_matrix = init_random_qtable(num_states, num_actions)



    env.reset()
    pos = env.agent_state[1] * env.env_size[0] + env.agent_state[0]
    action = epsilon_greedy_policy(pos, policy_matrix, epsilon, env.action_space)

    for t in range(5000):
        n_visits[pos] += 1
        print(pos)

        for it in range(num_states):
            if n_visits[it] == 0:
                n_ratio[it].append(0)
            else:
                n_ratio[it].append(n_visits[it] / (t + 1))


        next_state, reward, done, info = env.step(action)
        ne_pos = next_state[1] * env.env_size[0] + next_state[0]
        next_action = epsilon_greedy_policy(ne_pos, policy_matrix, epsilon, env.action_space)

        q_table[pos][env.action_space.index(action)] -= alpha * (
            q_table[pos][env.action_space.index(action)] -
            (reward + gamma * q_table[ne_pos][env.action_space.index(next_action)])
        )

        best_action = np.argmax(q_table[ne_pos])
        policy_matrix[ne_pos] = epsilon / (len(env.action_space) - 1)
        policy_matrix[ne_pos][best_action] = 1 - epsilon

        action = next_action
        pos = ne_pos

        # if done:
        #     break
    print(policy_matrix)





    data = np.array(n_ratio)
    plt.figure()
    for i, row in enumerate(data):
        plt.plot(row, label=f"State {i + 1}")
    plt.xlabel("Episodes")
    plt.ylabel("State Visit Frequency")
    plt.legend()
    plt.show()

    # env.add_policy(policy_matrix)

    env.render(animation_interval=10)