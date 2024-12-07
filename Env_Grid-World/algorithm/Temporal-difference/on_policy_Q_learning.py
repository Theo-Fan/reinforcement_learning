import sys

sys.path.append("../..")
from src.grid_world import GridWorld
import os
import random
import numpy as np

gamma = 0.9
num_episode = 1000
alpha = 0.1
epsilon = 0.1


def init_random_deterministic_policy(num_states, num_actions):
    matrix = np.zeros((num_states, num_actions))
    for row in matrix:
        row[np.random.randint(0, num_actions)] = 1
    return matrix


def epsilon_greedy_policy(state, policy_matrix, action_space):
    if random.uniform(0, 1) < epsilon:
        action = random.choice(action_space)  # 随机探索
    else:
        action = action_space[np.argmax(policy_matrix[state])]  # 选择最优动作
    return action


def train(env):
    for episode in range(num_episode):
        env.reset()
        pos = env.agent_state[1] * env.env_size[0] + env.agent_state[0]
        action = epsilon_greedy_policy(pos, q_table, env.action_space)
        done = False
        # while not done:
        for t in range(3000):
            next_state, reward, done, info = env.step(action)
            ne_pos = next_state[1] * env.env_size[0] + next_state[0]
            next_action = epsilon_greedy_policy(ne_pos, q_table, env.action_space)
            action_idx = env.action_space.index(action)

            q_ne_max = q_table[ne_pos][np.argmax(q_table[ne_pos])]

            # print(np.argmax(q_table[ne_pos]))
            # print(f"Episode: {episode} q_ne_max: {q_ne_max}, q_table[pos]: {q_table[ne_pos]}")
            # os._exit(0)

            # policy evaluation & improvement
            q_table[pos][action_idx] -= alpha * (
                    q_table[pos][action_idx] -
                    (reward + gamma * q_ne_max)
            )

            pos = ne_pos
            action = next_action
        if episode % 100 == 0:
            print(f"Current Training episode: {episode}")


def test(env):
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


if __name__ == "__main__":
    env = GridWorld()
    num_state = env.num_states
    num_action = len(env.action_space)
    q_table = init_random_deterministic_policy(
        num_states=num_state,
        num_actions=num_action
    )

    print("Init Q-table: \n",q_table)

    train(env)

    test(env)

    env.add_policy(q_table)

    env.render(animation_interval=10)
