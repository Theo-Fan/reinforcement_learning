import sys

sys.path.append("../..")
from env.grid_world import GridWorld
import os
import random
import numpy as np

gamma = 0.9
alpha = 0.1
epsilon = 0.2


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


def generate_episode(env, q_table):
    tot_episode = []
    env.reset()
    for i in range(3000000):
        pos = env.agent_state[1] * env.env_size[0] + env.agent_state[0]
        action = epsilon_greedy_policy(pos, q_table, env.action_space)
        next_state, reward, done, info = env.step(action)
        # print(env.agent_state, next_state)
        ne_pos = next_state[1] * env.env_size[0] + next_state[0]

        tot_episode.append((pos, action, reward, ne_pos))

        pos = ne_pos

    return tot_episode


def train(env, q_table):
    episode = generate_episode(env, q_table)
    print(set(episode))
    print(len(set(episode)))

    for idx, it in enumerate(episode):
        state, action, reward, next_state = it
        action_idx = env.action_space.index(action)
        q_ne_max = q_table[next_state][np.argmax(q_table[next_state])]
        q_table[state][action_idx] -= alpha * (
                q_table[state][action_idx] -
                (reward + gamma * q_ne_max)
        )


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


if __name__ == "__main__":
    env = GridWorld()
    num_state = env.num_states
    num_action = len(env.action_space)
    q_table = init_random_deterministic_policy(
        num_states=num_state,
        num_actions=num_action
    )

    print("Init Q-table: \n", q_table)

    train(env, q_table)
    print(q_table)

    test(env, q_table)

    env.add_policy(q_table)

    env.render(animation_interval=10)