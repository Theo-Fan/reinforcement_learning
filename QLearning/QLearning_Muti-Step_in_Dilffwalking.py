import time
from re import findall
import gymnasium as gym
import numpy as np
import pandas as pd


def build_qtable(n_state, actions):
    table = pd.DataFrame(
        np.zeros((n_state, actions)),
        columns=[0, 1, 2, 3]
    )
    return table


def take_action(state, q_table, epsilon, actions):
    state_action = q_table.iloc[state, :]
    if np.random.uniform() < epsilon:
        action = np.random.choice(actions)
    else:
        action = state_action.idxmax()
    return action


def test(q_table):
    test_env = gym.make("CliffWalking-v0", render_mode='human')
    action_dim = int(findall(r"\d+\.?\d*", str(test_env.action_space))[0])

    for i in range(10):
        state, _ = test_env.reset()
        done = False
        step = -1
        while not done:
            step += 1
            action = take_action(state, q_table, epsilon, action_dim)
            next_state, reward, terminated, truncated, info = test_env.step(action)

            done = truncated or terminated
            state = next_state

            print(f'Test ---- step_num: {step}, action: {action}, reward: {reward}, obs: {state}, done: {done}')
            time.sleep(0.1)


def train(alpha, gamma, epsilon, n_step):
    env = gym.make("CliffWalking-v0")
    state_space = int(findall(r"\d+\.?\d*", str(env.observation_space))[0])
    action_space = int(findall(r"\d+\.?\d*", str(env.action_space))[0])

    q_table = build_qtable(state_space, action_space)

    for epoch in range(max_epochs):
        state, _ = env.reset()
        done = False
        step_count = 0
        state_action_reward = []

        while not done:
            action = take_action(state, q_table, epsilon, action_space)
            next_state, reward, terminated, truncated, info = env.step(action)

            state_action_reward.append((state, action, reward))
            if len(state_action_reward) > n_step:
                state_action_reward.pop(0) # delete the oldest one

            if len(state_action_reward) == n_step or terminated or truncated:
                G = sum([gamma ** i * r for i, (_, _, r) in enumerate(state_action_reward)])
                if not (terminated or truncated):
                    G += gamma ** n_step * q_table.iloc[next_state, :].max()

                state_update, action_update, _ = state_action_reward[0]
                q_table.iloc[state_update, action_update] += alpha * (G - q_table.iloc[state_update, action_update])

            state = next_state
            done = terminated or truncated
            step_count += 1

        print(f"Epoch {epoch + 1}: Finished in {step_count} steps.")

    print("Final Q-table:\n", q_table)
    return q_table


if __name__ == '__main__':
    max_epochs = 1000
    alpha = 0.1
    gamma = 0.9
    epsilon = 0.1
    n_step = 3  # 多步TD目标的步数

    q_table = train(alpha, gamma, epsilon, n_step)
    test(q_table)