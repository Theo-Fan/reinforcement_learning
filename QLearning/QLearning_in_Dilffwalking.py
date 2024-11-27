import time
from re import findall

import gymnasium as gym
import numpy as np
import pandas as pd


def build_qtable(n_state, actions):
    table = pd.DataFrame(  # 初始化 Q 表，每个状态-动作对的初始值为0
        np.zeros((n_state, actions)),
        columns=[0, 1, 2, 3]
    )
    return table


def take_action(state, q_table, epsilon, actions):
    # epsilon-greedy 策略选择动作
    state_action = q_table.iloc[state, :]
    if np.random.uniform() < epsilon:
        action = np.random.choice(actions)  # 随机选择动作
    else:
        action = state_action.idxmax()  # 选择当前 Q 值最大的动作
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

            ne_state, r, terminated, truncated, info = test_env.step(action)

            done = truncated or terminated
            state = ne_state

            print(f'Test ---- step_num: {step}, action: {action}, reward: {r}, obs: {state}, done: {done}')
            time.sleep(0.1)


def train(alpha, gamma, epsilon):
    env = gym.make("CliffWalking-v0")

    state_space = int(findall(r"\d+\.?\d*", str(env.observation_space))[0])
    print("State space size:", state_space)

    action_space = int(findall(r"\d+\.?\d*", str(env.action_space))[0])  # 获取动作维度
    print("Action space size:", action_space)

    q_table = build_qtable(state_space, action_space)

    # 训练过程
    for epoch in range(max_epochs):
        state, _ = env.reset()  # 初始化状态
        done = False
        step_count = 0  # 记录步数

        while not done:
            action = take_action(state, q_table, epsilon, action_space)

            next_state, reward, terminated, truncated, info = env.step(action)

            q_predict = q_table.iloc[state, action]

            if not (terminated or truncated):
                q_target = reward + gamma * q_table.iloc[next_state, :].max()
            else:
                q_target = reward  # 到达终点或掉入悬崖，Q值仅为当前奖励
                done = True

            q_table.iloc[state, action] += alpha * (q_target - q_predict)

            state = next_state
            step_count += 1

        print(f"Epoch {epoch + 1}: Finished in {step_count} steps.")

    print("Final Q-table:\n", q_table)
    return q_table


if __name__ == '__main__':
    max_epochs = 500  # 训练的 epoch 数量
    alpha = 0.1  # 学习率
    gamma = 0.9  # 折扣因子
    epsilon = 0.1  # epsilon greedy 策略参数

    q_table = train(alpha, gamma, epsilon)  # 训练

    test(q_table)  # 测试
