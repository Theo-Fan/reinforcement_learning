from re import findall

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt


def take_action(state):
    if np.random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[state, :])
    return action


def train():
    rewards = []
    for episode in range(num_episodes):
        state, _ = env.reset()
        # 根据 epsilon-greedy 策略选择初始动作
        action = take_action(state)
        done = False
        total_reward = 0

        while not done:
            # 执行动作，获取新的状态和奖励
            next_state, reward, done, truncated, info = env.step(action)
            total_reward += reward

            next_action = take_action(next_state)

            # 更新 Q 值
            Q[state, action] += alpha * (reward + gamma * Q[next_state, next_action] - Q[state, action])

            # 更新状态和动作
            state = next_state
            action = next_action

        rewards.append(total_reward)

    print("训练完成！")
    print("Final Q-table:\n", Q)
    print(Q.shape)

    # 绘制奖励曲线
    # plt.plot(range(num_episodes), rewards)
    # plt.xlabel('Episode')
    # plt.ylabel('Total Reward')
    # plt.title('Training Reward over Episodes')
    # plt.show()


def test(num_tests=5):
    test_env = gym.make('Taxi-v3', render_mode='human')
    for i in range(num_tests):
        state, _ = test_env.reset()
        test_env.render()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            action = np.argmax(Q[state, :])
            state, reward, done, truncated, info = test_env.step(action)
            total_reward += reward
            steps += 1
            test_env.render()

        print(f"测试回合 {i + 1}：总奖励 = {total_reward}，总步数 = {steps}")


if __name__ == "__main__":
    env = gym.make('Taxi-v3')

    # 设置参数
    alpha = 0.1
    gamma = 0.99
    epsilon = 0.1
    num_episodes = 5000


    state_space = int(findall(r"\d+\.?\d*", str(env.observation_space))[0])
    print("State space size:", state_space)

    action_space = int(findall(r"\d+\.?\d*", str(env.action_space))[0])  # 获取动作维度
    print("Action space size:", action_space)

    Q = np.zeros([state_space, action_space])

    train()
    test(num_tests=5)
