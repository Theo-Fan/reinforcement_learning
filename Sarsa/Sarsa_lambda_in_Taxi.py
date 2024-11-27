import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt


def take_action(state):
    if np.random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[state, :])
    return action

# 训练函数
def train(Q):
    rewards = []
    for episode in range(num_episodes):
        state, _ = env.reset()
        action = take_action(state)

        done = False
        total_reward = 0
        # init eligibility trace
        E = np.zeros_like(Q)

        while not done:
            next_state, reward, done, truncated, info = env.step(action)
            total_reward += reward

            next_action = take_action(next_state)

            # calculate TD err
            err = reward + gamma * Q[next_state, next_action] - Q[state, action]

            # update eligibility trace
            # Method 1:
            # E[state, action] += 1

            # Method 2 (important):
            E[state, :] = 0
            E[state, action] = 1

            # Using eligibility trace update q-table: Q(s, a) = Q(s, a) + alpha * err * E(s, a)
            Q += alpha * err * E

            # decay eligibility trace after update
            E *= gamma * lambd

            # 更新状态和动作
            state = next_state
            action = next_action

        rewards.append(total_reward)

    print("训练完成！")

    # 绘制奖励曲线
    plt.plot(range(num_episodes), rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Reward over Episodes')
    plt.show()

# 测试函数
def test(num_tests=5):
    env = gym.make('Taxi-v3', render_mode='human')
    for i in range(num_tests):
        state, _ = env.reset()
        env.render()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            action = np.argmax(Q[state, :])
            state, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            env.render()

        print(f"测试回合 {i+1}：总奖励 = {total_reward}，总步数 = {steps}")

# 主程序
if __name__ == "__main__":
    # 创建 Taxi-v3 环境
    env = gym.make('Taxi-v3')

    # 设置参数
    alpha = 0.1  # 学习率
    gamma = 0.99  # 折扣因子
    epsilon = 0.1  # epsilon-greedy 策略中的 epsilon
    lambd = 0.9  # 资格迹的衰减率
    num_episodes = 5000  # 总训练回合数

    # 初始化 Q 表，大小为（状态数，动作数）
    Q = np.zeros([env.observation_space.n, env.action_space.n])

    train(Q)
    test(num_tests=5)