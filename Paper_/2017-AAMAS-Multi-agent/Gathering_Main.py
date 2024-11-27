import csv
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

import matplotlib.pyplot as plt

from Gathering_Env import GameEnv


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.model(x)


class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=1e-4, gamma=0.99, epsilon_start=1.0,
                 epsilon_end=0.1, epsilon_decay=100000, batch_size=64, memory_size=10000):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon_start  # 探索概率
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.learn_step_counter = 0
        self.replace_target_iter = 50  # 每 1000 步更新目标网络
        self.device = torch.device("mps")
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.eval_net = DQN(state_size, action_size).to(self.device)
        self.target_net = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=learning_rate)
        self.loss_func = nn.MSELoss()

        self.epsilon_step = (epsilon_start - epsilon_end) / epsilon_decay if epsilon_decay != 0 else 0

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if random.random() < self.epsilon:
            action = random.randrange(self.action_size)
        else:
            with torch.no_grad():
                q_values = self.eval_net(state)
                action = q_values.argmax().item()
        return action

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        # 从回放记忆中采样一个小批量
        batch = random.sample(self.memory, self.batch_size)
        state_batch = torch.FloatTensor(np.array([item[0] for item in batch])).to(self.device)
        action_batch = torch.LongTensor([item[1] for item in batch]).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor([item[2] for item in batch]).unsqueeze(1).to(self.device)
        next_state_batch = torch.FloatTensor(np.array([item[3] for item in batch])).to(self.device)
        done_batch = torch.FloatTensor([item[4] for item in batch]).unsqueeze(1).to(self.device)

        # 计算当前 Q 值
        q_eval = self.eval_net(state_batch).gather(1, action_batch)
        # 计算下一个状态的最大 Q 值
        q_next = self.target_net(next_state_batch).detach()
        q_target = reward_batch + self.gamma * q_next.max(1)[0].unsqueeze(1) * (1 - done_batch)

        # 计算损失
        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 定期更新目标网络
        self.learn_step_counter += 1
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())

        # 减少 epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_step
        else:
            self.epsilon = self.epsilon_min

    # 保存模型
    def save_model(self, filepath):
        torch.save(self.eval_net.state_dict(), filepath)

    # 加载模型
    def load_model(self, filepath):
        self.eval_net.load_state_dict(torch.load(filepath))
        self.target_net.load_state_dict(self.eval_net.state_dict())

    # 设置 epsilon 值
    def set_epsilon(self, epsilon):
        self.epsilon = epsilon


def train():
    env = GameEnv(agent_hidden=N_target, food_hidden=N_apple)  # 设置每个 episode 的最大步数
    state_size = env.get_state().shape[0]
    action_size = 8  # 有 8 个可能的动作

    agent1 = DQNAgent(state_size, action_size)
    agent2 = DQNAgent(state_size, action_size)

    num_episodes = 1000

    rewards_agent1 = []
    rewards_agent2 = []

    model_dir = f"Models_{N_target}_{N_apple}"
    os.makedirs(model_dir, exist_ok=True)

    for episode in range(num_episodes):
        env.reset()
        state = env.get_state()
        total_reward1 = 0
        total_reward2 = 0
        done = False
        step = 0
        while not done:
            action1 = agent1.select_action(state)
            action2 = agent2.select_action(state)

            # 在环境中执行动作
            next_state, reward1, done, info = env.step(action1, action2)
            reward2 = info["agent2_reward"]

            agent1.store_transition(state, action1, reward1, next_state, done)
            agent2.store_transition(state, action2, reward2, next_state, done)

            # 代理学习
            agent1.learn()
            agent2.learn()

            state = next_state
            total_reward1 += reward1
            total_reward2 += reward2

            step += 1

            if episode > 300:
                temp = env.render_env()
                plt.imshow(temp)
                plt.title(f"Train Episode {episode}")
                plt.show(block=False)
                plt.pause(0.01)
                plt.clf()

        rewards_agent1.append(total_reward1)
        rewards_agent2.append(total_reward2)

        print(
            f"Train --------------- Episode {episode + 1}/{num_episodes}, Agent1 Reward: {total_reward1}, Agent2 Reward: {total_reward2}, Epsilon: {agent1.epsilon:.2f}")
        # 每 50 episode 保存一次模型
        if (episode + 1) % 200 == 0:
            agent1.save_model(os.path.join(model_dir, f"agent1_episode_{episode + 1}.pth"))
            agent2.save_model(os.path.join(model_dir, f"agent2_episode_{episode + 1}.pth"))
            print(f"Models saved at episode {episode + 1}")

    # 训练结束后保存最终模型
    agent1.save_model(os.path.join(model_dir, "agent1_final.pth"))
    agent2.save_model(os.path.join(model_dir, "agent2_final.pth"))
    print("Final models saved.")

    # 保存训练奖励到 CSV 文件
    rewards_file = os.path.join(model_dir, "training_rewards.csv")
    with open(rewards_file, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Episode", "Agent1_Total_Reward", "Agent2_Total_Reward"])
        for i in range(num_episodes):
            writer.writerow([i+1, rewards_agent1[i], rewards_agent2[i]])
    print(f"Training rewards saved to {rewards_file}")

    # 绘制奖励曲线
    # plt.plot(rewards_agent1, label='Agent 1')
    # plt.plot(rewards_agent2, label='Agent 2')
    # plt.xlabel('Episode')
    # plt.ylabel('Total Reward')
    # plt.legend()
    # plt.show()


def test():
    env = GameEnv(agent_hidden=N_target, food_hidden=N_apple)
    state_size = env.get_state().shape[0]
    action_size = 8  # 有 8 个可能的动作

    agent1 = DQNAgent(state_size, action_size)
    agent2 = DQNAgent(state_size, action_size)

    model_dir = f"Models_{N_target}_{N_apple}"
    # 加载训练好的模型
    agent1.load_model(os.path.join(model_dir, "agent1_final.pth"))
    agent2.load_model(os.path.join(model_dir, "agent2_final.pth"))

    agent1.set_epsilon(0.05)
    agent2.set_epsilon(0.05)

    num_test_episodes = 2  # 测试的回合数

    test_rewards_agent1 = []
    test_rewards_agent2 = []

    for episode in range(num_test_episodes):
        env.reset()
        state = env.get_state()
        total_reward1 = 0
        total_reward2 = 0
        done = False
        step = 0
        while not done:
            action1 = agent1.select_action(state)
            action2 = agent2.select_action(state)

            # 在环境中执行动作
            next_state, reward1, done, info = env.step(action1, action2)
            reward2 = info["agent2_reward"]

            state = next_state
            total_reward1 += reward1
            total_reward2 += reward2

            step += 1

            # 可选：渲染环境，观察代理的行为
            # temp = env.render_env()
            # plt.imshow(temp)
            # plt.title(f"Test Episode {episode + 1}")
            # plt.axis('off')
            # plt.show(block=False)
            # plt.pause(0.01)
            # plt.clf()

        test_rewards_agent1.append(total_reward1)
        test_rewards_agent2.append(total_reward2)

        print(
            f"Test --------------- Episode {episode + 1}/{num_test_episodes}, Agent1 Reward: {total_reward1}, Agent2 Reward: {total_reward2}")

    # 将测试的奖励保存到 CSV 文件中
    csv_file = os.path.join(model_dir, "test_rewards.csv")
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Episode', 'Agent1 Reward', 'Agent2 Reward'])
        for i in range(num_test_episodes):
            writer.writerow([i+1, test_rewards_agent1[i], test_rewards_agent2[i]])

    print(f"Test rewards saved to {csv_file}")



if __name__ == "__main__":
    N_target = 55
    N_apple = 55

    train()

    # test()

"""
1. 总体结果
2. 调整agent_hidden, apple_hidden 参数（9个）
3. 训练完成后平移apple位置，测试训练效果
4. 每个agent收益曲线（可以和 2 一起生成）
"""
