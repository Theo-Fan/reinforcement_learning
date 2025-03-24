import sys

import torch


sys.path.append("../..")
from env.grid_world import GridWorld

import random
import numpy as np
from torch import nn

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)  # 输出概率分布
        )

    def forward(self, x):
        return self.model(x)

def train():
    net.train()
    for episode_id in range(num_episodes):
        state = env.reset()[0]
        episode = []

        for _ in range(max_steps_per_episode):
            state_vec = np.zeros(env.num_states)
            pos = env.agent_state[1] * env.env_size[0] + env.agent_state[0]
            state_vec[pos] = 1
            state_tensor = torch.tensor(state_vec, dtype=torch.float32)

            action_probs = net(state_tensor).detach().numpy()
            print(f"idx: {_} Action Probs: {action_probs}")
            action = np.random.choice(len(action_probs), p=action_probs)

            next_state, reward, done, _ = env.step(env.action_space[action])
            episode.append((pos, action, reward))



        for t in range(len(episode)):
            state_idx, action, _ = episode[t]

            G = sum([gamma ** k * r for k, (_, _, r) in enumerate(episode[t:])])

            state_vec = np.zeros(env.num_states)
            state_vec[state_idx] = 1
            state_tensor = torch.tensor(state_vec, dtype=torch.float32)
            action_probs = net(state_tensor)

            loss = -torch.log(action_probs[action]) * G

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

if __name__ == "__main__":
    env = GridWorld()
    state = env.reset()

    state_dim = env.num_states
    action_dim = len(env.action_space)
    hidden_dim = 128
    gamma = 0.95
    lr =  5e-4
    num_episodes = 1000
    max_steps_per_episode = 100

    net = PolicyNetwork(state_dim, hidden_dim, action_dim)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    train()

    state = env.reset()

    for t in range(1000):
        env.render(animation_interval=0.5)  # 渲染环境

        state_vec = np.zeros(env.num_states)
        pos = env.agent_state[1] * env.env_size[0] + env.agent_state[0]
        state_vec[pos] = 1
        state_tensor = torch.tensor(state_vec, dtype=torch.float32)
        action_probs = net(state_tensor).detach().numpy()
        action = np.random.choice(len(action_probs), p=action_probs)  # 按照概率分布采样动作

        next_state, reward, done, _ = env.step(env.action_space[action])
        print(f"Step: {t}, Action: {action}, State: {next_state + (np.array([1, 1]))}, Reward: {reward}, Done: {done}")
        if done:
            break

    # 添加随机策略矩阵
    # policy_matrix = np.random.rand(env.num_states, len(env.action_space))
    # policy_matrix /= policy_matrix.sum(axis=1)[:, np.newaxis]  # 归一化每行的概率和为 1
    #
    # env.add_policy(policy_matrix)

    # 添加状态值
    # values = np.random.uniform(0, 10, (env.num_states,))
    # env.add_state_values(values)

    # 渲染环境
    env.render(animation_interval=2)