import time
from collections import deque

import numpy as np
import torch
import random
import torch.nn as nn
import gymnasium as gym

import torch.nn.functional as F

env_name = 'MountainCarContinuous-v0'

actor_lr = 1e-3
critic_lr = 1e-3
hidden_dim = 128

gamma = 0.98
tau = 0.005  # use for soft update
sigma = 0.01  # noise

buffer_size = 10000
batch_size = 64
mini_size = 512


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_bound):
        super(Actor, self).__init__()
        self.action_bound = action_bound
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x) * self.action_bound


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.model(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def add(self, data):
        self.buffer.append(data)

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)

        batch_data = {
            'state': np.array(state),
            'action': action,
            'reward': reward,
            'next_state': np.array(next_state),
            'done': done
        }

        return batch_data

    def size(self):
        return len(self.buffer)


class DDPG:
    def __init__(self, state_dim, action_dim, action_bound):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound

        self.actor = Actor(state_dim, action_dim, action_bound)
        self.critic = Critic(state_dim, action_dim)
        self.target_actor = Actor(state_dim, action_dim, action_bound)
        self.target_critic = Critic(state_dim, action_dim)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.replay_buffer = ReplayBuffer(buffer_size)

    def take_action(self, state, noise=True):
        # 保证 state 有 batch 维度
        state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        action = self.actor(state_tensor).detach().cpu().numpy()[0]  # 取得第一维输出
        if noise:
            noise_val = sigma * np.random.randn(self.action_dim)
            action += noise_val
        # 可选：裁剪动作到合法范围
        action = np.clip(action, -self.action_bound, self.action_bound)
        return action

    def soft_update(self, net, target_net):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def update(self):
        data_dict = self.replay_buffer.sample(batch_size)
        state = torch.tensor(data_dict['state'], dtype=torch.float)
        action = torch.tensor(data_dict['action'], dtype=torch.float)
        reward = torch.tensor(data_dict['reward'], dtype=torch.float).unsqueeze(1)
        next_state = torch.tensor(data_dict['next_state'], dtype=torch.float)
        done = torch.tensor(data_dict['done'], dtype=torch.float).unsqueeze(1)

        q_val = self.critic(state, action)

        next_action = self.target_actor(next_state)
        next_q = self.target_critic(next_state, next_action)

        td_target = reward + gamma * next_q * (1 - done)
        critic_loss = F.mse_loss(q_val, td_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -torch.mean(self.critic(state, self.actor(state)))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.actor, self.target_actor)
        self.soft_update(self.critic, self.target_critic)


def main():
    env = gym.make(env_name)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high[0]

    agent = DDPG(state_dim, action_dim, action_bound)

    num_episodes = 500

    for episode in range(num_episodes):
        state = env.reset()[0]
        done = False
        tot_reward = 0
        while not done:
            action = agent.take_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.replay_buffer.add((state, action, reward, next_state, done))
            state = next_state
            tot_reward += reward

            if agent.replay_buffer.size() > mini_size:
                agent.update()
        if episode % 50 == 0:
            print(f"Episode: {episode}, Episode Reward: {tot_reward}")
            # tot_reward = 0

    env = gym.make(env_name, render_mode='human')
    state = env.reset()[0]
    done = False
    while not done:
        action = agent.take_action(state, noise=False)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = next_state
        print(' Test ---- action, reward, obs, done: ', action, reward, state, done)
        time.sleep(0.05)


if __name__ == '__main__':
    main()
