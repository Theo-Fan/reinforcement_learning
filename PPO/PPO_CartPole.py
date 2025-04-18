from re import findall

import gymnasium as gym
import torch
import sys
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import Categorical
import torch.nn.functional as F
import collections
import random


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(Actor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        t = self.model(x)
        # entropy = dist.entropy()
        return t


class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim=64):
        super(Critic, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.model(x)


class Agent:
    def __init__(self, env):
        self.env = env

        self.memory_capacity = 5

        self.gamma = 0.98
        self.lr = 1e-3
        self.gae_lmbda = 0.95
        self.clip = 0.2
        self.batch_size = 64

        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = int(findall(r"\d+\.?\d*", str(self.env.action_space))[0])

        self.actor = Actor(self.state_dim, self.action_dim)
        self.critic = Critic(self.state_dim)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.lr)

    def take_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).view(1, -1)
        dist = Categorical(self.actor(state_tensor))
        dist_prob = dist.probs
        action = dist.sample()
        return action.item(), dist_prob

    def calc_GAE(self, TD_error):
        advantage_lst = []
        advantage = 0.0
        # print(len(TD_error))
        for delta in TD_error.flip(0):
            advantage = self.gamma * self.gae_lmbda * advantage + delta
            advantage_lst.append(advantage)

        advantage_lst.reverse()
        return torch.tensor(advantage_lst, dtype=torch.float32)

    def learn(self, data):
        # TODO implement learn method
        # 1. calculate advantage
        # 2. update actor and critic networks

        states = torch.tensor(data['states'], dtype=torch.float32)
        actions = torch.tensor(data['actions'], dtype=torch.int64).view(-1, 1)
        rewards = torch.tensor(data['rewards'], dtype=torch.float32).view(-1, 1)
        next_states = torch.tensor(data['next_states'], dtype=torch.float32)
        dones = torch.tensor(data['dones'], dtype=torch.float32).view(-1, 1)

        TD_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        TD_error = TD_target - self.critic(states)

        # print(TD_error)

        advantage = self.calc_GAE(TD_error)

        old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()

        # TODO implement Importance Sampling for PPO
        for _ in range(10):
            log_porbs = torch.log(self.actor(states).gather(1, actions))
            ratio = torch.exp(log_porbs - old_log_probs)  # TODO importance sampling ratio
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * advantage  # clipped surrogate

            actor_loss = torch.mean(-torch.min(surr1, surr2))
            critic_loss = torch.mean(F.mse_loss(self.critic(states), TD_target.detach()))

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()


def main():
    env_name = 'CartPole-v1'
    env = gym.make(env_name)

    num_episodes = 500

    state_dim = env.observation_space.shape[0]
    action_dim = int(findall(r"\d+\.?\d*", str(env.action_space))[0])  # 获取动作维度

    print(f"state dim: {state_dim}\naction dim: {action_dim}")

    ppo_agent = Agent(env)

    for episode in range(1, num_episodes + 1):
        state = env.reset()[0]

        transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}

        episode_reward = 0

        done = False
        while not done:
            action, probs = ppo_agent.take_action(state)  # TODO

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            transition_dict['states'].append(state)
            transition_dict['actions'].append(action)
            transition_dict['next_states'].append(next_state)
            transition_dict['rewards'].append(reward)
            transition_dict['dones'].append(done)

            episode_reward += reward
            state = next_state

        ppo_agent.learn(transition_dict)
        if episode % 10 == 0:
            print(f"回合：{episode}/{num_episodes}，奖励：{episode_reward:.2f}")
    print('完成训练！')


if __name__ == '__main__':
    main()
