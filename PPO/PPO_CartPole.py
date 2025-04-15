from re import findall

import gymnasium as gym
import torch
import torch.nn as nn
from torch.distributions import Categorical


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
        dist = Categorical(t)
        entropy = dist.entropy()
        return dist, entropy


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


class PPOmemory:
    def __init__(self, capacity):
        self.capacity = capacity
        # TODO add other info for estimating advantage

    def push(self, data):
        # TODO implement push method
        pass


class Agent:
    def __init__(self, env):
        self.env = env

        self.memory_capacity = 5

        self.gamma = 0.98
        self.lr = 1e-3
        self.gae_lmbda = 0.95
        self.polciy_clip = 0.2
        self.batch_size = 64

        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = int(findall(r"\d+\.?\d*", str(self.env.action_space))[0])

        self.actor = Actor(self.state_dim, self.action_dim)
        self.critic = Critic(self.state_dim)

        self.memory = PPOmemory(self.memory_capacity)

    def take_action(self, state):
        pass

    def learn(self):
        pass


def main():
    env_name = 'CartPole-v1'
    env = gym.make(env_name)

    num_episodes = 200
    max_steps = 500

    state_dim = env.observation_space.shape[0]
    action_dim = int(findall(r"\d+\.?\d*", str(env.action_space))[0])  # 获取动作维度

    ppo_agent = Agent(env)

    for episode in range(1, num_episodes + 1):
        state = env.reset()[0]

        episode_reward = 0
        for step in range(1, max_steps + 1):
            action = ppo_agent.take_action(state) # TODO

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward

            ppo_agent.memory.push( ) # TODO


            ppo_agent.learn()

            if done:
                break

            state = next_state
        if episode % 10 == 0:
            print(f"回合：{episode}/{num_episodes}，奖励：{episode_reward:.2f}")
    print('完成训练！')






if __name__ == '__main__':
    main()
