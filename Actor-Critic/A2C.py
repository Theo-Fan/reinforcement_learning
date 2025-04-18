import sys

import gymnasium as gym
from gymnasium.spaces.utils import flatten_space
import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, state_dim, num_action, hidden_dim=128):
        super(Actor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_action),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return Categorical(self.model(x))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(Critic, self).__init__()
        self.action_dim = action_dim
        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action_dix):
        # one-hot encoding the action
        action = torch.zeros(self.action_dim)
        action[action_dix] = 1
        # concat state and action
        x = torch.cat([state, action], dim=-1)
        return self.model(x)


def train(env, actor, critic):
    # hyperparameters
    num_episodes = 1500
    gamma = 0.98
    lr = 1e-3
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=lr)

    for episode in range(num_episodes):
        state, _ = env.reset(seed=42)  # state: [ 0.0273956  -0.00611216  0.03585979  0.0197368 ]

        log_probs = []
        rewards = []
        vals = []
        ne_vals = []
        dones = []

        episode_reward = 0
        done = False
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32)

            action_dict = actor(state_tensor)
            action = action_dict.sample()

            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            next_state_tensor = torch.tensor(next_state, dtype=torch.float32)
            ne_action_dict = actor(next_state_tensor)
            ne_action = ne_action_dict.sample()

            episode_reward += reward
            log_probs.append(action_dict.log_prob(action))
            vals.append(critic(state_tensor, action.item()))
            ne_vals.append(critic(next_state_tensor, ne_action.item()))
            rewards.append(reward)
            dones.append(done)

            state = next_state

        # calc return [Monte Carlo]
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        vals = torch.cat(vals).squeeze()
        ne_vals = torch.cat(ne_vals).squeeze()
        log_probs = torch.stack(log_probs)
        dones = torch.tensor(dones, dtype=torch.float32)
        advantage = returns - vals.detach()

        TD_target = rewards + gamma * ne_vals * (1 - dones)
        TD_error = TD_target.detach() - vals.detach()

        # The q-val unefficiently，using TD error, return are better
        actor_loss = (-log_probs * advantage).mean()

        critic_loss = F.mse_loss(vals, TD_target)

        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        if episode % 20 == 0:
            print(f"Episode {episode}/{num_episodes}, Reward: {episode_reward}")


def test(actor):
    env = gym.make('CartPole-v1', render_mode="human")
    state, _ = env.reset()
    done = False
    total_reward = 0
    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32)
        action_dict = actor(state_tensor)
        action = action_dict.sample()

        next_state, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated

        total_reward += reward
        state = next_state

    print(f"Test Reward: {total_reward}")
    return total_reward


def main():
    env = gym.make('CartPole-v1')

    state_dim = env.observation_space.shape[0]
    action_dim = flatten_space(env.action_space).shape[0]  # change type to Box

    print(state_dim, action_dim)

    actor = Actor(state_dim, action_dim)
    critic = Critic(state_dim, action_dim)

    train(env, actor, critic)

    test(actor)

    env.close()


if __name__ == '__main__':
    main()
