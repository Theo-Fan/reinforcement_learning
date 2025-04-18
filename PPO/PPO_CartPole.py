import gymnasium as gym
import numpy as np
import torch
import sys
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import Categorical
import torch.nn.functional as F

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
        return self.model(x)


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

        self.actor_lr = 1e-3
        self.critic_lr = 1e-2
        self.gamma = 0.98
        self.gae_lmbda = 0.95
        self.clip = 0.2
        self.batch_size = 64
        self.epoch = 10

        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n

        self.actor = Actor(self.state_dim, self.action_dim)
        self.critic = Critic(self.state_dim)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.critic_lr)

    def take_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).view(1, -1)
        dist = Categorical(self.actor(state_tensor))
        dist_prob = dist.probs
        action = dist.sample()
        return action.item(), dist_prob

    def calc_gae(self, TD_error): # Generalized Advantage Estimation
        TD_error = TD_error.detach().numpy()
        advantage_lst = []
        advantage = 0.0
        for delta in TD_error[::-1]:
            advantage = self.gamma * self.gae_lmbda * advantage + delta
            advantage_lst.append(advantage)

        advantage_lst.reverse()
        return torch.tensor(advantage_lst, dtype=torch.float32)

    def learn(self, transition_dict):
        # TODO implement learn method
        # 1. calculate advantage
        # 2. update actor and critic networks

        states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float32)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.int64).view(-1, 1)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float32).view(-1, 1)
        next_states = torch.tensor(np.array(transition_dict['next_states']), dtype=torch.float32)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float32).view(-1, 1)

        TD_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        TD_error = TD_target - self.critic(states)

        advantage = self.calc_gae(TD_error)
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()

        # TODO implement Importance Sampling for PPO
        for _ in range(self.epoch):
            log_probs = torch.log(self.actor(states).gather(1, actions))
            ratio = torch.exp(log_probs - old_log_probs)  # TODO importance sampling ratio
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


def train(env, num_episodes, ppo_agent):
    for episode in range(1, num_episodes + 1):
        state = env.reset()[0]

        episode_reward = 0
        transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}

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
            print(f"Episode：{episode}/{num_episodes}，reward：{episode_reward:.2f}")
    print(f"Training finished!")

def test(env_name, ppo_agent):
    env = gym.make(env_name, render_mode='human')
    state = env.reset()[0]

    done = False
    episode_reward = 0
    while not done:
        action, probs = ppo_agent.take_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = next_state
        episode_reward += reward
    print(f"Test reward: {episode_reward:.2f}")


def main():
    env_name = 'CartPole-v1'
    env = gym.make(env_name)
    num_episodes = 300
    ppo_agent = Agent(env)

    train(env, num_episodes, ppo_agent)

    test(env_name, ppo_agent)


if __name__ == '__main__':
    main()
