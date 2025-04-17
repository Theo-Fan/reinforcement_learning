from re import findall

import gymnasium as gym
import torch
import sys
import torch.nn as nn
from torch.optim import Adam
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
        # entropy = dist.entropy()
        return dist


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


# this buffer is similar to the replay buffer,
# It used for calc advantage and update actor/critic nets
class PPOmemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, prob):
        # TODO implement push method
        self.buffer.append((state, action, reward, next_state, done, prob))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)

        batch_data = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'probs': probs,
        }

        return batch_data

    def size(self):
        return len(self.buffer)


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

        self.memory = PPOmemory(self.memory_capacity)

    def take_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).view(1, -1)
        dist = self.actor(state_tensor)
        dist_prob = dist.probs
        action = dist.sample()
        return action.item(), dist_prob

    def learn(self):
        # TODO implement learn method
        # 1. calculate advantage
        # 2. update actor and critic networks
        # 3. clear memory
        pass


def main():
    env_name = 'CartPole-v1'
    env = gym.make(env_name)

    num_episodes = 200
    max_steps = 500

    state_dim = env.observation_space.shape[0]
    action_dim = int(findall(r"\d+\.?\d*", str(env.action_space))[0])  # 获取动作维度

    print(f"state dim: {state_dim}\naction dim: {action_dim}")

    ppo_agent = Agent(env)

    for episode in range(1, num_episodes + 1):
        state = env.reset()[0]
        episode_reward = 0

        for step in range(1, max_steps + 1):
            action, probs = ppo_agent.take_action(state)  # TODO

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward

            ppo_agent.memory.push(state, action, reward, next_state, done, probs)  # TODO

            if done:
                break
            state = next_state

        ppo_agent.learn()
        if episode % 10 == 0:
            print(f"回合：{episode}/{num_episodes}，奖励：{episode_reward:.2f}")
    print('完成训练！')


if __name__ == '__main__':
    main()
