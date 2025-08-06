import collections
import random
import time

import numpy as np
import torch
import gymnasium as gym


class ReplayBuffer:
    def __init__(self, capacity):
        self.mini_size = 500
        self.capacity = capacity
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)

        batch_data = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done
        }

        return batch_data

    def size(self):
        return len(self.buffer)


class Q_Network(torch.nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(Q_Network, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(self, env):
        self.env = env
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n  # 获取动作维度

        self.hidden_dim = 128
        self.gamma = 0.98
        self.epsilon = 0.01
        self.lr = 2e-3
        self.target_update_freq = 10

        self.q_net = Q_Network(self.state_dim, self.action_dim, self.hidden_dim)
        self.target_net = Q_Network(self.state_dim, self.action_dim, self.hidden_dim)

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.lr)
        self.loss = torch.nn.MSELoss()

        self.replay_buffer = ReplayBuffer(capacity=10000)

    def take_action(self, state, is_epsilon=True):
        if is_epsilon and np.random.uniform(0, 1) < self.epsilon:
            action = self.env.action_space.sample()
        else:
            state = torch.tensor(np.array(state), dtype=torch.float32)

            action = torch.argmax(self.q_net(state)).item()  # max q-value action

        return action

    def update(self, data):
        # change date type to tensor
        state = torch.tensor(data['state'], dtype=torch.float32)
        action = torch.tensor(data['action'], dtype=torch.int64).view(-1, 1)  # shape: (batch_size, 1)
        reward = torch.tensor(data['reward'], dtype=torch.float32).view(-1, 1)
        next_state = torch.tensor(data['next_state'], dtype=torch.float32)
        done = torch.tensor(data['done'], dtype=torch.float32).view(-1, 1)

        # calculate q value
        q_val = self.q_net(state).gather(dim=1, index=action)  # get q value of action
        ne_q_val = self.target_net(next_state).max(dim=1)[0].view(-1, 1)  # for next_state, get max q value
        TD_target = reward + (1 - done) * self.gamma * ne_q_val

        # calculate loss
        loss = self.loss(q_val, TD_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def main():
    env = gym.make('CartPole-v1')
    agent = DQNAgent(env)

    episode = 500
    max_steps = 1000

    # train
    for i in range(episode):
        state, info = agent.env.reset()

        for step in range(max_steps):
            action = agent.take_action(state)

            next_state, reward, terminated, truncated, _ = agent.env.step(action)

            done = terminated or truncated
            agent.replay_buffer.add(state, action, reward, next_state, done)

            if agent.replay_buffer.size() > agent.replay_buffer.mini_size:
                data = agent.replay_buffer.sample(batch_size=64)
                agent.update(data)  # TODO

            if step % agent.target_update_freq == 0:
                agent.target_net.load_state_dict(agent.q_net.state_dict())

            state = next_state

            if done:
                break

        if i % 100 == 0:
            print(f"Episode: {i}/{episode}")

    # test
    agent.env = gym.make('CartPole-v1', render_mode='human')
    state, info = agent.env.reset()
    done = False
    while not done:
        action = agent.take_action(state, is_epsilon=False)
        next_state, reward, terminated, truncated, _ = agent.env.step(action)
        done = terminated or truncated

        state = next_state

        print(' Test ---- action, reward, obs, done: ', action, reward, state, done)
        time.sleep(0.05)


if __name__ == '__main__':
    main()
