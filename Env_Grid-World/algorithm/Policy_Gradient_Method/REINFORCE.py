import sys

sys.path.append("../..")
from env.grid_world import GridWorld

import torch
import numpy as np
from torch import nn

num_episodes = 1000
max_steps = 100
gamma = 0.95
lr = 5e-4


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)  # change the probability output to be in the range [0, 1]
        )

    def forward(self, x):
        return self.model(x)


def change_state(state, num_states=25):
    state_vec = torch.zeros(num_states)
    state_vec[state] = 1
    return state_vec


def train(env, net):
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    net.train()
    for episode_id in range(num_episodes):
        env.reset()
        episode = []

        for _ in range(max_steps):
            pos = env.agent_state[1] * env.env_size[0] + env.agent_state[0]

            state_tensor = change_state(pos)
            action_probs = net(state_tensor).detach().numpy()

            # print(f"idx: {_} Action Probs: {action_probs}")
            action = np.random.choice(len(action_probs), p=action_probs)

            next_state, reward, done, _ = env.step(env.action_space[action])
            episode.append((pos, action, reward))

        for t in range(len(episode)):
            state_idx, action, _ = episode[t]

            G = sum([gamma ** k * r for k, (_, _, r) in enumerate(episode[t:])])  # calculate the return

            state_tensor = change_state(state_idx)
            action_probs = net(state_tensor)

            loss = -torch.log(action_probs[action]) * G  # policy ascent need to be negative

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if episode_id % 100 == 0:
            print(f"Episode {episode_id}/{num_episodes} completed.")


def test(env, net):
    num_state = env.num_states
    num_action = len(env.action_space)
    env.reset()
    with torch.no_grad():

        policy_matrix = torch.zeros((num_state, num_action))
        for i in range(num_state):
            state_vec = change_state(i)
            best_action = torch.argmax(net(state_vec))
            policy_matrix[i][best_action] = 1.0

        for t in range(1000):
            env.render(animation_interval=0.5)

            pos = env.agent_state[1] * env.env_size[0] + env.agent_state[0]
            state_tensor = change_state(pos)

            action_probs = net(state_tensor).detach().numpy()

            action = np.random.choice(len(action_probs), p=np.array(action_probs))  # choose action by probability

            next_state, reward, done, _ = env.step(env.action_space[action])
            print(
                f"Step: {t}, Action: {action}, State: {next_state + (np.array([1, 1]))}, Reward: {reward}, Done: {done}")
            if done:
                break

    return policy_matrix.numpy()


def main():
    env = GridWorld()
    env.reset()

    state_dim = env.num_states
    action_dim = len(env.action_space)

    net = PolicyNetwork(state_dim, action_dim)

    train(env, net)

    policy = test(env, net)

    print(f"Policy Matrix: \n{policy}")
    env.add_policy(policy)

    # 添加状态值
    # values = np.random.uniform(0, 10, (env.num_states,))
    # env.add_state_values(values)

    # Render the environment
    env.render(animation_interval=10)


if __name__ == "__main__":
    main()
