"""
    BUG:
        This code is not working properly.
        Maybe the problem is in the environment or network.
"""

import sys

sys.path.append("../..")
from env.grid_world import GridWorld

import torch
from torch import nn
from torch.optim import Adam
import torch.nn.functional as F
import numpy as np

from torch.distributions import Categorical

lr = 1e-3
num_episodes = 1000
max_steps = 300
epsilon = 0.1
gamma = 0.95

# [(0, 1), (1, 0), (0, -1), (-1, 0), (0, 0)]  # 下， 右， 上， 左， 原地

ref = {
    0: "down",
    1: "right",
    2: "up",
    3: "left",
    4: "nothing"
}


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(Actor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1),  # change the probability output to be in the range [0, 1]
        )

    def forward(self, x):
        return Categorical(self.model(x))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(Critic, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        state_vec = change_state(state)
        action_vec = change_state(action, len_space=self.action_dim)
        x = torch.cat((state_vec, action_vec), dim=0)
        return self.model(x)


def change_state(state, len_space=25):
    state_vec = torch.zeros(len_space)
    state_vec[state] = 1
    return state_vec


def train(env, actor, critic):
    optimizer_actor = Adam(actor.parameters(), lr=lr)
    optimizer_critic = Adam(critic.parameters(), lr=lr)

    tot_reward = 0
    for episode in range(num_episodes):
        env.reset()


        for t in range(max_steps):
            pos = env.agent_state[1] * env.env_size[0] + env.agent_state[0]

            actor_dict = actor(change_state(pos))
            action_idx, action_probs = actor_dict.sample(), actor_dict.probs

            next_state, reward, done, info = env.step(env.action_space[action_idx.item()])
            tot_reward += reward

            if done:
                print(f"\t\tEpisode: {episode}, finished after {t} timesteps")
                break

            next_state = env.agent_state[1] * env.env_size[0] + env.agent_state[0]
            ne_actor_dict = actor(change_state(next_state))
            ne_action_idx, ne_action_probs = ne_actor_dict.sample(), ne_actor_dict.probs


            val_state = critic(pos, action_idx.item())
            val_next_state = critic(next_state, ne_action_idx.item())
            TD_target = reward + gamma * val_next_state * (1 - done)

            actor_loss = -torch.log(action_probs[action_idx]) * val_state.detach()
            critic_loss = F.mse_loss(val_state, TD_target)

            optimizer_actor.zero_grad()
            optimizer_critic.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            optimizer_actor.step()
            optimizer_critic.step()

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{num_episodes} completed. Total reward: {tot_reward}")
            tot_reward = 0


def test(env, net):
    num_state = env.num_states
    num_action = len(env.action_space)
    env.reset()
    with torch.no_grad():

        policy_matrix = torch.zeros((num_state, num_action))
        for i in range(num_state):
            actor_dict = net(change_state(i))
            logits = actor_dict.probs
            best_action = torch.argmax(logits)
            policy_matrix[i][best_action] = 1.0

        for t in range(15):
            env.render(animation_interval=0.5)

            pos = env.agent_state[1] * env.env_size[0] + env.agent_state[0]

            actor_dict = net(change_state(pos))
            action_idx, action_probs = actor_dict.sample(), actor_dict.probs

            next_state, reward, done, _ = env.step(env.action_space[action_idx.item()])
            print(
                f"Step: {t}, Action: {env.action_space[action_idx.item()]}, State: {next_state + (np.array([1, 1]))}, Reward: {reward}, Done: {done}")
            if done:
                break

    return policy_matrix.numpy()


def main():
    env = GridWorld()
    num_state = env.num_states
    num_action = len(env.action_space)

    actor_net = Actor(num_state, num_action)

    critic_net = Critic(num_state, num_action, hidden_dim=64)

    train(env, actor_net, critic_net)  # TODO

    # sys.exit()
    policy_matrix = test(env, actor_net)  # TODO

    # Render the environment
    print(f"Policy Matrix: \n{policy_matrix}")
    env.add_policy(policy_matrix)
    env.render(animation_interval=2)


if __name__ == '__main__':
    main()
