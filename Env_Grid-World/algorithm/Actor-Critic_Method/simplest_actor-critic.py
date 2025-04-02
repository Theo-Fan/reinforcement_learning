import sys

sys.path.append("../..")
from env.grid_world import GridWorld

import torch
from torch import nn
from torch.optim import Adam
import numpy as np

lr = 1e-3
num_episodes = 1000
max_steps = 100
gamma = 0.95


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)  # change the probability output to be in the range [0, 1]
        )

    def forward(self, x):
        return self.model(x)


class CriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(CriticNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Output a single value for the state-action pair
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.model(x)


def change_state(state, num_states=25):
    state_vec = torch.zeros(num_states)
    state_vec[state] = 1
    return state_vec


def change_action(action, num_actions=5):
    action_vec = torch.zeros(num_actions)
    action_vec[action] = 1
    return action_vec


def net_update(transition_dict, policy_net, critic_net, policy_optimizer, critic_optimizer, critic_loss):
    state_vec = torch.stack(transition_dict['states'])
    action_vec = torch.stack(transition_dict['actions'])
    ne_state_vec = torch.stack(transition_dict['next_states'])
    ne_action_vec = torch.stack(transition_dict['next_actions'])
    reward = torch.tensor(transition_dict['rewards'], dtype=torch.float32).view(-1, 1)
    done = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1)
    action_probs = torch.stack(transition_dict['action_probs'])
    action_idx = torch.tensor(transition_dict['action_idxs'], dtype=torch.long).view(-1, 1)

    # print(type(action_vec))
    # sys.exit()

    # critic action
    q_val = critic_net(state_vec, action_vec)
    ne_q_val = critic_net(ne_state_vec, ne_action_vec)

    # TD target & error
    TD_target = reward + gamma * ne_q_val * (1 - done)
    TD_error = TD_target - q_val

    # Value update(Critic Update)
    value_loss = torch.mean(critic_loss(q_val, TD_target.detach()))

    # Policy update(Actor Update)
    policy_loss = torch.mean(-torch.log(action_probs.gather(1, action_idx)) * q_val.detach())
    policy_optimizer.zero_grad()
    critic_optimizer.zero_grad()
    policy_loss.backward()
    value_loss.backward()
    policy_optimizer.step()
    critic_optimizer.step()


def train(env, policy_net, critic_net):
    policy_optimizer = Adam(policy_net.parameters(), lr=lr)
    critic_optimizer = Adam(critic_net.parameters(), lr=lr)
    critic_loss = nn.MSELoss()

    for episode in range(num_episodes):
        env.reset()

        transition_dict = {'states': [], 'actions': [], 'next_states': [],
                           'next_actions': [], 'rewards': [], 'dones': [],
                           'action_idxs': [], 'action_probs': []}

        episode_return = 0
        for step in range(max_steps):
            # state
            pos = env.agent_state[1] * env.env_size[0] + env.agent_state[0]
            state_vec = change_state(pos)

            # sampling action
            action_probs = policy_net(state_vec)
            action_dist = torch.distributions.Categorical(action_probs)
            action_idx = action_dist.sample()

            # print(f"Action Probs: {action_probs} Action: {action_idx}")

            ne_state, reward, done, info = env.step(env.action_space[action_idx])

            # next state
            ne_pos = env.agent_state[1] * env.env_size[0] + env.agent_state[0]
            ne_state_vec = change_state(ne_pos)

            # sampling next action
            ne_action_probs = policy_net(ne_state_vec)
            ne_action_dist = torch.distributions.Categorical(ne_action_probs)
            ne_action_idx = ne_action_dist.sample()

            # change action to one-hot vector
            action_vec = change_action(action_idx.item())
            ne_action_vec = change_action(ne_action_idx.item())

            # store transition
            transition_dict['states'].append(state_vec)
            transition_dict['actions'].append(action_vec)
            transition_dict['next_states'].append(ne_state_vec)
            transition_dict['next_actions'].append(ne_action_vec)
            transition_dict['rewards'].append(reward)
            transition_dict['dones'].append(done)
            transition_dict['action_idxs'].append(action_idx.item())
            transition_dict['action_probs'].append(action_probs)

            episode_return += reward

            if done:
                break

        net_update(transition_dict, policy_net, critic_net, policy_optimizer, critic_optimizer, critic_loss)

        if episode % 100 == 0:
            print(f"Episode {episode}/{num_episodes} completed.")


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

        for t in range(15):
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
    num_state = env.num_states
    num_action = len(env.action_space)

    policy_net = PolicyNetwork(num_state, num_action)
    critic_net = CriticNetwork(num_state, num_action)

    train(env, policy_net, critic_net)  # TODO

    policy_matrix = test(env, policy_net)  # TODO

    # Render the environment
    print(f"Policy Matrix: \n{policy_matrix}")
    env.add_policy(policy_matrix)
    env.render(animation_interval=2)


if __name__ == '__main__':
    main()
