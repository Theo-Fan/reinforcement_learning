import collections
import random
import sys

sys.path.append("../..")
from env.grid_world import GridWorld
import torch
import numpy as np
import torch.nn as nn

"""
    In this script, we use a neural network to approximate the q-table.
    We will clearly find: 
        Compare with q-table, the neural network can't optimize all the states[Exist some state are not take best action]
        In other words, the aim of function approximation is find a good path, not all the good path.
"""

gamma = 0.9
num_episode = 500
epsilon = 0.1
target_update_freq = 10
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device("cpu")


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


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.net(x)


def epsilon_greedy_network(state, net, action_space, is_epsilon=True):
    state = change_state(state)
    q_val = net(state)
    action_idx = torch.argmax(q_val).item()
    if is_epsilon and np.random.uniform(0, 1) < epsilon:
        tmp_action = [i for i in range(len(action_space)) if i != action_idx]
        action_idx = np.random.choice(tmp_action)
    action = action_space[action_idx]

    return action, action_idx


def change_state(pos, num_state=25):
    state_vec = torch.zeros(num_state, )
    state_vec[pos] = 1.0

    return state_vec


def net_update(data, q_net, target_net, loss_fn, optimizer):
    state = torch.stack([change_state(i) for i in data['state']])
    next_state = torch.stack([change_state(i) for i in data['next_state']])
    action = torch.tensor(data['action'], dtype=torch.int64).view(-1, 1)
    reward = torch.tensor(data['reward'], dtype=torch.int64).view(-1, 1)
    done = torch.tensor(data['done'], dtype=torch.float32).view(-1, 1)

    q_val = q_net(state).gather(1, action)

    next_q_val = target_net(next_state).max(1)[0].view(-1, 1)
    TD_target = reward + (1 - done) * gamma * next_q_val

    # ======>  we use the TD-error as the loss  <=======
    loss = loss_fn(q_val, TD_target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()



def train(env, q_net, target_net):
    # loss function
    loss_fn = torch.nn.MSELoss()

    # optimizer
    lr = 1e-3
    optimizer = torch.optim.Adam(q_net.parameters(), lr)

    replay_buffer = ReplayBuffer(capacity=10000)

    for episode in range(num_episode):
        env.reset()

        for t in range(100):

            # state
            pos = env.agent_state[1] * env.env_size[0] + env.agent_state[0]

            # action
            action, action_idx = epsilon_greedy_network(pos, q_net, env.action_space)

            next_state, reward, done, info = env.step(action)

            # next state
            ne_pos = next_state[1] * env.env_size[0] + next_state[0]

            # add to replay buffer
            replay_buffer.add(pos, action_idx, reward, ne_pos, done)

            if replay_buffer.size() > replay_buffer.mini_size:
                data = replay_buffer.sample(batch_size=64)
                net_update(data, q_net, target_net, loss_fn, optimizer)  # TODO

            if t % target_update_freq == 0:
                # update target network
                target_net.load_state_dict(q_net.state_dict())

            if done:
                print(f"\t\tEpisode: {episode}, finished after {t} timesteps")
                break

        if episode % 100 == 0:
            print(f"Episode: {episode}/{num_episode}")


def test(env, q_net):  # TODO when test, epsilon need to set 0
    num_state = env.num_states
    num_action = len(env.action_space)
    env.reset()
    with torch.no_grad():

        policy_matrix = torch.zeros((num_state, num_action))
        for i in range(num_state):
            state_vec = torch.zeros(num_state, )
            state_vec[i] = 1.0
            best_action = torch.argmax(q_net(state_vec))
            policy_matrix[i][best_action] = 1.0

        for t in range(20):
            env.render(animation_interval=0.5)
            state = env.agent_state

            pos = env.agent_state[1] * env.env_size[0] + env.agent_state[0]
            state_vec = change_state(pos)

            action, action_idx = epsilon_greedy_network(pos, q_net, env.action_space, is_epsilon=False)

            next_state, reward, done, info = env.step(action)
            print(
                f"Step: {t}, Action: {action}, Cur-state: {state}, Re-pos: {pos}, "
                f"Idx: {np.argmax(q_net(state_vec))}, Next-state: {next_state}, Reward: {reward}, Done: {done}")
            if done:
                break

    return policy_matrix.numpy()


def main():
    env = GridWorld()
    num_state = env.num_states
    num_action = len(env.action_space)

    q_network = QNetwork(num_state, num_action)
    target_network = QNetwork(num_state, num_action)

    train(env, q_network, target_network)

    policy_matrix = test(env, q_network)

    # Render the environment
    print(f"Policy Matrix: \n{policy_matrix}")
    env.add_policy(policy_matrix)
    env.render(animation_interval=10)


if __name__ == "__main__":
    main()
