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

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device("cpu")


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


def epsilon_greedy_network(state, net, action_space):
    q_val = net(state.to(device))
    action_idx = torch.argmax(q_val).item()
    if np.random.uniform(0, 1) < epsilon:
        tmp_action = [i for i in range(len(action_space)) if i != action_idx]
        action_idx = np.random.choice(tmp_action)
    action = action_space[action_idx]

    return action, action_idx


def train(env, q_net):
    num_state = env.num_states

    # loss function
    loss_fn = torch.nn.MSELoss()
    loss_fn.to(device)

    # optimizer
    lr = 1e-3
    optimizer = torch.optim.SGD(q_net.parameters(), lr)

    # q_net.train()
    for episode in range(num_episode):
        env.reset()

        for t in range(4000):

            # state
            pos = env.agent_state[1] * env.env_size[0] + env.agent_state[0]
            state_vec = torch.zeros(num_state, )
            state_vec[pos] = 1.0

            # action
            action, action_idx = epsilon_greedy_network(state_vec, q_net, env.action_space)

            next_state, reward, done, info = env.step(action)

            # next state
            ne_pos = next_state[1] * env.env_size[0] + next_state[0]
            ne_state_vec = torch.zeros(num_state, )
            ne_state_vec[ne_pos] = 1.0

            # next_action (For Sarsa, QLearning don't need)
            # ne_action, ne_action_idx = epsilon_greedy_network(ne_state_vec, q_net, env.action_space)

            cur_q = q_net(state_vec.to(device))[action_idx]
            next_q = torch.max(q_net(ne_state_vec.to(device)))  # different between Sarsa and QLearning
            TD_target = reward + gamma * next_q

            # ======>  we use the TD-error as the loss  <=======
            loss = loss_fn(cur_q, TD_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if done:
                print(f"\t\tEpisode: {episode}, finished after {t} timesteps")
                break

        if episode % 100 == 0:
            print(f"Episode: {episode}/{num_episode}, Loss: {loss.item()}")


def test(env, q_net): # TODO when test, epsilon need to set 0
    num_state = env.num_states
    num_action = len(env.action_space)
    env.reset()
    with torch.no_grad():

        policy_matrix = torch.zeros((num_state, num_action))
        for i in range(num_state):
            state_vec = torch.zeros(num_state, )
            state_vec[i] = 1.0
            best_action = torch.argmax(q_net(state_vec.to(device)))
            policy_matrix[i][best_action] = 1.0

        for t in range(20):
            env.render(animation_interval=0.5)
            state = env.agent_state

            pos = env.agent_state[1] * env.env_size[0] + env.agent_state[0]
            state_vec = torch.zeros(num_state, )
            state_vec[pos] = 1.0

            action, action_idx = epsilon_greedy_network(state_vec, q_net, env.action_space)

            next_state, reward, done, info = env.step(action)
            print(
                f"Step: {t}, Action: {action}, Cur-state: {state}, Re-pos: {pos}, "
                f"Idx: {np.argmax(q_net(state_vec.to(device)))}, Next-state: {next_state}, Reward: {reward}, Done: {done}")
            if done:
                break

    return policy_matrix.numpy()


def main():
    env = GridWorld()
    num_state = env.num_states
    num_action = len(env.action_space)

    q_network = QNetwork(num_state, num_action)
    q_network.to(device)

    train(env, q_network)

    policy_matrix = test(env, q_network)

    # Render the environment
    print(f"Policy Matrix: \n{policy_matrix}")
    env.add_policy(policy_matrix)
    env.render(animation_interval=10)


if __name__ == "__main__":
    main()
