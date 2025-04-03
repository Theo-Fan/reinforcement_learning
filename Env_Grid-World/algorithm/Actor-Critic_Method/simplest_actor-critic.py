import sys
from pprint import pprint

sys.path.append("../..")
from env.grid_world import GridWorld

import torch
from torch import nn
from torch.optim import Adam
import torch.nn.functional as F
import numpy as np

lr = 1e-3
num_episodes = 1000
max_steps = 100
gamma = 0.95

# [(0, 1), (1, 0), (0, -1), (-1, 0), (0, 0)]  # 下， 右， 上， 左， 原地

ref = {
    0: "down",
    1: "right",
    2: "up",
    3: "left",
    4: "nothing"
}


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(ActorCritic, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(),
        )
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        x = self.model(state)
        policy_logits = self.actor(x)
        value = self.critic(x)
        return policy_logits, value


def change_state(state, num_states=25):
    state_vec = torch.zeros(num_states)
    state_vec[state] = 1
    return state_vec


def train(env, net):
    optimizer = Adam(net.parameters(), lr=lr)

    for episode in range(num_episodes):
        env.reset()

        tot_reward = 0
        done = False

        state_tensor = change_state(env.agent_state[1] * env.env_size[0] + env.agent_state[0])
        _, value = net(state_tensor)

        while not done:
            pos = env.agent_state[1] * env.env_size[0] + env.agent_state[0]
            state_tensor = change_state(pos)

            logits, value = net(state_tensor)

            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()

            log_prob = dist.log_prob(action)
            entropy = dist.entropy()

            next_state, reward, done, info = env.step(env.action_space[action.item()])
            tot_reward += reward

            next_state = env.agent_state[1] * env.env_size[0] + env.agent_state[0]

            next_value = 0 if done else net(change_state(next_state))[1]

            td_target = reward + gamma * next_value * (1 - int(done))
            td_error = td_target - value

            actor_loss = -log_prob * td_error.detach() - 0.01 * entropy
            critic_loss = td_error.pow(2)
            loss = actor_loss + critic_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            value = next_value

        if episode % 100 == 0:
            print(f"Episode {episode}/{num_episodes} completed. Total reward: {tot_reward}")


def test(env, net):
    num_state = env.num_states
    num_action = len(env.action_space)
    env.reset()
    with torch.no_grad():

        policy_matrix = torch.zeros((num_state, num_action))
        for i in range(num_state):
            state_vec = change_state(i)
            logits, _ = net(state_vec)
            best_action = torch.argmax(logits)
            policy_matrix[i][best_action] = 1.0

        for t in range(15):
            env.render(animation_interval=0.5)

            pos = env.agent_state[1] * env.env_size[0] + env.agent_state[0]
            state_tensor = change_state(pos)

            action_probs, _ = net(state_tensor)

            probs = torch.softmax(action_probs, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()

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

    net = ActorCritic(num_state, num_action)

    train(env, net)  # TODO

    # sys.exit()
    policy_matrix = test(env, net)  # TODO

    # Render the environment
    print(f"Policy Matrix: \n{policy_matrix}")
    env.add_policy(policy_matrix)
    env.render(animation_interval=2)


if __name__ == '__main__':
    main()
