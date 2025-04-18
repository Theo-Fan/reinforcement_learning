import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


env = gym.make('CliffWalking-v0')
state_dim = env.observation_space.n  # 状态数量 48
action_dim = env.action_space.n      # 动作数量 4


lr = 1e-3
gamma = 0.99
hidden_size = 128
entropy_coef = 0.01
num_episodes = 2000


def one_hot(state, dim):
    vec = torch.zeros(dim)
    vec[state] = 1.0
    return vec.unsqueeze(0)  # shape: [1, state_dim]

class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden_size, action_dim)
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, state_onehot):
        x = self.shared(state_onehot)
        policy_logits = self.actor(x)
        value = self.critic(x)
        return policy_logits, value

# 初始化网络和优化器
model = ActorCritic()
optimizer = optim.RMSprop(model.parameters(), lr=lr)

# 训练
reward_history = []

for episode in range(num_episodes):
    state, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        state_tensor = one_hot(state, state_dim)
        logits, value = model(state_tensor)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        next_state, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated
        total_reward += reward

        next_value = 0.0
        if not done:
            next_state_tensor = one_hot(next_state, state_dim)
            _, next_value_tensor = model(next_state_tensor)
            next_value = next_value_tensor.item()

        td_target = reward + gamma * next_value
        td_error = td_target - value.item()

        actor_loss = -log_prob * td_error - entropy_coef * entropy
        critic_loss = F.mse_loss(value, torch.tensor([[td_target]]))
        loss = actor_loss + critic_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        state = next_state

    reward_history.append(total_reward)

    if episode % 100 == 0:
        avg_reward = sum(reward_history[-100:]) / 100
        print(f"Episode {episode}, Avg Reward (last 100): {avg_reward:.2f}, Total: {total_reward}")

env.close()

# 测试
test_env = gym.make("CliffWalking-v0", render_mode="human")
state, _ = test_env.reset()
done = False
total_test_reward = 0

while not done:
    state_tensor = one_hot(state, state_dim)
    with torch.no_grad():
        logits, _ = model(state_tensor)
        probs = torch.softmax(logits, dim=-1)
        action = torch.argmax(probs, dim=-1).item()

    state, reward, terminated, truncated, _ = test_env.step(action)
    done = terminated or truncated
    total_test_reward += reward

print(f"Test Episode Reward: {total_test_reward}")
test_env.close()