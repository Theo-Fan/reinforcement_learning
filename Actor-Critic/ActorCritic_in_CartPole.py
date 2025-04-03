import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim

# 创建环境
env = gym.make('CartPole-v1')

# 超参数
lr = 1e-3
gamma = 0.99
hidden_size = 128

# 状态和动作维度
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n


# 定义Actor-Critic网络
class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            # nn.Linear(hidden_size, hidden_size),
            # nn.ReLU(),
        )
        self.actor = nn.Linear(hidden_size, action_dim)
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, state):
        x = self.shared(state)
        policy_logits = self.actor(x)
        value = self.critic(x)
        return policy_logits, value


# 初始化网络和优化器
model = ActorCritic()
optimizer = optim.Adam(model.parameters(), lr=lr)

# 训练循环
for episode in range(3000):
    state, _ = env.reset()
    log_probs = []
    values = []
    rewards = []
    done = False
    total_reward = 0

    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        logits, value = model(state_tensor)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()

        next_state, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated

        total_reward += reward
        log_probs.append(dist.log_prob(action))
        values.append(value)
        rewards.append(reward)
        state = next_state

    # 计算 returns 和 advantage
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns)
    values = torch.cat(values).squeeze()
    log_probs = torch.stack(log_probs)
    advantage = returns - values.detach()

    # 计算损失
    actor_loss = -(log_probs * advantage).mean()
    critic_loss = nn.functional.mse_loss(values, returns)
    loss = actor_loss + critic_loss

    # 更新网络
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if episode % 100 == 0:
        print(f"Episode {episode}, Total Reward: {total_reward}")

env.close()

test_env = gym.make("CartPole-v1", render_mode="human")  # 打开渲染窗口
state, _ = test_env.reset()
done = False
total_test_reward = 0

while not done:
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    with torch.no_grad():
        logits, _ = model(state_tensor)
        probs = torch.softmax(logits, dim=-1)
        action = torch.argmax(probs, dim=-1).item()

    state, reward, terminated, truncated, _ = test_env.step(action)
    done = terminated or truncated
    total_test_reward += reward

    # time.sleep(0.03)  # 控制速度，免得太快

print(f"Test Episode Reward: {total_test_reward}")
test_env.close()
