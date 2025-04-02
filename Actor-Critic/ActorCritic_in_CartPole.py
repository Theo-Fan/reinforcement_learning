import torch
from torch import log
from torch.distributions import Categorical  # 用于离散概率分布的采样
from torch.optim import Adam
from torch.nn import Sequential, Linear, ReLU, Softmax, Module
import gymnasium as gym
from torch import tensor

# 定义策略网络 (Policy Network)，用于输出动作的概率分布
class PolicyNet(Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        # 定义一个包含全连接层和激活函数的神经网络
        self.model = Sequential(
            Linear(state_dim, hidden_dim),  # 输入层
            ReLU(),  # 激活函数
            Linear(hidden_dim, action_dim),  # 输出层
            Softmax(dim=0)  # 将输出转化为概率分布
        )

    def forward(self, x):
        # 前向传播，返回动作的概率
        return self.model(x)

# 定义价值网络 (Value Network)，用于评估状态的价值
class ValueNet(Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        # 定义一个简单的两层神经网络
        self.model = Sequential(
            Linear(state_dim, hidden_dim),  # 输入层
            ReLU(),  # 激活函数
            Linear(hidden_dim, 1)  # 输出层，返回单个状态价值
        )

    def forward(self, x):
        # 前向传播，返回状态的价值
        return self.model(x)

# 定义 Actor-Critic 算法
class ActorCritic:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr, gamma, device):
        # 初始化策略网络和价值网络
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)

        # 初始化优化器
        self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=critic_lr)

        self.gamma = gamma  # 折扣因子
        self.device = device  # 计算设备 (CPU/GPU)
        self.count = 0  # 用于记录更新次数

    # 根据状态选择动作
    def take_action(self, state):
        probs = self.actor(tensor(state).to(self.device))  # 获取动作概率
        categorical = Categorical(probs)  # 构造离散分布
        return categorical.sample().item()  # 根据概率分布采样动作

    # 更新网络参数
    def update(self, transitions):
        # 重置梯度
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

        # 遍历所有状态转移元组 (state, action, next_state, reward, done)
        for state, action, next_state, reward, done in transitions:
            state = state.to(self.device)
            td_target = reward.to(self.device) + self.gamma * self.critic(next_state.to(self.device)) * (1 - done)
            # 计算TD目标
            critic_loss = pow(td_target.detach() - self.critic(state), 2)  # 计算价值网络的损失

            delta = td_target - self.critic(state)  # 计算优势函数

            actor_loss = -log(self.actor(state)[action]) * delta.detach()  # 计算策略网络的损失

            # 反向传播计算梯度
            actor_loss.backward()
            critic_loss.backward()

        # 更新价值网络和策略网络的参数
        self.critic_optimizer.step()
        self.actor_optimizer.step()

if __name__ == "__main__":
    critic_lr = 5e-3  # 价值网络学习率
    actor_lr = 2e-3  # 策略网络学习率
    num_episodes = 1000  # 训练轮数
    hidden_dim = 128  # 隐藏层神经元数量
    gamma = 0.98  # 折扣因子

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]  
    action_dim = env.action_space.n  

    # 初始化Actor-Critic智能体
    agent = ActorCritic(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, gamma, device)

    # 开始训练
    for i in range(num_episodes):
        transitions = []  # 存储每轮的状态转移信息
        episode_return = 0  # 记录每轮的总回报
        state = env.reset()[0]  # 重置环境，获取初始状态

        # episode update
        # while True:
        #     action = agent.take_action(state)  # 智能体选择动作
        #     next_state, reward, terminated, truncated, _ = env.step(action)  # 执行动作，获取下一状态和奖励
        #     episode_return += reward  # 累积回报
        #     transitions.append(
        #         (tensor(state), tensor(action), tensor(next_state), tensor(reward), terminated or truncated)
        #     )  # 保存状态转移元组
        #
        #     if terminated or truncated:  # 如果结束，则退出循环
        #         break
        #     state = next_state  # 更新状态
        # agent.update(transitions)  # 更新智能体

        # 单步更新
        while True:
            action = agent.take_action(state)  # 智能体选择动作
            next_state, reward, terminated, truncated, _ = env.step(action)  # 执行动作
            episode_return += reward  # 累积回报

            # 转换为张量并移动到设备上
            state_tensor = tensor(state).to(agent.device)
            next_state_tensor = tensor(next_state).to(agent.device)
            reward_tensor = tensor(reward).to(agent.device)

            # 计算 TD 目标和 TD 误差
            td_target = reward_tensor + agent.gamma * agent.critic(next_state_tensor) * (
                        1 - int(terminated or truncated))

            delta = td_target - agent.critic(state_tensor)

            # 更新 Critic 网络（梯度下降）
            agent.critic_optimizer.zero_grad()
            critic_loss = delta.pow(2)  # 均方误差损失
            critic_loss.backward()
            agent.critic_optimizer.step()

            # 更新 Actor 网络（梯度上升）
            agent.actor_optimizer.zero_grad()
            log_prob = log(agent.actor(state_tensor)[action])
            actor_loss = -log_prob * delta.detach()  # 取负号实现梯度上升
            actor_loss.backward()
            agent.actor_optimizer.step()

            if terminated or truncated:
                break  # 回合结束

            state = next_state  # 更新状态

        if (i + 1) % 100 == 0:
            print(f"Episode: {i + 1}, Total Return: {episode_return}.")

    env.close()  # 关闭环境


    # 在训练完成后，使用人类可见的方式展示智能体
    env = gym.make("CartPole-v1", render_mode="human")
    for i in range(10):  # 运行10轮
        state = env.reset()[0]
        while True:
            env.render()  # 渲染环境
            action = agent.take_action(state)  # 智能体选择动作
            next_state, reward, terminated, truncated, _ = env.step(action)  # 执行动作
            if terminated or truncated:  # 如果结束，则退出循环
                break
            state = next_state  # 更新状态
    env.close()