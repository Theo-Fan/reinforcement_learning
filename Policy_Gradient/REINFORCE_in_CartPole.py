import torch
from matplotlib import pyplot as plt
from torch import log, tensor
from torch.distributions import Categorical
from torch.optim import Adam
from torch.nn import Module, Sequential, Linear, ReLU, Softmax
import gymnasium as gym
from numpy import mean


# 定义策略网络
class PolicyNetwork(Module):
    def __init__(self, state_dim, hidden_dim, action_card):
        super(PolicyNetwork, self).__init__()
        self.model = Sequential(
            Linear(state_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, action_card),
            Softmax(dim=0)  # 将输出转换为概率分布
        )

    def forward(self, x):
        return self.model(x)


class REINFORCE:
    def __init__(self, state_dim, hidden_dim, action_card, lr, gamma, device):
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_card = action_card
        self.gamma = gamma
        self.device = device

        self.policy_net = PolicyNetwork(self.state_dim, self.hidden_dim, self.action_card).to(self.device)

        self.optimizer = Adam(self.policy_net.parameters(), lr=lr)

    # 根据当前状态选择动作
    def take_action(self, state):
        probs = self.policy_net(tensor(state))  # 计算动作概率
        categorical = Categorical(probs)  # 创建一个离散概率分布
        return categorical.sample().item()  # 从分布中采样一个动作

    # 更新策略网络
    def update(self, transitions):
        G = 0
        self.optimizer.zero_grad()  # 梯度清零

        for state, action, next_state, reward, done in reversed(transitions):  # 因为最后一个元素为最近时间的，因此需要对轨迹进行反转
            G = self.gamma * G + reward  # 马尔可夫奖励过程
            # 下述 Loss Function 为 REINFORCE 核心部分，由于要最大化 policy_net，反向传播只能最小化 loss, 因此加上 负号
            loss = -log(self.policy_net(state)[action]) * G
            loss.backward() # 反向传播

        # 更新策略网络参数
        self.optimizer.step()


if __name__ == "__main__":
    lr = 1e-3  # 学习率
    num_episodes = 1000
    hidden_dim = 128
    gamma = 0.98  # 折扣因子
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    # 初始化环境
    env = gym.make("CartPole-v1")

    # 获取状态和动作空间的维度
    state_dim = env.observation_space.shape[0]
    action_card = env.action_space.n

    # 创建REINFORCE算法的agent
    agent = REINFORCE(
        state_dim,
        hidden_dim,
        action_card,
        lr,
        gamma,
        device
    )

    # 开始训练
    for i in range(num_episodes):
        transitions = []  # 存储一个episode中的状态、动作、回报序列
        episode_return = 0  # 记录当前episode的总回报
        state = env.reset()[0]  # 初始化环境并获取初始状态

        while True:
            action = agent.take_action(state)  # 根据策略选择动作
            next_state, reward, terminated, truncated, _ = env.step(action)  # 执行动作
            episode_return += reward  # 累积回报

            # 存储当前transition
            transitions.append((tensor(state), tensor(action), next_state, tensor(reward), terminated or truncated))

            if terminated or truncated:  # 判断episode是否结束
                break

            state = next_state  # 更新状态

        agent.update(transitions)  # 更新策略

        print(f"episodes: {i}, episode_returns: {episode_return}.")

    env.close()

    # 绘制回报曲线
    # plt.figure(figsize=(12, 8), dpi=80)
    # plt.plot(returns)
    # plt.xlabel("episode")
    # plt.ylabel("return")
    # plt.show()

    # 测试
    env = gym.make("CartPole-v1", render_mode="human")

    for i in range(10):
        state = env.reset()[0]  # 重置环境
        while True:
            env.render()
            action = agent.take_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            if terminated or truncated:
                break

            state = next_state

    env.close()
