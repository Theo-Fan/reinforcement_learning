import time
import random
from re import findall

import gymnasium as gym
import numpy as np
import collections  # Python内置数据结构模块，包含 deque（双端队列）等
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt

from config_DQN import *

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)  # 创建一个固定大小的队列来存储经验，FIFO 方式

    def add(self, state, action, reward, next_state, done):  # 将一组经验 (state, action, reward, next_state, done) 添加到缓冲区
        self.buffer.append((state, action, reward, next_state, done))  # 将经验元组追加到队列中

    def sample(self, batch_size):  # 随机从缓冲区采样一个批次数据
        transitions = random.sample(self.buffer, batch_size)  # 随机采样 batch_size 个经验
        # 分别解包经验中的状态、动作、奖励、下一个状态和结束标志，并转换为 NumPy 数组
        state, action, reward, next_state, done = zip(*transitions)
        b_s, b_a, b_r, b_ns, b_d = np.array(state), action, reward, np.array(next_state), done
        # 返回一个包含状态、动作、奖励、下一个状态和结束标志的字典
        transition_dict = {
            'states': b_s,
            'actions': b_a,
            'next_states': b_ns,
            'rewards': b_r,
            'dones': b_d
        }
        return transition_dict  # 返回字典，供训练时使用

    def size(self):  # 返回当前缓冲区中存储的经验数量
        return len(self.buffer)


class Qnet(torch.nn.Module):  # 定义 Q 网络类，继承自 PyTorch 的 Module 基类
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):  # 定义前向传播函数
        return self.model(x)


class VAnet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(VAnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc_A = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_V = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # print("Input shape:", x.shape)  # 添加这一行调试
        A = self.fc_A(F.relu(self.fc1(x)))
        V = self.fc_V(F.relu(self.fc1(x)))
        Q = V + A - A.mean()
        return Q


class DQN:
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma, epsilon, target_update, device,
                 dqn_type):
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

        # 初始化 Q 网络和目标 Q 网络，分别用于决策和目标计算
        if dqn_type == 'DuelingDQN':  # DuelingDQN 采用不同网络架构
            self.q_net = VAnet(state_dim, hidden_dim, action_dim).to(device)
            self.target_q_net = VAnet(state_dim, hidden_dim, action_dim).to(device)
        else:  # 普通 Q 网络
            self.q_net = Qnet(state_dim, hidden_dim, action_dim).to(device)  # Q 网络，用于决策
            self.target_q_net = Qnet(state_dim, hidden_dim, action_dim).to(device)  # 目标 Q 网络，用于计算目标值

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略中的 epsilon 值，用于控制随机探索的概率
        self.target_update = target_update  # 目标网络更新频率，多少次更新后同步目标网络
        self.count = 0  # 用于记录更新次数

        self.dqn_type = dqn_type
        self.device = device  # 计算设备（CPU 或 GPU）

    def take_action(self, state, is_epsilon=True):  # 采用 epsilon-贪婪策略选择动作
        if is_epsilon and np.random.random() < self.epsilon:  # 以 epsilon 概率随机选择动作
            action = np.random.randint(self.action_dim)  # 随机选择动作
        else:
            # 将状态转换为张量并移动到计算设备上
            state = torch.tensor(np.array(state), dtype=torch.float32).to(self.device)
            # 根据 Q 网络计算出每个动作的 Q 值，选择 Q 值最大的动作
            action = self.q_net(state).argmax().item()
        return action  # 返回选择的动作

    def update(self, transition_dict):  # 更新 Q 网络参数
        # 将经验字典中的状态、动作、奖励、下一个状态和结束标志转换为张量
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        # 从 Q 网络中提取与执行的动作对应的 Q 值
        q_values = self.q_net(states).gather(1, actions)

        if self.dqn_type == 'DoubleDQN':
            next_actions = self.q_net(next_states).argmax(dim=1, keepdim=True)
            max_next_q_values = self.target_q_net(next_states).gather(1, next_actions)
        else:
            # 从目标 Q 网络中计算下一个状态的最大 Q 值
            max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
        # 计算 TD 目标：奖励 + 折扣因子 * 下一个状态的最大 Q 值 * (1 - done)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)
        # 计算均方误差损失函数
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets).to(device))
        self.optimizer.zero_grad()  # 梯度清零
        dqn_loss.backward()  # 反向传播，计算梯度
        self.optimizer.step()  # 更新 Q 网络参数

        # 每隔 target_update 步更新目标 Q 网络的参数
        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())  # 将 Q 网络的参数复制到目标 Q 网络
        self.count += 1  # 计数器增加

    def save_model(self, model_path):  # 保存模型
        torch.save(self.q_net.state_dict(), model_path)  # 将 Q 网络的参数保存到指定路径

    def load_model(self, model_path):  # 加载模型
        state_dict = torch.load(model_path)  # 从文件加载 Q 网络的参数
        self.q_net.load_state_dict(state_dict, False)  # 加载参数到 Q 网络中


def cal_reward(state, _reward, done, step_num, max_step_num):
    reward = _reward  # 初始奖励为环境返回的即时奖励
    if step_num >= max_step_num - 1:  # 如果达到最大步数，则增加额外奖励
        reward += reward * 5  # 增加 5 倍奖励，鼓励完成任务
    elif done:  # 如果提前结束（失败或成功）
        reward = -1  # 设置为负奖励，表示失败
    return reward  # 返回计算后的奖励


def plot_figure(results):
    keys = ['reward', 'success']  # 需要绘制的指标：奖励和成功率
    for k in keys:  # 对每个指标进行绘图
        iteration_list = list(range(len(results['ave_' + k])))  # 获取迭代次数列表
        plt.plot(iteration_list, results['ave_' + k], color='b', label='ave_' + k)  # 平均值曲线
        plt.xlabel('Iteration')  # x 轴标签
        plt.ylabel(k)  # y 轴标签
        plt.title('DQN on {}'.format(game_name, k))  # 图标题
        plt.show()  # 显示图像

        figure_path = train_figure_path.replace('.png', '_{}.png'.format(k))  # 图像保存路径
        plt.savefig(figure_path)  # 保存图像到文件


def train(dqn_type):
    # ==== 初始化环境 ====
    env = gym.make(game_name)  # 创建 Gym 环境

    # ==== 设置随机种子 ====
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    # ==== 初始化经验回放缓冲区 ====
    replay_buffer = ReplayBuffer(buffer_size)

    # ==== 初始化 DQN 代理 ====
    state_dim = env.observation_space.shape[0]  # 获取状态维度
    action_dim = int(findall(r"\d+\.?\d*", str(env.action_space))[0])  # 获取动作维度

    print(state_dim, action_dim)
    agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon, target_update, device, dqn_type)

    results = {}  # 存储训练结果
    for k in ['reward', 'success']:  # 初始化结果列表
        results['ave_' + k] = []

    # ==== 训练循环 ====
    for i in range(iteration_num):  # 每次迭代
        with tqdm(total=episode_num, desc='Iteration %d' % i) as pbar:  # 显示进度条
            rewards, successes = [], []  # 存储每个 episode 的奖励和成功率
            for i_episode in range(episode_num):  # 每次 episode
                state, _ = env.reset()  # 重置环境，获取初始状态

                for step_num in range(max_step_num):  # 在一个 episode 中执行动作
                    action = agent.take_action(state)  # 选择动作
                    # 执行动作并获得下一个状态、奖励和是否结束
                    next_state, _reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated  # 判断是否结束

                    # 计算奖励
                    reward = cal_reward(next_state, _reward, done, step_num, max_step_num)

                    # 将经验存储到经验回放缓冲区
                    replay_buffer.add(state, action, reward, next_state, done)

                    # 当经验回放缓冲区大小超过最小阈值后，开始训练
                    if replay_buffer.size() > minimal_size:
                        transition_dict = replay_buffer.sample(batch_size)  # 从缓冲区采样经验
                        agent.update(transition_dict)  # 使用采样的经验更新 Q 网络

                    state = next_state  # 更新状态
                    if done: break  # 如果 episode 结束，退出循环

                # 记录奖励和成功情况
                success = 0
                if step_num >= 499:  # 如果完成所有步数，认为是成功
                    success = 1
                successes.append(success)
                rewards.append(reward)

                # 计算平均、最大和最小奖励和成功率
                ave_reward = np.mean(rewards)
                ave_success = np.mean(successes)

                # 每 10 个 episode 显示一次进度信息
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode': '%d' % (episode_num * i + i_episode + 1),
                        '\treward': '%.3f' % ave_reward,
                        '\tsuccess': '%.3f' % ave_success,
                    })
                pbar.update(1)  # 更新进度条

            # 保存每次迭代的结果
            results['ave_reward'].append(ave_reward)
            results['ave_success'].append(ave_success)

    agent.save_model(model_path)  # 保存训练后的模型
    # plot_figure(results)  # 绘制训练结果图像


def test(dqn_type):
    env_play = gym.make(game_name, render_mode='human')  # 创建带有 UI 的环境
    state, _ = env_play.reset()  # 重置环境，获取初始状态

    # ==== 初始化 DQN 代理 ====
    state_dim = env_play.observation_space.shape[0]  # 获取状态维度
    action_dim = int(findall(r"\d+\.?\d*", str(env_play.action_space))[0])  # 获取动作维度

    print(state_dim, action_dim)

    agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon, target_update, device, dqn_type)
    agent.load_model(model_path)  # 加载训练好的模型

    # time.sleep(10)

    step_num = -1  # 步数计数器
    done = False  # 结束标志

    # ==== 测试循环 ====
    while not done:
        step_num += 1  # 更新步数
        action = agent.take_action(state, is_epsilon=False)  # 不使用 epsilon 贪婪策略，选择最优动作
        # 执行动作并获得下一个状态、奖励和是否结束
        next_state, reward, terminated, truncated, info = env_play.step(int(action))
        done = terminated or truncated  # 判断是否结束
        state = next_state  # 更新状态

        # 打印测试的步数、动作、奖励、状态和是否结束
        print(' Test ---- step_num, action, reward, obs, done: ', step_num, action, reward, state, done)
        time.sleep(0.05)


if __name__ == '__main__':
    dqn_type = "DQN" # DoubleDQN or DuelingDQN or DQN

    train(dqn_type)
    test(dqn_type)
