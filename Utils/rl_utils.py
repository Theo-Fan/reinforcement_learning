from tqdm import tqdm  # 进度条库，用于在循环中提供可视化反馈
import numpy as np  # 数值计算库，用于数组和矩阵操作
import torch  # PyTorch深度学习库
import collections  # 提供专门的容器数据类型，例如双端队列deque
import random  # 用于随机采样

# 实现离线策略（off-policy）强化学习代理的经验回放缓存区类
class ReplayBuffer:
    def __init__(self, capacity):
        # 使用双端队列（deque）初始化缓冲区，最大容量为capacity，超出容量时自动移除旧数据
        self.buffer = collections.deque(maxlen=capacity)

    # 将新体验加入缓冲区
    def add(self, state, action, reward, next_state, done):
        # 将状态、动作、奖励、下一个状态以及是否结束的标志组成元组添加到缓冲区中
        self.buffer.append((state, action, reward, next_state, done))

    # 从缓冲区中随机采样一批数据
    def sample(self, batch_size):
        # 随机采样batch_size个转换
        transitions = random.sample(self.buffer, batch_size)
        # 将采样的转换分别解包为状态、动作、奖励、下一个状态和完成标志，并转换为数组形式
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    # 获取缓冲区中当前存储的经验数量
    def size(self):
        return len(self.buffer)


# 计算数组的移动平均值
def moving_average(a, window_size):
    # 计算累计和
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    # 计算窗口的中间部分的平均值
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    # 计算窗口的开头部分的平均值
    r = np.arange(1, window_size - 1, 2)
    begin = np.cumsum(a[:window_size - 1])[::2] / r
    # 计算窗口的结尾部分的平均值
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    # 合并并返回完整的移动平均值数组
    return np.concatenate((begin, middle, end))


# 训练在线策略（on-policy）强化学习代理
def train_on_policy_agent(env, agent, num_episodes):
    return_list = []  # 存储每一集的回报
    for i in range(10):  # 分成10个迭代阶段
        # 使用tqdm显示每个阶段的进度条
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0  # 初始化本集的回报
                # 初始化转换字典，存储每一集中的状态、动作、奖励等
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                state = env.reset()  # 重置环境，获取初始状态
                done = False  # 初始化done标志为False
                while not done:
                    # 代理根据当前状态选择动作
                    action = agent.take_action(state)
                    # 环境根据动作返回下一个状态、奖励、是否结束以及其他信息
                    next_state, reward, done, _ = env.step(action)
                    # 将转换存储到字典中
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state  # 更新当前状态
                    episode_return += reward  # 累加本集的奖励
                return_list.append(episode_return)  # 记录本集回报
                agent.update(transition_dict)  # 更新代理
                # 每训练10集，更新进度条并显示最近10集的平均回报
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list


# 训练离线策略（off-policy）强化学习代理
def train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size):
    return_list = []  # 存储每一集的回报
    for i in range(10):  # 分成10个迭代阶段
        # 使用tqdm显示每个阶段的进度条
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0  # 初始化本集的回报
                state = env.reset()  # 重置环境，获取初始状态
                done = False  # 初始化done标志为False
                while not done:
                    # 代理根据当前状态选择动作
                    action = agent.take_action(state)
                    # 环境根据动作返回下一个状态、奖励、是否结束以及其他信息
                    next_state, reward, done, _ = env.step(action)
                    # 将转换存储到经验回放缓冲区中
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state  # 更新当前状态
                    episode_return += reward  # 累加本集的奖励
                    # 当缓冲区中的经验数目超过最小要求时，开始采样并更新代理
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r,
                                           'dones': b_d}
                        agent.update(transition_dict)  # 用采样的数据更新代理
                return_list.append(episode_return)  # 记录本集回报
                # 每训练10集，更新进度条并显示最近10集的平均回报
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list


# 计算优势函数
def compute_advantage(gamma, lmbda, td_delta):
    # 将TD误差转换为numpy数组
    td_delta = td_delta.detach().numpy()
    advantage_list = []  # 存储优势值
    advantage = 0.0  # 初始化优势为0
    # 逆序计算每个时间步的优势
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta  # 递归计算优势
        advantage_list.append(advantage)  # 记录优势值
    advantage_list.reverse()  # 反转顺序，使其与时间步匹配
    return torch.tensor(advantage_list, dtype=torch.float)  # 返回优势的张量形式