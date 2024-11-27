import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

import copy

np.random.seed(1)
torch.manual_seed(1)

# define the network architecture
class Net(nn.Module):
	def __init__(self, n_feature, n_hidden, n_output):
		super(Net, self).__init__()
		self.el = nn.Linear(n_feature, n_hidden)
		self.q = nn.Linear(n_hidden, n_output)

	def forward(self, x):
		x = self.el(x)
		x = F.relu(x)
		x = self.q(x)
		return x


class DeepQNetwork():
	def __init__(self, n_actions, n_features, n_hidden=20, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9,
				replace_target_iter=200, memory_size=500, batch_size=32, e_greedy_increment=None,
				):
		self.n_actions = n_actions
		self.n_features = n_features
		self.n_hidden = n_hidden
		self.lr = learning_rate
		self.gamma = reward_decay
		self.epsilon_max = e_greedy
		self.replace_target_iter = replace_target_iter
		self.memory_size = memory_size
		self.batch_size = batch_size
		self.epsilon_increment = e_greedy_increment
		self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

		# total learning step
		self.learn_step_counter = 0

		# initialize zero memory [s, a, r, s_]
		self.memory = np.zeros((self.memory_size, n_features*2+2))

		self.loss_func = nn.MSELoss()
		self.cost_his = []

		self._build_net()
		

	def _build_net(self):
		self.q_eval = Net(self.n_features, self.n_hidden, self.n_actions)
		self.q_target = Net(self.n_features, self.n_hidden, self.n_actions)
		self.optimizer = torch.optim.RMSprop(self.q_eval.parameters(), lr=self.lr)

	# create experience replay pool
	def store_transition(self, s, a, r, s_):
		if not hasattr(self, 'memory_counter'):
			self.memory_counter = 0
		transition = np.hstack((s, [a, r], s_))
		# replace the old memory with new memory
		index = self.memory_counter % self.memory_size
		self.memory[index, :] = transition 
		self.memory_counter += 1

	def choose_action(self, observation):
		observation = torch.Tensor(observation[np.newaxis, :])
		if np.random.uniform() < self.epsilon:
			actions_value = self.q_eval(observation)

			action = np.argmax(actions_value.data.numpy())
		else:
			action = np.random.randint(0, self.n_actions)
		return action

	def learn(self):
		# 检查是否需要替换目标网络的参数
		# 每隔一定的步骤（self.replace_target_iter），将评估网络（q_eval）的参数赋值给目标网络（q_target）
		if self.learn_step_counter % self.replace_target_iter == 0:
			# 目标网络更新为评估网络的参数
			self.q_target.load_state_dict(self.q_eval.state_dict())
			print("\ntarget params replaced\n")

		# 从记忆池中随机抽取一个批次的记忆样本进行训练
		if self.memory_counter > self.memory_size:
			# 如果记忆池中的记忆数目超过了最大容量（memory_size），从中随机抽取 batch_size 个记忆
			sample_index = np.random.choice(self.memory_size, size=self.batch_size)
		else:
			# 如果记忆池中的记忆数目未超过最大容量，从已有的记忆中随机抽取 batch_size 个记忆
			sample_index = np.random.choice(self.memory_counter, size=self.batch_size)

		# 从记忆池中获取样本
		batch_memory = self.memory[sample_index, :]

		# 计算 q_next 和 q_eval，用于 DQN 的目标更新
		# q_next 是目标网络对下一状态 s_(t+1) 的评估，用于计算 y 值（目标值）
		# q_eval 是评估网络对当前状态 s_t 的评估，用于计算当前的 Q 值
		q_next, q_eval = (
			self.q_target(torch.Tensor(batch_memory[:, -self.n_features:])),
			self.q_eval(torch.Tensor(batch_memory[:, :self.n_features]))
		)

		# 生成 q_target，该变量表示目标 Q 值，用于计算损失
		# 这里需要对 q_eval 进行拷贝，因为我们要对 q_eval 进行操作，确保未选择的 Q 值不会被修改
		q_target = torch.Tensor(q_eval.data.numpy().copy())

		# 生成批次的索引，用于后续根据行动索引更新 q_target
		batch_index = np.arange(self.batch_size, dtype=np.int32)

		# 获取当前批次中，每个样本的动作索引
		# eval_act_index 表示在评估网络中，针对每个样本所采取的动作索引
		eval_act_index = batch_memory[:, self.n_features].astype(int)

		# 获取当前批次中的奖励值
		reward = torch.Tensor(batch_memory[:, self.n_features + 1])

		# 根据 DQN 更新公式更新 q_target 中的目标值
		# 对应于选择的动作，q_target 的值为：奖励值 + 折扣因子 * 目标网络中对下一状态的最大 Q 值
		q_target[batch_index, eval_act_index] = reward + self.gamma * torch.max(q_next, 1)[0]

		# 计算损失：将评估网络的 Q 值与目标 Q 值进行比较，计算均方误差
		loss = self.loss_func(q_eval, q_target)

		# 反向传播并更新网络参数
		self.optimizer.zero_grad()  # 清除之前的梯度
		loss.backward()  # 反向传播，计算梯度
		self.optimizer.step()  # 更新网络参数

		# 增加 epsilon，逐渐提高探索的概率
		# epsilon 用于控制 epsilon-greedy 策略中的探索与利用的平衡
		self.cost_his.append(loss)  # 记录损失历史
		# epsilon 逐渐递增，直到达到 epsilon_max
		self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max

		# 增加学习步数计数器
		self.learn_step_counter += 1

	def plot_cost(self):
		plt.plot(np.arange(len(self.cost_his)), self.cost_his)
		plt.ylabel('Cost')
		plt.xlabel('training steps')
		plt.show()