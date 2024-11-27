import numpy as np
import pandas as pd


class RL(object):
    def __init__(self, action_space, alpha=0.01, gamma=0.9, epilson=0.9):
        self.actions = action_space
        self.lr = alpha
        self.gamma = gamma
        self.epsilon = epilson

        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def check_state_exist(self, state):
        # This function is similar with  'build_qtable' but this one need less memory
        if state not in self.q_table.index:
            # current state is not exist, create new row(set state as idx, q_table.columns as columns)
            new_row = pd.DataFrame([[0] * len(self.actions)], index=[state], columns=self.q_table.columns)
            self.q_table = pd.concat([self.q_table, new_row])

    def choose_action(self, observation):
        self.check_state_exist(observation)
        if np.random.rand() < self.epsilon:
            state_action = self.q_table.loc[observation, :]
            action = state_action.idxmax()
            # """ action = cur_state_action.idxmax()
            # Could be replaced with:
            #     action = np.random.choice(cur_state_action[cur_state_action == np.max(cur_state_action)].index)
            # Explain:
            #     if cur_state_action = [0.5, 1.2, 0.9, 1.2], np.max(cur_state_action) will return  1.2
            #     cur_state_action == np.max(cur_state_action) will return [False, True, False, True]
            #     cur_state_action[cur_state_action == np.max(cur_state_action)] will return [1.2, 1.2]
            #     take the index will return [1, 3], and the equation is: np.random.choice([1, 3])
            #     The result is 1 or 3
            # """
        else:
            action = np.random.choice(self.actions)
        return action

    def learn(self, *args):
        pass


class QLearning(RL):
    def __init__(self, actions, alpha=0.01, gamma=0.9, epilson=0.9):
        super(QLearning, self).__init__(actions, alpha, gamma, epilson)

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        # updata function: Q(s, a) = Q(s, a) + alpha * [r + gamma * max(Q(s_, :)) - Q(s, a)]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()
        else:
            q_target = r
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)


class Sarsa(RL):
    def __init__(self, actions, alpha=0.01, gamma=0.9, epilson=0.9):
        super(Sarsa, self).__init__(actions, alpha, gamma, epilson)

    def learn(self, s, a, r, s_, a_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        # updata function: Q(s, a) = Q(s, a) + alpha * [r + gamma * Q(s_, a_) - Q(s, a)]
        if s_ != "terminal":
            q_target = r + self.gamma * self.q_table.loc[s_, a_]
        else:
            q_target = r
        TD_error = q_target - q_predict

        self.q_table.loc[s, a] += self.lr * TD_error


""" Sarsa Lambda Process
初始化 Q(s, a) 和 E(s, a) 为 0
初始化当前状态 s 和动作 a

循环直到终止条件：
    执行动作 a，观察奖励 r 和新的状态 s'
    选择新的动作 a'（使用ε-greedy策略）
    计算 TD 误差 δ = r + γ * Q(s', a') - Q(s, a)
    更新迹函数 E(s, a) = E(s, a) + 1 or other method
    对于所有状态-动作对 (s, a)：
        Q(s, a) = Q(s, a) + α * δ * E(s, a)
        E(s, a) = γ * λ * E(s, a)
    更新 s = s', a = a'    
"""

class SarsaLambda(RL):
    def __init__(self, actions, alpha=0.01, gamma=0.9, epilson=0.9, lambda_=0.9):
        super(SarsaLambda, self).__init__(actions, alpha, gamma, epilson)

        self.alpha = alpha
        self.lambda_ = lambda_
        self.eligibility_trace = self.q_table.copy() # eligibility trace is equal to q-table

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            new_row = pd.DataFrame([[0] * len(self.actions)], index=[state], columns=self.q_table.columns)
            self.q_table = pd.concat([self.q_table, new_row])
            self.eligibility_trace = pd.concat([self.eligibility_trace, new_row]) # add to eligibility trace as well


    def learn(self, s, a, r, s_, a_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]

        if s != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, a_]
        else:
            q_target = r

        err = q_target - q_predict # TD Error

        # Method 1:
        # self.eligibility_trace.loc[s, a] += 1

        # Method 2: (important)
        self.eligibility_trace.loc[s, :] *= 0
        self.eligibility_trace.loc[s, a] = 1

        # Using eligibility trace update q-table: Q(s, a) = Q(s, a) + alpha * err * E(s, a)
        self.q_table += self.alpha * err * self.eligibility_trace

        # decay eligibility trace after update
        self.eligibility_trace *= self.gamma * self.lambda_


