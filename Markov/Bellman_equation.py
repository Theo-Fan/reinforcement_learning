import numpy as np

np.random.seed(0)

# 定义状态转移概率矩阵P
P = [
    [0.9, 0.1, 0.0, 0.0, 0.0, 0.0],
    [0.5, 0.0, 0.5, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.6, 0.0, 0.4],
    [0.0, 0.0, 0.0, 0.0, 0.3, 0.7],
    [0.0, 0.2, 0.3, 0.5, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
]
P = np.array(P)

rewards = [-1, -2, -2, 10, 1, 0]
gamma = 0.5

def compute_return(start_index, chain, gamma):
    tot_val = 0
    # 下行代码，反转操作的目的是确保每个状态的回报是从未来累积到当前的，而不是从当前直接计算所有的未来回报，因此需要从最后一步倒推至起始状态。
    for i in range(len(chain))[::-1]:
        tot_val = tot_val * gamma + rewards[chain[i] - 1]
        print(chain[i], rewards[chain[i] - 1], tot_val)
    """
    上述代码等价于:
        for i in range(len(chain)):
            tot_val += rewards[chain[i] - 1] * gamma**i
            print(tot_val)
    """
    return tot_val

"""
    V = r + gamma * \sum(p * v)
    
    => V = R + gamma * P * V
    => (I - gamma * P) * V = R
     
"""
def compute(P, rewards, gamman, states_num):
    # 利用贝尔曼方程的矩阵形式计算解析解,states_num是MRP的状态数
    rewards = np.array(rewards).reshape((-1, 1)) # 将 rewards 写成列向量的形式
    """
        np.eye(states_num, states_num)：生成一个大小为 states_num × states_num 的单位矩阵  I 。
        np.eye(states_num, states_num) - gamma * P  ===> (I - gamma * P)
        np.linalg.inv(A)  ==>  对矩阵 A 进行求逆运算
        np.dot(A, B)  ===>  将矩阵 A 和矩阵 B 进行点乘
    """
    val = np.dot(np.linalg.inv(np.eye(states_num, states_num) - gamma  * P), rewards)

    return val


chain = [1, 2, 3, 6]
start_idx = 0
G = compute_return(start_idx, chain, gamma)
print(f"本序列的 rewards 为：{G}")


########################################################

""" 计算该马尔可夫奖励过程中所有状态的价值 """
V = compute(P, rewards, gamma, 6)
print("MRP中每个状态价值 V(i) 分别为\n", V)

# 验证 V(s_4) = r(s_4) + gamma * p(s_5 | s_4) * v(s_5) + p(s_6 | s_4) * v(s_6)

v_4 = 0
for i in range(len(P[0])):
    if P[4 - 1][i] == 0:
        continue
    v_4 += P[4 - 1][i] * V[i]

v_4 = v_4 * gamma + rewards[4 - 1]
print("验证 V(s_4) 的值为：", v_4)