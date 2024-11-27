"""
    马尔可夫奖励过程 MRP
"""

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


chain = [1, 2, 3, 6]
start_idx = 0
G = compute_return(start_idx, chain, gamma)
print(f"本序列的 rewards 为：{G}")