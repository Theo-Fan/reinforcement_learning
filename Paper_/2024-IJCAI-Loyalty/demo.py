import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt
import pandas as pd

N = 1000  # agent 数量
z = 30  # 平均邻居数量
# W = 4  # 时间尺度比例
H = 200000  # 最大迭代次数
alpha = 0.05  # Q-learning学习率
tau = 5  # Boltzmann探索参数
beta = 0.005  # 费米分布温度参数
T, R, P, S = 2, 1, 0, -1


def initialize_network(N, z):
    G = nx.random_regular_graph(z, N)
    strategies = np.random.choice(["C", "D"], N, p=[0.3, 0.7])
    Q = {i: {"C": np.random.uniform(0.4, 0.6), "D": np.random.uniform(0.4, 0.6)} for i in range(N)}
    return G, strategies, Q


def q_learning_update(Q, agent_id, action, reward, alpha):
    Q[agent_id][action] = (1 - alpha) * Q[agent_id][action] + alpha * reward


def calculate_payoff(strategy_i, strategy_j):
    if strategy_i == "C" and strategy_j == "C":
        return R
    elif strategy_i == "C" and strategy_j == "D":
        return S
    elif strategy_i == "D" and strategy_j == "C":
        return T
    else:
        return P


def boltzmann_policy(Q, tau):
    exp_Q = np.exp(tau * np.array([Q["C"], Q["D"]]))
    probabilities = exp_Q / np.sum(exp_Q)
    return probabilities


# 更新网络连接
def update_network(G, strategies, Q, tau):
    i = random.choice(list(G.nodes))
    neighbors = list(G.neighbors(i))
    if not neighbors:
        return

    j = random.choice(neighbors)
    action_i = np.random.choice([0, 1], p=boltzmann_policy(Q[i], tau))
    action_j = np.random.choice([0, 1], p=boltzmann_policy(Q[j], tau))

    if action_i == 1 or action_j == 1:  # 断开连接
        G.remove_edge(i, j)
        candidate_nodes = list(set(G.nodes) - set(G.neighbors(i)) - {i})
        if candidate_nodes:
            cooperative_candidates = [n for n in candidate_nodes if strategies[n] == "C"]
            if cooperative_candidates:
                new_neighbor = random.choice(cooperative_candidates)
            else:
                new_neighbor = random.choice(candidate_nodes)
            c = random.choice([0, 1])
            if c == 0:
                G.add_edge(i, new_neighbor)
                reward_i = calculate_payoff(strategies[i], strategies[new_neighbor])
                reward_j = 0
            else:
                G.add_edge(j, new_neighbor)
                reward_i = 0
                reward_j = calculate_payoff(strategies[j], strategies[new_neighbor])
        else:
            reward_i = reward_j = 0
    else:  # 保持连接
        reward_i = calculate_payoff(strategies[i], strategies[j])
        reward_j = calculate_payoff(strategies[j], strategies[i])

    q_learning_update(Q, i, strategies[i], reward_i, alpha)
    q_learning_update(Q, j, strategies[j], reward_j, alpha)


def update_strategy(G, strategies, beta):
    """策略模仿"""
    i = random.choice(list(G.nodes))
    neighbors = list(G.neighbors(i))
    if not neighbors:
        return

    j = random.choice(neighbors)

    payoff_i = sum(calculate_payoff(strategies[i], strategies[k]) for k in G.neighbors(i))
    payoff_j = sum(calculate_payoff(strategies[j], strategies[k]) for k in G.neighbors(j))
    imitation_prob = 1 / (1 + np.exp(-beta * (payoff_j - payoff_i)))
    if np.random.rand() < imitation_prob:
        strategies[i] = strategies[j]


def run_simulation(N, z, W, H, tau, beta):
    G, strategies, Q = initialize_network(N, z)
    cooperation_levels = []

    cnt_update_network = 0
    cnt_update_strategy = 0

    cnt_cooperate = 0
    cnt_defact = 0
    for i in range(1000):
        print(strategies[i], end=" ")
        if strategies[i] == "C":
            cnt_cooperate += 1
        else:
            cnt_defact += 1

    print(f"\ncooperate: {cnt_cooperate}, defact: {cnt_defact}")

    for t in range(H):
        x = np.random.rand()
        if x >= 1 / (1 + W):  # 网络更新
            update_network(G, strategies, Q, tau)
            cnt_update_network += 1
        else:  # 策略更新
            update_strategy(G, strategies, beta)
            cnt_update_strategy += 1

        # 记录合作比例
        cooperation_levels.append(np.mean([1 if s == "C" else 0 for s in strategies]))
        # if t % 1000 == 0:
        #     print(f"Iteration {t}, Cooperation Level: {np.mean([1 if s == 'C' else 0 for s in strategies])}")

    cnt_cooperate = 0
    cnt_defact = 0
    for i in range(1000):
        print(strategies[i], end=" ")
        if strategies[i] == "C":
            cnt_cooperate += 1
        else:
            cnt_defact += 1

    print(f"\n Q: {Q}")

    print(
        f"\ncooperate: {cnt_cooperate}, defact: {cnt_defact}, update_network: {cnt_update_network}, update_strategy: {cnt_update_strategy}")

    return cooperation_levels


def smooth_data(data, window_size=50):
    """平滑数据函数，使用移动平均"""
    smoothed_data = np.convolve(data, np.ones(window_size) / window_size, mode='valid')
    return smoothed_data


# 可视化结果
def plot_results(results, pwd, window_size=50):
    plt.figure(figsize=(10, 6))
    for W, levels in results.items():
        smoothed_levels = smooth_data(levels, window_size)
        plt.plot(range(len(smoothed_levels)), smoothed_levels, label=f"W={W}")
    plt.xlabel("Iterations")
    plt.ylabel("Percentage of Cooperation")
    plt.title("Cooperation Levels for Different Time Scale Ratios (W)")
    plt.legend()
    plt.savefig(pwd)
    plt.show()


def save_results_to_csv(results, filename):
    data = {f"W={W}": levels for W, levels in results.items()}
    max_len = max(len(levels) for levels in results.values())

    # 填充短数组为相同长度
    for key in data.keys():
        if len(data[key]) < max_len:
            data[key].extend([None] * (max_len - len(data[key])))

    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")


if __name__ == '__main__':

    W_values = [1, 2, 4, 5, 6, 7, 8, 12]
    for i in range(1, 10):
        results = {}
        for W in W_values:
            print(f"Running simulation for W={W}...")
            cooperation_levels = run_simulation(N, z, W, H, tau, beta)
            results[W] = cooperation_levels

        save_results_to_csv(results, filename=f"cooperation_levels{i}.csv")

        plot_results(results, f"img/result{i}.png", window_size=500)
