import random
import numpy as np
import networkx as nx
import pandas as pd
from matplotlib import pyplot as plt


def calculate_payoff(strategy_i, strategy_j, R=1, S=-1, T=2, P=0):
    if strategy_i == "C" and strategy_j == "C":
        return R
    elif strategy_i == "C" and strategy_j == "D":
        return S
    elif strategy_i == "D" and strategy_j == "C":
        return T
    else:
        return P


def initialize_network(N, z):
    G = nx.random_regular_graph(z, N)
    strategies = np.random.choice(["C", "D"], N, p=[0.3, 0.7])
    Q = {i: {
        "state_C": {
            "action_C": np.zeros(1),
            "action_D": np.zeros(1),
        },
        "state_D": {
            "action_C": np.zeros(1),
            "action_D": np.zeros(1),
        },
    } for i in range(N)}
    return G, strategies, Q


def boltzmann_policy(Q, state, tau):
    exp_Q = np.exp(tau * np.array([Q[state]["action_C"], Q[state]["action_D"]]))
    probabilities = exp_Q / np.sum(exp_Q)
    return probabilities.ravel()


def update_Q(state, action, reward, Q, alpha):
    Q[state][action] = (1 - alpha) * Q[state][action] + alpha * reward


def update_network(G, strategies, Q, tau, alpha):
    i = random.choice(list(G.nodes))
    neighbors = list(G.neighbors(i))
    if not neighbors:
        return

    j = random.choice(neighbors)

    state_i = f"state_{strategies[j]}"
    state_j = f"state_{strategies[i]}"

    action_i = np.random.choice(["C", "D"], p=boltzmann_policy(Q[i], state_i, tau))  # C: 保持连接, D: 断开连接
    action_j = np.random.choice(["C", "D"], p=boltzmann_policy(Q[j], state_j, tau))

    if action_i == "D" or action_j == "D":  # 断开连接
        G.remove_edge(i, j)
        candidate_nodes = list(set(G.nodes) - set(G.neighbors(i)) - {i})
        new_neighbor = random.choice(candidate_nodes)

        choose_one = np.random.choice([1, 2])  # 从 i, j 中选取一个与新的节点进行连接
        if choose_one == 1:  # i
            G.add_edge(i, new_neighbor)
            reward_i = calculate_payoff(strategies[i], strategies[new_neighbor])
            reward_j = 0
        else:
            G.add_edge(j, new_neighbor)
            reward_i = 0
            reward_j = calculate_payoff(strategies[j], strategies[new_neighbor])

    else:  # 保持连接
        reward_i = calculate_payoff(strategies[i], strategies[j])
        reward_j = calculate_payoff(strategies[j], strategies[i])

    update_Q(state_i, f"action_{action_i}", reward_i, Q[i], alpha)
    update_Q(state_j, f"action_{action_j}", reward_j, Q[j], alpha)

    strategies[i] = action_i
    strategies[j] = action_j


def update_strategy(G, strategies, beta):
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


def run_simulation(N, z, W, H, tau, beta, alpha):
    G, strategies, Q = initialize_network(N, z)
    cooperation_levels = []

    for t in range(H):
        x = np.random.rand()
        if x >= 1 / (1 + W):  # 网络更新
            update_network(G, strategies, Q, tau, alpha)
        else:  # 策略更新
            update_strategy(G, strategies, beta)

        # 记录合作比例
        cooperation_levels.append(np.mean([1 if s == "C" else 0 for s in strategies]))

    print("Q: ", Q)

    final_cooperation = np.mean([1 if s == "C" else 0 for s in strategies])

    print(f"Final Cooperation Level: {final_cooperation:.2f}")
    return cooperation_levels


def smooth_data(data, window_size=50):
    smoothed_data = np.convolve(data, np.ones(window_size) / window_size, mode='valid')
    return smoothed_data


# 绘制结果
def plot_results(results, filename, window_size=50):
    plt.figure(figsize=(10, 6))
    for W, levels in results.items():
        smoothed_levels = smooth_data(levels, window_size)
        plt.plot(range(len(smoothed_levels)), smoothed_levels, label=f"W={W}")
    plt.xlabel("Iterations")
    plt.ylabel("Percentage of Cooperation")
    plt.title("Cooperation Levels for Different Time Scale Ratios (W)")
    plt.legend()
    plt.savefig(filename)
    plt.show()


# 保存结果
def save_results_to_csv(results, filename):
    data = {f"W={W}": levels for W, levels in results.items()}
    max_len = max(len(levels) for levels in results.values())

    for key in data.keys():
        if len(data[key]) < max_len:
            data[key].extend([None] * (max_len - len(data[key])))

    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")

if __name__ == '__main__':
    N = 1000  # Agent 数量
    z = 30  # 平均连接数
    H = 200000  # 迭代次数

    alpha = 0.05
    tau = 5
    beta = 0.005

    W_values = [1, 2, 4, 5, 6, 7, 8, 12]
    for i in range(1, 11):
        results = {}
        for W in W_values:
            print(f"Running simulation for W={W}...")
            cooperation_levels = run_simulation(N, z, W, H, tau, beta, alpha)
            results[W] = cooperation_levels

        save_results_to_csv(results, f"data/cooperation_results{i}.csv")
        plot_results(results, f"img/cooperation_plot{i}.png", window_size=1000)

        print()

