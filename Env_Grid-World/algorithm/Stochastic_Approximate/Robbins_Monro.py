import numpy as np
import matplotlib.pyplot as plt

# 定义函数 g(w) (可以使用神经网络进行给出)
def g(w):
    return np.tanh(w - 1)

w1 = 3  # 初始值
a_k = lambda k: 1 / k  # 步长公式

# Robbins-Monro 迭代过程
w_values = [w1]
w = w1
for k in range(1, 100):
    w = w - a_k(k) * g(w)
    w_values.append(w)

print(w)

w_range = np.linspace(0, 4, 500)
g_values = g(w_range)

plt.figure(figsize=(10, 6))
plt.plot(w_range, g_values, label="$g(w)$", color='red')
plt.axhline(0, color='black', linewidth=0.8, linestyle='-')

for i in range(len(w_values) - 1):
    # 垂直线：从 w_k 到 g(w_k)
    plt.plot([w_values[i], w_values[i]], [0, g(w_values[i])], 'k--', color='orange', alpha=0.75)
    # 水平线：从 g(w_k) 到 w_{k+1}
    plt.plot([w_values[i], w_values[i + 1]], [g(w_values[i]), 0], 'k:', color="lightblue", alpha=0.7)

for i, w in enumerate(w_values[:5], start=1):
    plt.text(w, -0.1, f"$w_{i}$", horizontalalignment='center')

plt.xlabel("$w$")
plt.ylabel("$g(w)$")
plt.title("Robbins-Monro Algorithm Iterative Visualization")
plt.show()