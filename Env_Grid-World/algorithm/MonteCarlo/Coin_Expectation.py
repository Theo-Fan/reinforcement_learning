import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子，以确保实验可重复
np.random.seed(12)

# 样本数量
num_samples = 500

# 模拟抛硬币实验：生成+1或-1，概率各为0.5
samples = np.random.choice([-1, 1], size=num_samples)

# 计算逐步平均值
cumulative_average = np.cumsum(samples) / np.arange(1, num_samples + 1)
overall_average = np.mean(samples)

plt.figure(figsize=(8, 5))
plt.plot(samples, 'o', markerfacecolor='none', label='samples', alpha=0.7)
plt.plot(cumulative_average, '-', linewidth=2.5, label='average')
# 绘制整体平均值线
plt.axhline(y=overall_average, color='grey', linestyle='-', linewidth=0.5, alpha=0.5)

plt.xlim(0, 200)
plt.ylim(-2, 2)

plt.xlabel('Sample index', fontsize=14)
# plt.ylabel('Value', fontsize=14)
plt.xticks(np.arange(0, 201, 50), fontsize=12)  # 横坐标0到200，间隔为20
plt.yticks(np.arange(-2, 2.1, 1), fontsize=12)  # 纵坐标-2到2，间隔为0.5
plt.legend(fontsize=12)

plt.tight_layout()
plt.show()
