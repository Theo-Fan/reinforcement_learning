import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

n_samples = 100
x_samples = np.random.uniform(-10, 10, size=n_samples)
y_samples = np.random.uniform(-10, 10, size=n_samples)
samples = np.column_stack((x_samples, y_samples))

# 计算真实均值
true_mean = samples.mean(axis=0)

def sgd(samples, alpha, max_iter=100):
    w = np.array([20.0, 20.0])
    path = [w]
    for i in range(max_iter):
        sample = samples[np.random.randint(len(samples))]  # 随机选择一个样本
        w = w - alpha * (w - sample)
        path.append(w)
        print(w)
    return np.array(path)

def mbgd(samples, alpha, batch_size, max_iter=100):
    w = np.array([20.0, 20.0])
    path = [w]
    for i in range(max_iter):
        indices = np.random.choice(len(samples), batch_size, replace=False)  # 随机选取一个小批量
        batch = samples[indices]
        gradient = (w - batch).mean(axis=0)  # 计算小批量梯度（求导完的形式）
        w = w - alpha * gradient
        path.append(w)
    return np.array(path)

alpha = 0.1
max_iter = 50
sgd_path = sgd(samples, alpha, max_iter)
mbgd_path_5 = mbgd(samples, alpha, batch_size=5, max_iter=max_iter)
mbgd_path_50 = mbgd(samples, alpha, batch_size=50, max_iter=max_iter)

# 绘制图像
plt.figure(figsize=(10, 8))
plt.scatter(x_samples, y_samples, color="black", alpha=0.5, zorder=3, label="Samples")
plt.scatter(*true_mean, color="red", marker="x", s=150, zorder=3, label="Mean")

# 绘制路径
plt.plot(sgd_path[:, 0], sgd_path[:, 1], '-o', label="SGD (m=1)", color="orange", zorder=1, alpha=0.4)
plt.plot(mbgd_path_5[:, 0], mbgd_path_5[:, 1], '-^', label="MBGD (m=5)", zorder=1, color="green", alpha=0.4)
plt.plot(mbgd_path_50[:, 0], mbgd_path_50[:, 1], '-*', label="MBGD (m=50)", zorder=1, color="blue", alpha=0.4)

plt.xlabel("x", fontsize=12)
plt.ylabel("y", fontsize=12)
plt.legend(fontsize=12)
plt.title("SGD and MBGD Convergence Paths", fontsize=14)
plt.grid(alpha=0.3)
plt.show()
