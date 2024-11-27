import random
import math


def monte_carlo_expectation(func, distribution, num_samples):
    total_sum = 0
    for _ in range(num_samples):
        x = distribution()  # 从指定的概率分布中抽样
        total_sum += func(x)
    expectation_estimate = total_sum / num_samples
    return expectation_estimate


# f(X) = X^2 + sin(X)
func = lambda x: x ** 2 + math.sin(x)

# 从标准正态分布 N(0, 1) 中抽样
distribution = lambda: random.gauss(0, 1)

# 样本数量
num_samples = 1000000

# 计算期望值
expectation = monte_carlo_expectation(func, distribution, num_samples)
print(f"使用蒙特卡洛方法计算的期望值: {expectation}")
