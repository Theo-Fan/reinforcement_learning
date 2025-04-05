import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# distribution p0 and p1
p0 = {+1: 0.5, -1: 0.5}
p1 = {+1: 0.8, -1: 0.2}


# Importance weight
def importance_weight(x):
    return p0[x] / p1[x]


n = 200
samples = np.random.choice([+1, -1], size=n, p=[p1[+1], p1[-1]])

sample_vals = []
avg_vals = []
importance_vals = []

running_sum = 0
running_weighted_sum = 0

for i in range(n):
    x = samples[i]
    sample_vals.append(x)

    # non-importance sampling
    running_sum += x
    avg_vals.append(running_sum / (i + 1))

    # importance sampling
    w = importance_weight(x)
    running_weighted_sum += x * w
    weighted_avg = running_weighted_sum / (i + 1)
    importance_vals.append(weighted_avg)

plt.figure(figsize=(8, 5))

plt.plot(
    range(n),
    sample_vals,
    marker='o',
    linestyle='None',
    markeredgecolor='#C15226',
    markerfacecolor='none',
    label="samples"
)

# average
plt.plot(avg_vals, color='#2C64B0', linestyle=':', linewidth=2, label="average")

# importance sampling
plt.plot(importance_vals, color='#B8CE90', linestyle='-', linewidth=1.5, label="importance sampling")

# 辅助线 
plt.axhline(y=1, color='black', linewidth=0.3, alpha=0.2)
plt.axhline(y=0, color='black', linewidth=0.3, alpha=0.2)
plt.axhline(y=-1, color='black', linewidth=0.3, alpha=0.2)

plt.xlabel("Sample index")
plt.ylim(-2, 2)
plt.legend()
plt.tight_layout()
plt.show()
