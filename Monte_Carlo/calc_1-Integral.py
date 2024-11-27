import random
import math


# f(x) = \sin(x) \cdot e^{x} + \frac{1}{1 + x^2}

def monte_carlo_1d(func, a, b, n):
    cnt = 0
    for _ in range(n):
        x = random.uniform(a, b)
        cnt += func(x)
    integral_estimate = (b - a) * (cnt / n) # (b - a) * average value
    return integral_estimate

def f(x):
    return math.sin(x) * math.exp(x) + 1 / (1 + x**2)


if __name__ == '__main__':
    a, b = 0, 1
    n = 1000000
    res_1d = monte_carlo_1d(f, a, b, n)
    print(f'1-dimension integral: {res_1d:.4f}')

