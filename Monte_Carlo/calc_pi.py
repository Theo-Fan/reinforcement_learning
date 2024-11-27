import random


def monte_carlo_pi(n):
    cnt = 0

    for _ in range(n):
        x, y = random.uniform(-1, 1), random.uniform(-1, 1)
        if x**2 + y**2 <= 1:
            cnt += 1

    pi_estimate = 4 * cnt / n
    return pi_estimate


if __name__ == '__main__':
    n = 1000000
    pi_approxmation = monte_carlo_pi(n)
    print(f'Point number: {n}, Calc Pi approximation: {pi_approxmation}')