import random

# \int_0^1 \int_0^1 \int_0^1 (x^2 + y^2 + z^2) \, dx \, dy \, dz

def monte_carlo_3d(func, x_range, y_range, z_range, num_points):
    total_sum = 0
    for _ in range(num_points):
        x = random.uniform(*x_range)
        y = random.uniform(*y_range)
        z = random.uniform(*z_range)
        total_sum += func(x, y, z)
    volume = (x_range[1] - x_range[0]) * (y_range[1] - y_range[0]) * (z_range[1] - z_range[0])
    integral_estimate = volume * (total_sum / num_points)
    return integral_estimate


# 定义被积函数
func_3d = lambda x, y, z: x ** 2 + y ** 2 + z ** 2


x_range = (0, 1)
y_range = (0, 1)
z_range = (0, 1)
num_points = 1000000
result_3d = monte_carlo_3d(func_3d, x_range, y_range, z_range, num_points)
print(f"3-dimensional integral: {result_3d}")
