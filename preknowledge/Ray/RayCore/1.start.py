import ray

ray.init() # ray 初始化

@ray.remote
def square(x):
    return x * x

futures = [square.remote(i) for i in range(4)]

print(ray.get(futures))