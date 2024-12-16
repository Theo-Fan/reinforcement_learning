import ray

ray.init()


# 定义 Counter Actor
@ray.remote
class Counter:
    def __init__(self):
        self.i = 0

    def get(self):
        return self.i

    def incr(self, _, value):
        print(f"current is {_}")
        self.i += value


# 创建一个 Counter Actor
c = Counter.remote()

# 向 Actor 提交调用。这些调用 异步运行，但在远程参与者进程上按提交顺序运行。
for _ in range(10):
    c.incr.remote(_, 1)



print(ray.get(c.get.remote()))  # 读取 actor 最终的状态
