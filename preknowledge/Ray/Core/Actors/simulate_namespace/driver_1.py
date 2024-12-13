import ray

@ray.remote
class Actor:
    def __init__(self):
        self.value = 0

    def get_value(self):
        return self.value

# 初始化 Ray 并指定命名空间为 "colors"
ray.init(address="auto", namespace="colors")

# 创建一个 "detached" 的 Actor，名称为 "orange"
Actor.options(name="orange", lifetime="detached").remote()

print("Actor 'orange' created in namespace 'colors'")
