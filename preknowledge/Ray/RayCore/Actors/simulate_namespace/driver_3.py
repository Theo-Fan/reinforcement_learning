import ray

# 初始化 Ray 并指定命名空间为 "colors"
ray.init(address="auto", namespace="colors")

# 直接获取 Actor "orange"
actor = ray.get_actor("orange")
print("Actor 'orange' retrieved in namespace 'colors'")
