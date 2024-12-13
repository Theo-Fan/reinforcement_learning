import ray

# 初始化 Ray 并指定命名空间为 "fruit"
ray.init(address="auto", namespace="fruit")

try:
    # 尝试在当前命名空间 "fruit" 中获取 Actor "orange"
    ray.get_actor("orange")
except ValueError:
    print("Actor 'orange' not found in namespace 'fruit'")

# 显式指定命名空间为 "colors" 来获取 Actor
actor = ray.get_actor("orange", namespace="colors")
print(ray.get(actor.get_value.remote())) # 调用driver_1.Actor类中的get_value方法
print("Actor 'orange' successfully retrieved from namespace 'colors'")