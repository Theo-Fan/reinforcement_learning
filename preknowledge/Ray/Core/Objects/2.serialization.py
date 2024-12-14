import ray
from ray import cloudpickle

FILE = "external_store.pickle"

ray.init()

my_dict = {"hello": "world"}
obj_ref = ray.put(my_dict)

# 使用 cloudpickle.dump 将 ObjectRef 序列化到文件中（FILE）。用于后续重新加载。
with open(FILE, "wb+") as f:
    cloudpickle.dump(obj_ref, f)

# ObjectRef remains pinned in memory because it was serialized with ray.cloudpickle.
del obj_ref

# 使用 cloudpickle.load 从文件中加载序列化的 ObjectRef，并赋值给 new_obj_ref。
# 反序列化后，new_obj_ref 指向与原始 obj_ref 相同的对象。
with open(FILE, "rb") as f:
    new_obj_ref = cloudpickle.load(f)

# The deserialized ObjectRef works as expected.
assert ray.get(new_obj_ref) == my_dict

# Explicitly free the object.
ray._private.internal_api.free(new_obj_ref)