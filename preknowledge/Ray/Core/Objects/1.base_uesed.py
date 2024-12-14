import ray
import time

# Get the value of one object ref.
obj_ref = ray.put(1)
assert ray.get(obj_ref) == 1

# Get the values of multiple object refs in parallel.
assert ray.get([ray.put(i) for i in range(3)]) == [0, 1, 2]

# You can also set a timeout to return early from a ``get``
# that's blocking for too long.
from ray.exceptions import GetTimeoutError


# ``GetTimeoutError`` is a subclass of ``TimeoutError``.

@ray.remote
def long_running_function():
    time.sleep(8)


obj_ref = long_running_function.remote()
try:
    ray.get(obj_ref, timeout=4)
except GetTimeoutError:  # You can capture the standard "TimeoutError" instead
    print("`get` timed out.")

print("\n#######################################################################################")


@ray.remote
def echo(a: int, b: int, c: int):
    """This function prints its input values to stdout."""
    print(a, b, c)


# Passing the literal values (1, 2, 3) to `echo`.
echo.remote(1, 2, 3)

# Put the values (1, 2, 3) into Ray's object store.
a, b, c = ray.put(1), ray.put(2), ray.put(3)
print(type(a), type(b), type(c))

print(f"type of ray.get(a) : {type(ray.get(a))}")
# echo 需要传入三个 int 类型，而 a, b, c 类型为 <class 'ray._raylet.ObjectRef'>
# 相当于在调用函数时，自动调用了 ray.get
echo.remote(a, b, c)

print("\n#######################################################################################")


@ray.remote
def echo_and_get(x_list):  # List[ObjectRef]
    """This function prints its input values to stdout."""
    print("args:", x_list)
    print("values:", ray.get(x_list))  # ray.get 同样接收 ObjectRef List，并返回对应的值


# Put the values (1, 2, 3) into Ray's object store.
a, b, c = ray.put(1), ray.put(2), ray.put(3)

# Passing an object as a nested argument to `echo_and_get`.
# Ray does not de-reference nested args, so `echo_and_get` sees the references.
echo_and_get.remote([a, b, c])
# args: [ObjectRef(00ffffffffffffffffffffffffffffffffffffff0100000008e1f505), ObjectRef(00ffffffffffffffffffffffffffffffffffffff0100000009e1f505), ObjectRef(00ffffffffffffffffffffffffffffffffffffff010000000ae1f505)]
# values: [1, 2, 3]


print("\n#######################################################################################")


@ray.remote
class Actor:
    def __init__(self, arg):
        print(arg)  # (Actor pid=7277) [ObjectRef(00ffffffffffffffffffffffffffffffffffffff010000000be1f505)]

    def method(self, arg):
        pass


obj = ray.put(2)

# Examples of passing objects to actor constructors.
actor_handle = Actor.remote(obj)
actor_handle = Actor.remote([obj])

# Examples of passing objects to actor method calls.
actor_handle.method.remote(obj)
actor_handle.method.remote([obj])

print("\n#######################################################################################")
a, b, c = ray.put(1), ray.put(2), ray.put(3)


@ray.remote
def print_via_capture():
    """This function prints the values of (a, b, c) to stdout."""
    print(ray.get([a, b, c]))


# Passing object references via closure-capture. Inside the `print_via_capture` function,
# the global object refs (a, b, c) can be retrieved and printed.
print_via_capture.remote() # result: [1, 2, 3]


