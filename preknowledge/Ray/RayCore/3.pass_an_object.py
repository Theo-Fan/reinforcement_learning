import ray
import numpy as np

@ray.remote
def sum_matrix(matrix: np.ndarray) -> np.ndarray: # 传入 matrix 为 np.ndarray 类型，同样返回为 np.ndarray 类型
    return np.sum(matrix)

print(ray.get(sum_matrix.remote(np.ones((100, 100)))))

matrix_ref = ray.put(np.ones((100, 100)))

print(f"Type of matrix_ref: {type(matrix_ref)}")
print(f"Value of matrix_ref: {matrix_ref}")

print(ray.get(sum_matrix.remote(matrix_ref))) # 以对象引用作为参数调用任务。