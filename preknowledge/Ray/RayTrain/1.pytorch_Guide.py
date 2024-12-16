"""
ScalingConfig():
    trainer_resource: dict, 为 worker 分配资源
    num_workers: int, 训练中使用的工作节点数量
    use_gpu: bool, 是否使用GPU
    resources_per_worker: dict, 指定每个工作节点需要的 CPU 和 GPU 数量
"""
# scaling_config = ScalingConfig(num_workers=1, use_gpu=False)
# scaling_config = ScalingConfig(resources_per_worker={"CPU": 4, "GPU": 2})
# scaling_config = ScalingConfig()


########################### Example ####################################
import torch
import ray
from ray import train
from ray.train import Checkpoint, ScalingConfig
from ray.train.torch import TorchTrainer

# Set this to True to use GPU.
# If False, do CPU training instead of GPU training.
use_gpu = False

# Step 1: Create a Ray Dataset from in-memory Python lists.
# You can also create a Ray Dataset from many other sources and fileformats.
# 通过列表创建一个 Ray Dataset。内容包含包含 200 个字典的数据集，每个元素包含两个键值对："x"：输入数据。"y"：目标标签（2 * x）。
train_dataset = ray.data.from_items(
    [
        {"x": [x], "y": [2 * x]} for x in range(200)
    ]
)
print("Step 1: train_dataset", train_dataset)


# Step 2: Preprocess your Ray Dataset.
# 将函数 increment 应用于数据集的每个批次，将 y 中的值 + 1
def increment(batch):
    batch["y"] = batch["y"] + 1
    return batch


train_dataset = train_dataset.map_batches(increment)
print("Step 2: train_dataset", train_dataset)


# 定义训练函数
def train_func():
    batch_size = 16

    # Step 4: Access the dataset shard for the training worker via `get_dataset_shard`.
    train_data_shard = train.get_dataset_shard("train")
    # `iter_torch_batches` returns an iterable object that yield tensor batches.
    # Ray Data automatically moves the Tensor batches to GPU if you enable GPU training.
    train_dataloader = train_data_shard.iter_torch_batches(
        batch_size=batch_size, dtypes=torch.float32
    )

    for epoch_idx in range(1):
        for batch in train_dataloader:
            inputs, labels = batch["x"], batch["y"]
            assert type(inputs) == torch.Tensor
            assert type(labels) == torch.Tensor
            assert inputs.shape[0] == batch_size
            assert labels.shape[0] == batch_size
            # Only check one batch for demo purposes.
            # Replace the above with your actual model training code.
            break


# Step 3: Create a TorchTrainer. Specify the number of training workers and pass in your Ray Dataset.
# The Ray Dataset is automatically split across all training workers.
trainer = TorchTrainer(
    train_func,
    datasets={
        "train": train_dataset
    },
    scaling_config=ScalingConfig(
        num_workers=2,
        use_gpu=use_gpu
    )
)

result = trainer.fit()
