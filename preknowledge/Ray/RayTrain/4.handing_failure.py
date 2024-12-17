from ray.train import RunConfig, FailureConfig

"""
failure_config = FailureConfig(
    failure_mode="retry",  # 在失败时重试
    max_retries=3,         # 最大重试次数为 3
    retry_interval_ms=1000, # 每次重试的间隔时间为 1000 毫秒
    backoff_factor=2.0      # 重试间隔时间指数级增长
)
"""

# run_config = RunConfig(failure_config=FailureConfig(max_failures=2)) # 失败后重试次数
# # No limit on the number of retries.
# run_config = RunConfig(failure_config=FailureConfig(max_failures=-1))


import os
import tempfile
from typing import Dict, Optional

import torch

import ray
from ray import train
from ray.train import Checkpoint
from ray.train.torch import TorchTrainer


def get_datasets() -> Dict[str, ray.data.Dataset]:
    return {"train": ray.data.from_items([{"x": i, "y": 2 * i} for i in range(10)])}


def train_loop_per_worker(config: dict):
    from torchvision.models import resnet18

    model = resnet18()

    # Checkpoint loading
    checkpoint: Optional[Checkpoint] = train.get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            model_state_dict = torch.load(os.path.join(checkpoint_dir, "model.pt"))
            model.load_state_dict(model_state_dict)

    model = train.torch.prepare_model(model)

    train_ds = train.get_dataset_shard("train")

    for epoch in range(5):
        # Do some training...

        # Checkpoint saving
        with tempfile.TemporaryDirectory() as tmpdir:
            torch.save(model.module.state_dict(), os.path.join(tmpdir, "model.pt"))
            train.report({"epoch": epoch}, checkpoint=Checkpoint.from_directory(tmpdir))


experiment_path = os.path.abspath('./checkpoints/dl_trainer_restore')

# 如果有checkpoint，则自动恢复
if TorchTrainer.can_restore(experiment_path):  # 如果之前的训练中断且状态已经保存，代码将从保存的训练状态继续进行，避免重复训练。
    trainer = TorchTrainer.restore(experiment_path, datasets=get_datasets())
    # result = trainer.fit()
else:
    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        datasets=get_datasets(),
        scaling_config=train.ScalingConfig(num_workers=2),
        run_config=train.RunConfig(
            name="dl_trainer_restore",
            storage_path=os.path.abspath('./checkpoints'),
            failure_config=FailureConfig(max_failures=2)
        ),
    )

result = trainer.fit()
