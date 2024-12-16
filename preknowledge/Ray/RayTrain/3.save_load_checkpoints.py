import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

import ray
import ray.train.torch
from ray import train
from ray.train import ScalingConfig, RunConfig
from ray.train.torch import TorchTrainer

# 初始化 Ray 环境
ray.init()

# 指定保存检查点的目录
SAVE_DIR = f"file://{os.path.abspath('./checkpoints')}"


def train_func(config):
    n = 100
    # 创建数据集
    X = torch.Tensor(np.random.normal(0, 1, size=(n, 4)))
    Y = torch.Tensor(np.random.uniform(0, 1, size=(n, 1)))

    # 定义模型并包装为分布式模型
    model = ray.train.torch.prepare_model(nn.Linear(4, 1))
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=3e-4)

    # 确保保存检查点的目录存在
    if train.get_context().get_world_rank() == 0:
        os.makedirs('./checkpoints', exist_ok=True)

    # ====== Resume training state from the checkpoint. ======
    start_epoch = 0
    checkpoint = train.get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            model_state_dict = torch.load(
                os.path.join('./checkpoints', "model.pt"),
                # map_location=...,  # Load onto a different device if needed.
            )
            model.module.load_state_dict(model_state_dict)
            # optimizer.load_state_dict(
            #     torch.load(os.path.join(checkpoint_dir, "optimizer.pt"))
            # )
            # start_epoch = (
            #     torch.load(os.path.join(checkpoint_dir, "extra_state.pt"))["epoch"] + 1
            # )

    # ========================================================
    for epoch in range(config["num_epochs"]):
        # 前向传播
        y_pred = model(X)
        loss = criterion(y_pred, Y)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 保存检查点 (仅主进程执行)
        if train.get_context().get_world_rank() == 0 and epoch % config.get("checkpoint_freq", 1) == 0:
            torch.save(
                model.module.state_dict(),
                os.path.join('./checkpoints', "model.pt")
            )

        # 上报训练指标
        train.report({"loss": loss.item()})


# 配置 TorchTrainer
trainer = TorchTrainer(
    train_func,
    train_loop_config={"num_epochs": 5, "checkpoint_freq": 1},
    scaling_config=ScalingConfig(num_workers=2),
    run_config=train.RunConfig(storage_path=SAVE_DIR),
)

# 启动训练
result = trainer.fit()
print("训练完成，最终指标:", result.metrics)

# Seed a training run with a checkpoint using `resume_from_checkpoint`
trainer = TorchTrainer(
    train_func,
    train_loop_config={"num_epochs": 5},
    scaling_config=ScalingConfig(num_workers=2),
    resume_from_checkpoint=result.checkpoint,
)

print("resume_from_checkpoint finished")