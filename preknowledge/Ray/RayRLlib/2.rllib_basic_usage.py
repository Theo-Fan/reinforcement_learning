import ray
from ray import air, tune, train
from ray.rllib.algorithms.ppo import PPOConfig

# tune
ray.init()

config = (
    PPOConfig()
    .api_stack(
        enable_rl_module_and_learner=True,
        enable_env_runner_and_connector_v2=True,
    )
    .environment("CartPole-v1")
    .training(
        lr=tune.grid_search([0.01, 0.001, 0.0001]),
    )
)

tuner = tune.Tuner(
    "PPO",
    param_space=config,
    run_config=train.RunConfig(
        stop={"env_runners/episode_return_mean": 150.0},
    ),
)

results = tuner.fit()

# Get the best result based on a particular metric.
best_result = results.get_best_result(
    metric="env_runners/episode_return_mean", mode="max"
)

# Get the best checkpoint corresponding to the best result.
best_checkpoint = best_result.checkpoint

# load checkpoint
# from ray.rllib.algorithms.algorithm import Algorithm
# algo = Algorithm.from_checkpoint(checkpoint_path)