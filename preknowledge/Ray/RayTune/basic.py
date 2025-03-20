from ray import train, tune


def objective(x, a, b):
    return a * x ** 0.5 + b


def trainable(config):
    for x in range(20):
        score = objective(x, config['a'], config['b'])
        tune.report({"score": score})



config = {
    "a": tune.uniform(0, 1),
    'b': tune.uniform(0, 1),
}

tuner = tune.Tuner(
    trainable,
    tune_config=tune.TuneConfig(
        metric='score', # 需要在 trainable 中的tune.report 进行报告。
        mode='max',
    ),
    run_config=tune.RunConfig(stop={"training_iteration": 20}), # 只会显示的调用一次trainable
    param_space=config,
)

results = tuner.fit()


best_result = results.get_best_result()  # Get best result object
best_config = best_result.config  # Get best trial's hyperparameters
best_logdir = best_result.path  # Get best trial's result directory
best_checkpoint = best_result.checkpoint  # Get best trial's best checkpoint
best_metrics = best_result.metrics  # Get best trial's last results
best_result_df = best_result.metrics_dataframe  # Get best result as pandas dataframe
