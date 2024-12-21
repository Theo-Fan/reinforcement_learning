from ray import tune, train

"""
Ray Tune Process:
    SearchSpaces -> Trainables        ->
                    Search Algotrithm ->  -----> Trials -> Analyses
                    Schedulers        ->

Explain:
    Search Spaces (搜索空间):
        搜索空间定义了需要调优的超参数及其取值范围。例如: learning rate 可以在 [0.001, 0.1] 之间搜索
    
    Trainables (可训练目标):
        需要训练的模型或目标任务。
    
    Search Algorithms (搜索算法):
        搜索算法决定了搜索超参数的策略。如：贝叶斯优化、遗传算法等
    
    Schedulers (调度器):
        调度器决定了何时停止搜索。如：早停、最大迭代次数等
    
    Trials (实验执行)
        运行实验。每次实验 (Trial) 是一个特定的超参数组合和模型训练的具体运行。
        Ray Tune 会管理这些实验并并行化执行，最大限度利用计算资源。
    
    Analyses (结果分析)

"""


# 1. define an objective function
def objective(config):
    score = config["a"] ** 2 + config["b"]
    return {"score": score}


# 2. define a search space
search_space = {
    "a": tune.grid_search([0.001, 0.01, 0.1, 1.0]),
    "b": tune.choice([1, 2, 3])
}

# 3. start a Tune run and print the best result
tuner = tune.Tuner(objective, param_space=search_space)

results = tuner.fit()
print(results.get_best_result(metric='score', mode="min").config)
