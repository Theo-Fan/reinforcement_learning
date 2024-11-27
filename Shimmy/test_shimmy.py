from shimmy.meltingpot_compatibility import ParallelEnv, MeltingPotCompatibilityV0

env = MeltingPotCompatibilityV0(substrate_name='clean_up', render_mode='human')

desired_num_agents = 2

# 设置自定义 agent 数量
env._num_players = desired_num_agents
env.possible_agents = [env.PLAYER_STR_FORMAT.format(index=i) for i in range(desired_num_agents)]
env.agents = env.possible_agents[:]

print(env.num_agents)
print(env.max_num_agents)
print(env.possible_agents)


