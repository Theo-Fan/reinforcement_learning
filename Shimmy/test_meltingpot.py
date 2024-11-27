""" Environment params
# keep-sorted start
    'allelopathic_harvest__open',
    'bach_or_stravinsky_in_the_matrix__arena',
    'bach_or_stravinsky_in_the_matrix__repeated',
    'boat_race__eight_races',
    'chemistry__three_metabolic_cycles',
    'chemistry__three_metabolic_cycles_with_plentiful_distractors',
    'chemistry__two_metabolic_cycles',
    'chemistry__two_metabolic_cycles_with_distractors',
    'chicken_in_the_matrix__arena',
    'chicken_in_the_matrix__repeated',
    'clean_up',
    'coins',
    'collaborative_cooking__asymmetric',
    'collaborative_cooking__circuit',
    'collaborative_cooking__cramped',
    'collaborative_cooking__crowded',
    'collaborative_cooking__figure_eight',
    'collaborative_cooking__forced',
    'collaborative_cooking__ring',
    'commons_harvest__closed',
    'commons_harvest__open',
    'commons_harvest__partnership',
    'coop_mining',
    'daycare',
    'externality_mushrooms__dense',
    'factory_commons__either_or',
    'fruit_market__concentric_rivers',
    'gift_refinements',
    'hidden_agenda',
    'paintball__capture_the_flag',
    'paintball__king_of_the_hill',
    'predator_prey__alley_hunt',
    'predator_prey__open',
    'predator_prey__orchard',
    'predator_prey__random_forest',
    'prisoners_dilemma_in_the_matrix__arena',
    'prisoners_dilemma_in_the_matrix__repeated',
    'pure_coordination_in_the_matrix__arena',
    'pure_coordination_in_the_matrix__repeated',
    'rationalizable_coordination_in_the_matrix__arena',
    'rationalizable_coordination_in_the_matrix__repeated',
    'running_with_scissors_in_the_matrix__arena',
    'running_with_scissors_in_the_matrix__one_shot',
    'running_with_scissors_in_the_matrix__repeated',
    'stag_hunt_in_the_matrix__arena',
    'stag_hunt_in_the_matrix__repeated',
    'territory__inside_out',
    'territory__open',
    'territory__rooms',
    # keep-sorted end
"""
from shimmy import MeltingPotCompatibilityV0

# env = MeltingPotCompatibilityV0(substrate_name='coins', render_mode="human")
# env = MeltingPotCompatibilityV0(substrate_name='commons_harvest__open', render_mode="human")
env = MeltingPotCompatibilityV0(substrate_name='clean_up', render_mode='human')


# from shimmy.utils.meltingpot import load_meltingpot

# env = load_meltingpot("prisoners_dilemma_in_the_matrix__arena")
# env = MeltingPotCompatibilityV0(env, render_mode="human")




# 设置自定义 agent 数量
desired_num_agents = 1
env._num_players = desired_num_agents
env.possible_agents = [env.PLAYER_STR_FORMAT.format(index=i) for i in range(desired_num_agents)]
env.agents = env.possible_agents[:]


observations = env.reset()
while env.agents:
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)
    rewards = {agent: reward for agent, reward in rewards.items()}
    print(rewards)
env.close()