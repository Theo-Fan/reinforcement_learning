from shimmy import MeltingPotCompatibilityV0

env = MeltingPotCompatibilityV0(substrate_name='clean_up', render_mode='human')

observations = env.reset()
while env.agents:
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    """ action spaces
        0: stop;    1: forward;  2: backward;   3: left;    4: right
        5: turn left;   6: turn right;  7: attack;  8: clean up
    """
    observations, rewards, terminations, truncations, infos = env.step(actions)

    for agent in rewards:
        if rewards[agent] != 0:
            print('agent:', agent, 'reward:', rewards[agent])

env.close()
