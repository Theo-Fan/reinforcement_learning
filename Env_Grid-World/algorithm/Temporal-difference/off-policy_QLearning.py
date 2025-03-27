import sys

sys.path.append("../..")
from env.grid_world import GridWorld
import numpy as np


"""
For off-policy Q-learning, We need to ensure the sampling is fruitful.
Here are two methods:
    1. set different parameter: epsilon
    2. take more timestep to collect the experience 
"""

gamma = 0.9
alpha = 0.1
""" =====> Method 1 <====="""
epsilon = 0.5 # 0.9


def init_random_qtable(num_states, num_actions):
    matrix = np.full((num_states, num_actions), 0.2)
    return matrix


def epsilon_greedy(state, policy, epsilon, action_space):
    action_idx = np.argmax(policy[state])
    if np.random.uniform(0, 1) < epsilon:
        tmp_action = [i for i in range(len(action_space)) if i != action_idx]
        action_idx = np.random.choice(tmp_action)
    action = action_space[action_idx]
    return action


def generate_episode(env, policy):
    tot_episode = []
    env.reset()
    """ =====> Method 2 <====="""
    timestep = 5000 # 500
    for j in range(timestep):
        s = np.random.randint(env.num_states)
        env.set_state((s % env.env_size[1], s // env.env_size[1]))

        for i in range(10):
            pos = env.agent_state[1] * env.env_size[0] + env.agent_state[0]
            action = epsilon_greedy(pos, policy, epsilon, env.action_space)

            next_state, reward, done, info = env.step(action)
            ne_pos = next_state[1] * env.env_size[0] + next_state[0]

            tot_episode.append((pos, action, reward, ne_pos))

    return tot_episode


def train(env, q_table, policy):
    episode = generate_episode(env, policy)
    for i in episode:
        print(i)
    print(set(episode))
    print("the length of this episode", len(set(episode)))

    for idx, it in enumerate(episode):
        state, action, reward, next_state = it

        action_idx = env.action_space.index(action)
        q_ne_max = np.max(q_table[next_state])

        # update q-table
        q_table[state][action_idx] -= alpha * (
                q_table[state][action_idx] - (reward + gamma * q_ne_max)
        )

        # update policy
        max_idx = np.argmax(q_table[state])
        policy[state] = 0
        policy[state][max_idx] = 1


def test(env, q_table):
    env.reset()
    for t in range(1000):
        env.render(animation_interval=0.5)
        state = env.agent_state
        pos = env.agent_state[1] * env.env_size[0] + env.agent_state[0]
        action = env.action_space[np.argmax(q_table[pos])]

        next_state, reward, done, info = env.step(action)
        print(
            f"Step: {t}, Action: {action}, Cur-state: {state}, Re-pos: {pos}, "
            f"Idx: {np.argmax(q_table[pos])}, Next-state: {next_state}, Reward: {reward}, Done: {done}")
        if done:
            break


def main():
    env = GridWorld()
    num_state = env.num_states
    num_action = len(env.action_space)

    # q_table is random, we set all q-value is 0.2
    q_table = init_random_qtable(
        num_states=num_state,
        num_actions=num_action
    )

    # for all actions have same probability to be selected
    policy = init_random_qtable(
        num_states=num_state,
        num_actions=num_action
    )

    train(env, q_table, policy)
    print("Final Q-table: \n", q_table)
    print("Final policy: \n", policy)
    test(env, policy)



    env.add_policy(policy)

    env.render(animation_interval=10)


if __name__ == "__main__":
    main()
