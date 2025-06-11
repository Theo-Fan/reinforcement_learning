import sys

sys.path.append("../..")
from env.grid_world import GridWorld

import numpy as np

gamma = 0.9

def evaluate(env, policy_matrix, v):
    state = env.reset()

    for t in range(1000):
        env.render(animation_interval=0.5)  # the figure will stop for 1 seconds
        state = env.agent_state
        pos = env.agent_state[1] * env.env_size[0] + env.agent_state[0]
        action = env.action_space[np.argmax(policy_matrix[pos])]

        next_state, reward, done, info = env.step(action)
        print(f"Step: {t}, Action: {action}, Cur-state: {state}, Re-pos:{pos}, "
              f"Idx:{np.argmax(policy_matrix[pos])}, Next-state: {next_state}, Reward: {reward}, Done: {done}")
        if done:
            break

    env.add_policy(policy_matrix)

    env.add_state_values(v)

    env.render(animation_interval=10)  # finally render 10 interval

def main():
    env = GridWorld()
    # reset: ((0, 0), {})
    # action_space: [(0, 1), (1, 0), (0, -1), (-1, 0), (0, 0)] down, right, up, left, stay
    # num_states: 25
    v = np.zeros(env.num_states)
    q_table = np.zeros((env.num_states, len(env.action_space)))
    policy_matrix = np.zeros((env.num_states, len(env.action_space)))

    for t in range(1000):

        for s in range(env.num_states):
            for idx, a in enumerate(env.action_space):
                init_state = env.set_state((s % env.env_size[1], s // env.env_size[1]))  # up -> down, left -> right
                next_state, reward, done, info = env.step(a)
                n_s = next_state[1] * env.env_size[0] + next_state[0]  # up -> down, left -> right
                q_table[s, idx] = reward + gamma * v[n_s]

            # get max action
            max_action_index = np.argmax(q_table[s])
            max_action_value = np.max(q_table[s])

            # policy update
            policy_matrix[s] = np.zeros(len(env.action_space))
            policy_matrix[s, max_action_index] = 1

            v[s] = max_action_value  # value update

    print(f"Value vector: {v}, Shape:{v.shape}")
    print(f"Policy Matrix: {policy_matrix}, Shape: {policy_matrix.shape}")

    # 验证
    evaluate(env, policy_matrix, v)





if __name__ == "__main__":
    main()