import sys

sys.path.append("../..")
from env.grid_world import GridWorld

import numpy as np

gamma = 0.9


def init_random_determinatal_policy(num_states, num_actions):
    matrix = np.zeros((num_states, num_actions))

    for row in matrix:
        row[np.random.randint(0, num_actions)] = 1

    return matrix

def evaluate(env, policy_matrix, v):
    state = env.reset()

    for t in range(1000):
        env.render(animation_interval=0.5)  # the figure will stop for 1 seconds

        pos = env.agent_state[1] * env.env_size[0] + env.agent_state[0]
        action = env.action_space[np.argmax(policy_matrix[pos])]

        next_state, reward, done, info = env.step(action)
        print(
            f"Step: {t}, Action: {action}, Cur-state: {env.agent_state}, Re-pos:{pos}, "
            f"Next-state: {next_state}, Reward: {reward}, Done: {done}")
        if done:
            break

    env.add_policy(policy_matrix)

    env.add_state_values(v)

    env.render(animation_interval=10)  # finally render 10 interval


def main():
    env = GridWorld()
    # reset: ((0, 0), {})
    # action_space: [(0, 1), (1, 0), (0, -1), (-1, 0), (0, 0)] down, right, up, left, stay
    # num_states: 25 # 从左到右，从上到下增大

    v = np.zeros(env.num_states)
    q_tabel = np.zeros((env.num_states, len(env.action_space)))
    policy_matrix = init_random_determinatal_policy(env.num_states, len(env.action_space))


    for t in range(30):
        v = np.zeros(env.num_states)

        # Policy Evaluation
        for i in range(500):
            for s in range(env.num_states):
                state = env.set_state((s % env.env_size[1], s // env.env_size[1]))  # up -> down, left -> right

                pos = env.agent_state[1] * env.env_size[0] + env.agent_state[0]
                action = env.action_space[np.argmax(policy_matrix[pos])]
                next_state, reward, done, info = env.step(action)

                c_s = state[0][1] * env.env_size[0] + state[0][0]
                n_s = next_state[1] * env.env_size[0] + next_state[0]

                v[c_s] = reward + gamma * v[n_s]


        # Policy Improvement
        for s in range(env.num_states):
            for idx, a in enumerate(env.action_space):
                init_state = env.set_state((s % env.env_size[1], s // env.env_size[1]))
                next_state, reward, done, info = env.step(a)
                n_s = next_state[1] * env.env_size[0] + next_state[0]
                q_tabel[s, idx] = reward + gamma * v[n_s]

            max_action_index = np.argmax(q_tabel[s])
            max_action_value = np.max(q_tabel[s])

            policy_matrix[s] = np.zeros(len(env.action_space))
            policy_matrix[s, max_action_index] = 1

    print(f"Policy Matrix: {policy_matrix}, Shape: {policy_matrix.shape}")

    evaluate(env, policy_matrix, v)



if __name__ == "__main__":
    main()