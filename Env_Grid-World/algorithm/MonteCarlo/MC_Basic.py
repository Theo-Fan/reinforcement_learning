import sys
import numpy as np

sys.path.append("../..")
from src.grid_world import GridWorld

gamma = 0.9


def init_random_deterministic_policy(num_states, num_actions):
    matrix = np.zeros((num_states, num_actions))
    for row in matrix:
        row[np.random.randint(0, num_actions)] = 1
    return matrix


def monte_carlo_policy_iteration(env, policy_matrix, q_table, num_iterations=100, trajectory_length=100):
    for t in range(num_iterations):
        for s in range(env.num_states):
            for idx, action in enumerate(env.action_space):
                env.set_state((s % env.env_size[1], s // env.env_size[1]))
                reward_trace = []
                # collect_q = [] # since policy and environment is deterministic, only need to take one trajectory
                cnt = 1

                # take trajectory
                # 说明：此处对于每个状态只有一个固定的策略，因此多个trajectory的均值和单个trajectory相同，因此无需多次采样
                for _ in range(trajectory_length): # trajectory_length 需要设置的长一些，以保证能访问到 target area
                    pos = env.agent_state[1] * env.env_size[0] + env.agent_state[0]
                    if cnt == 1:
                        next_state, reward, done, info = env.step(action)
                    else:
                        best_action = env.action_space[np.argmax(policy_matrix[pos])]
                        next_state, reward, done, info = env.step(best_action)
                        # print(f"Step: {cnt}, Action: {a_c}, Cur-state: {cur_state}, Re-pos:{pos}, Next-state: {next_state}, Reward: {reward}, Done: {done}")

                    reward_trace.append(reward)
                    cnt += 1
                    if done:
                        break

                for item in reversed(reward_trace):  # reverse reward
                    q_table[s, idx] = item + gamma * q_table[s, idx]

            # Policy Improvement
            max_action_index = np.argmax(q_table[s])
            policy_matrix[s] = np.zeros(len(env.action_space))
            policy_matrix[s, max_action_index] = 1

    return policy_matrix


if __name__ == "__main__":
    # reset: ((0, 0), {})
    # action_space: [(0, 1), (1, 0), (0, -1), (-1, 0), (0, 0)] down, right, up, left, stay
    # num_states: 25

    env = GridWorld()
    q_table = np.zeros((env.num_states, len(env.action_space)))
    policy_matrix = init_random_deterministic_policy(env.num_states, len(env.action_space))

    policy_matrix = monte_carlo_policy_iteration(env, policy_matrix, q_table)

    print(f"Policy Matrix: {policy_matrix}, Shape: {policy_matrix.shape}")

    state = env.reset()
    for t in range(100):
        env.render(animation_interval=0.5)
        pos = env.agent_state[1] * env.env_size[0] + env.agent_state[0]
        action = env.action_space[np.argmax(policy_matrix[pos])]

        next_state, reward, done, info = env.step(action)
        print(
            f"Step: {t}, Action: {action}, Cur-state: {state}, Re-pos:{pos}, "
            f"Idx: {np.argmax(policy_matrix[pos])}, Next-state: {next_state}, Reward: {reward}, Done: {done}"
        )
        if done:
            break

    env.add_policy(policy_matrix)
    env.render(animation_interval=10)
