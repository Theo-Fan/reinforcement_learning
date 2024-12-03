import random
import sys
import numpy as np

sys.path.append("../..")
from src.grid_world import GridWorld

gamma = 0.9
num_episode = 1000
alpha = 0.1
epsilon = 0.1


def init_random_deterministic_policy(num_states, num_actions):
    matrix = np.zeros((num_states, num_actions))
    for row in matrix:
        row[np.random.randint(0, num_actions)] = 1
    return matrix


def epsilon_greedy_policy(state, policy_matrix, epsilon, action_space):
    if random.uniform(0, 1) < epsilon:
        action = random.choice(action_space)  # 随机探索
    else:
        action = action_space[np.argmax(policy_matrix[state])]  # 选择最优动作
    return action


def train_sarsa(env, q_table, policy_matrix, num_episode, alpha, gamma, epsilon):
    for episode in range(num_episode):
        env.reset()
        pos = env.agent_state[1] * env.env_size[0] + env.agent_state[0]
        action = epsilon_greedy_policy(pos, policy_matrix, epsilon, env.action_space)
        for t in range(1000):
            next_state, reward, done, info = env.step(action)
            ne_pos = next_state[1] * env.env_size[0] + next_state[0]
            next_action = epsilon_greedy_policy(ne_pos, policy_matrix, epsilon, env.action_space)

            # update q-value [policy improvement]
            q_table[pos][env.action_space.index(action)] -= alpha * (
                    q_table[pos][env.action_space.index(action)] -
                    (reward + gamma * q_table[ne_pos][env.action_space.index(next_action)])
            )

            best_action = np.argmax(q_table[ne_pos])
            policy_matrix[ne_pos] = epsilon / (len(env.action_space) - 1)
            policy_matrix[ne_pos][best_action] += 1 - epsilon

            # important
            action = next_action
            pos = ne_pos

            if done:
                break

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{num_episode} completed.")

    return policy_matrix


def test_policy(env, policy_matrix):
    env.reset()
    for t in range(1000):
        env.render(animation_interval=0.5)
        state = env.agent_state
        pos = env.agent_state[1] * env.env_size[0] + env.agent_state[0]
        action = env.action_space[np.argmax(policy_matrix[pos])]

        next_state, reward, done, info = env.step(action)
        print(
            f"Step: {t}, Action: {action}, Cur-state: {state}, Re-pos: {pos}, "
            f"Idx: {np.argmax(policy_matrix[pos])}, Next-state: {next_state}, Reward: {reward}, Done: {done}")
        if done:
            break


if __name__ == "__main__":
    env = GridWorld()

    q_table = np.zeros((env.num_states, len(env.action_space)))
    policy_matrix = init_random_deterministic_policy(env.num_states, len(env.action_space))

    # train
    policy_matrix = train_sarsa(env, q_table, policy_matrix, num_episode, alpha, gamma, epsilon)
    # test
    test_policy(env, policy_matrix)

    print("Policy Matrix:", policy_matrix)

    env.add_policy(policy_matrix)
    env.render(animation_interval=10)
