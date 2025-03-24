import random
import sys
import numpy as np

sys.path.append("../..")
from src.grid_world import GridWorld

gamma = 0.9
num_episode = 1000
alpha = 0.1
epsilon = 0.1
n = 5  # n-step SARSA 的 n 值


def init_random_qtable(num_states, num_actions):
    policy = np.full((num_states, num_actions), 0.2)
    return policy


def epsilon_greedy(state, policy, epsilon, action_space):
    action_idx = np.argmax(policy[state])
    if np.random.random() < epsilon:
        tmp_action = [i for i in range(len(action_space)) if i != action_idx]
        action_idx = np.random.choice(tmp_action)
    action = action_space[action_idx]
    return action


def compute_n_step_return(reward_queue, gamma, n, q_table, last_state, last_action):
    """
    计算 n-step 回报
    G_t^(n) = R_t+1 + γR_t+2 + ... + γ^(n-1)R_t+n + γ^nQ(S_t+n, A_t+n)
    """
    G = 0
    for i in range(len(reward_queue)):
        G += (gamma ** i) * reward_queue[i]

    # 加上最后的 Q 值项
    G += (gamma ** n) * q_table[last_state][last_action]
    return G


def train_n_step_sarsa(env, q_table, num_episode, alpha, gamma, epsilon, n):
    for episode in range(num_episode):
        env.reset()

        # 初始化滑动窗口
        state_queue = []  # 最近 n 步的状态
        action_queue = []  # 最近 n 步的动作
        reward_queue = []  # 最近 n 步的奖励

        pos = env.agent_state[1] * env.env_size[0] + env.agent_state[0]
        action = epsilon_greedy(pos, q_table, epsilon, env.action_space)

        state_queue.append(pos)
        action_queue.append(env.action_space.index(action))

        for t in range(1000):
            next_state, reward, done, info = env.step(action)
            ne_pos = next_state[1] * env.env_size[0] + next_state[0]
            next_action = epsilon_greedy(ne_pos, q_table, epsilon, env.action_space)

            reward_queue.append(reward)
            state_queue.append(ne_pos)
            action_queue.append(env.action_space.index(next_action))

            # 窗口满了，计算 n-step 回报
            if len(reward_queue) > n:
                G = compute_n_step_return(
                    reward_queue, gamma, n, q_table, state_queue[-1], action_queue[-1]
                )

                # 更新 Q 值
                q_table[state_queue[0]][action_queue[0]] -= alpha * (
                    q_table[state_queue[0]][action_queue[0]] - G
                )

                # 更新策略矩阵
                # best_action = np.argmax(q_table[state_queue[0]])
                # policy_matrix[state_queue[0]] = epsilon / len(env.action_space)
                # policy_matrix[state_queue[0]][best_action] += 1 - epsilon

                # 移除窗口最早的元素
                reward_queue.pop(0)
                state_queue.pop(0)
                action_queue.pop(0)

            # 更新动作和状态
            action = next_action
            pos = ne_pos

            if done:
                # 对剩余的未完成窗口进行清算
                while len(reward_queue) > 0:
                    G = compute_n_step_return(
                        reward_queue, gamma, len(reward_queue), q_table, state_queue[-1], action_queue[-1]
                    )
                    q_table[state_queue[0]][action_queue[0]] -= alpha * (
                        q_table[state_queue[0]][action_queue[0]] - G
                    )

                    # 更新策略矩阵
                    # best_action = np.argmax(q_table[state_queue[0]])
                    # policy_matrix[state_queue[0]] = epsilon / len(env.action_space)
                    # policy_matrix[state_queue[0]][best_action] += 1 - epsilon

                    # 移除窗口最早的元素
                    reward_queue.pop(0)
                    state_queue.pop(0)
                    action_queue.pop(0)
                break

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{num_episode} completed.")



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
            f"Idx: {np.argmax(policy_matrix[pos])}, Next-state: {next_state}, Reward: {reward}, Done: {done}"
        )
        if done:
            break


if __name__ == "__main__":
    env = GridWorld()

    # q_table = np.zeros((env.num_states, len(env.action_space)))
    q_table = init_random_qtable(env.num_states, len(env.action_space))

    # 训练 n-step SARSA
    train_n_step_sarsa(env, q_table, num_episode, alpha, gamma, epsilon, n)
    # 测试策略
    test_policy(env, q_table)

    env.add_policy(q_table)
    print(q_table)
    env.render(animation_interval=10)