"""
    A model-free variant of policy iteration
"""
import sys
import numpy as np

sys.path.append("../..")
from src.grid_world import GridWorld

gamma = 0.9


def init_random_deterministic_policy(num_states, num_actions):
    policy = np.zeros((num_states, num_actions))
    for s in range(num_states):
        policy[s, np.random.choice(num_actions - 1)] = 1
    return policy


def mc_exploring_starts(env, gamma, num_episodes=3000):
    num_states = env.num_states
    num_actions = len(env.action_space)

    policy = init_random_deterministic_policy(num_states, num_actions)  # init random policy
    print("Init Optimal Policy:")
    print(policy)

    return_cnt = np.zeros((num_states, num_actions))  # Count of returns for each (state, action)
    return_sum = np.zeros((num_states, num_actions))  # Sum of returns for each (state, action)

    for episode in range(num_episodes):
        # === Step 1: Exploring Starts: Randomly choose a starting state and action ===
        state = np.random.choice(num_states)
        env.set_state((state % env.env_size[1], state // env.env_size[1]))  # set agent position

        # pos = env.agent_state[1] * env.env_size[0] + env.agent_state[0]
        # action = env.action_space[np.argmax(policy[pos])]  # get action from policy when agent is in start state
        action = env.action_space[np.random.choice(5)]

        episode_data = []  # Store (state, action, reward) tuples, Generate an episode starting from (s0, a0)

        for i in range(100):
            next_state, reward, done, _ = env.step(action)
            episode_data.append((state, action, reward))  # store (state, action, reward)
            state = next_state[1] * env.env_size[0] + next_state[0]  # update state to next state
            action = env.action_space[np.argmax(policy[state])]
            if done: break

        # debug
        # for i in range(5 if len(episode_data) > 5 else len(episode_data)):
        #     print(f"episode:{episode} idx: {i} data: {episode_data[i]}")

        # ==== Step 2: Policy Evaluation and Improvement ====
        g = 0
        vis = set()

        for t in range(1, len(episode_data) + 1):
            state, action, reward = episode_data[-t]
            g = gamma * g + reward

            # ==== First-visit method: Update Q only for first visits of (state, action) ====
            # if (state, action) not in vis:
            #     vis.add((state, action))
            #
            #     action_idx = env.action_space.index(action)  # change action to index
            #     return_cnt[state, action_idx] += 1
            #     return_sum[state, action_idx] = g / return_cnt[state, action_idx]
            #
            #     best_action_idx = np.argmax(return_sum[state])
            #     policy[state] = np.zeros(num_actions)
            #     policy[state, best_action_idx] = 1  # Update to greedy policy



            # ===== Every-visit method: Update Q for every visit of (state, action) =====
            action_idx = env.action_space.index(action)  # change action to index

            return_sum[state, action_idx] = g + return_sum[state, action_idx] * return_cnt[state, action_idx]
            return_cnt[state, action_idx] += 1
            return_sum[state, action_idx] /= return_cnt[state, action_idx]

            # Policy improvement: Make the policy greedy w.r.t. Q
            best_action_idx = np.argmax(return_sum[state])
            policy[state] = np.zeros(num_actions)
            policy[state, best_action_idx] = 1  # Update to greedy policy
    print(return_sum)
    return policy


def evaluate(env, optimal_policy):
    state = env.reset()
    for t in range(20):
        env.render(animation_interval=0.5)
        pos = env.agent_state[1] * env.env_size[0] + env.agent_state[0]
        action = env.action_space[np.argmax(optimal_policy[pos])]

        next_state, reward, done, info = env.step(action)
        print(
            f"Step: {t}, Action: {action}, Cur-state: {state}, Re-pos:{pos}, "
            f"Idx: {np.argmax(optimal_policy[pos])}, Next-state: {next_state}, Reward: {reward}, Done: {done}"
        )
        if done:
            break

    env.add_policy(optimal_policy * 1.5)
    env.render(animation_interval=10)


def main():
    env = GridWorld()
    optimal_policy = mc_exploring_starts(env, gamma)
    evaluate(env, optimal_policy)

    print("Optimal Policy:")
    print(optimal_policy)


if __name__ == "__main__":
    main()
