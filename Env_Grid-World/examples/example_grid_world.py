
import sys

sys.path.append("..")
from src.grid_world import GridWorld

import random
import numpy as np

# Example usage:
if __name__ == "__main__":             
    env = GridWorld()
    state = env.reset()               
    for t in range(1000):
        env.render(animation_interval=0.5)    # the figure will stop for 1 seconds
        action = random.choice(env.action_space)
        """
            action_space: [(0, 1), (1, 0), (0, -1), (-1, 0), (0, 0)] # 下， 右， 上， 左， 原地
        """
        next_state, reward, done, info = env.step(action)
        print(f"Step: {t}, Action: {action}, State: {next_state+(np.array([1,1]))}, Reward: {reward}, Done: {done}")
        # if done:
        #     break
    
    # Add policy
    policy_matrix=np.random.rand(env.num_states,len(env.action_space))                                            
    policy_matrix /= policy_matrix.sum(axis=1)[:, np.newaxis]  # make the sum of elements in each row to be 1

    print(type(policy_matrix))
    print(policy_matrix)
    env.add_policy(policy_matrix)


    # Add state values
    values = np.random.uniform(0,10,(env.num_states,))
    env.add_state_values(values)



    # Render the environment
    env.render(animation_interval=2)