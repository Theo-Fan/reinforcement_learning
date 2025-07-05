import jax
import jax.numpy as jnp

import optax
from flax import linen as nn
from typing import Sequence

import hydra

import time
import random
import collections

import gymnasium as gym
from omegaconf import DictConfig, OmegaConf


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)

        batch_data = {
            'state': states,
            'action': actions,
            'reward': rewards,
            'next_state': next_states,
            'done': dones
        }
        return batch_data

    def size(self):
        return len(self.buffer)


class Q_Network_Flax(nn.Module):
    # Flax 内部使用 @dataclass 自动生成 __init__ 和属性。
    hidden_size: Sequence[int]
    action_dim: int

    @nn.compact
    def __call__(self, x):
        for sz in self.hidden_size:
            x = nn.Dense(sz)(x)
            x = nn.relu(x)

        return nn.Dense(self.action_dim)(x)

# @jax.jit # 传入的参数必须是 jax 的可识别类型
def take_action(cfg, env, model, params, state, rng, is_epsilon=True):
    rng, sub_rng = jax.random.split(rng)
    if is_epsilon and jax.random.uniform(sub_rng) < cfg.algo.epsilon:
        return int(env.action_space.sample()), rng
    state = jnp.asarray(state, dtype=jnp.float32)
    return int(jnp.argmax(model.apply(params, state), -1)), rng

@jax.jit # 传入的参数必须是 jax 的可识别类型
def net_update(cfg, model, params_q, params_tar, optimizer_state, optimizer, data):
    # change data to jax array
    states = jnp.asarray(data['state'], dtype=jnp.float32)
    actions = jnp.asarray(data['action'], dtype=jnp.int32).reshape(-1, 1)
    rewards = jnp.asarray(data['reward'], dtype=jnp.float32).reshape(-1, 1)
    next_states = jnp.asarray(data['next_state'], dtype=jnp.float32)
    dones = jnp.asarray(data['done'], dtype=jnp.float32).reshape(-1, 1)

    def loss_fn(params_q):
        q_val = jnp.take_along_axis(model.apply(params_q, states), actions, 1)
        ne_q_val = jnp.max(model.apply(params_tar, next_states), axis=1, keepdims=True)

        td_target = rewards + (1 - dones) * cfg.algo.gamma * ne_q_val
        loss = jnp.mean((q_val - td_target) ** 2)
        return loss

    # compute gradients
    loss, grads = jax.value_and_grad(loss_fn)(params_q)

    # update parameters
    updates, optimizer_state = optimizer.update(grads, optimizer_state)
    params_q = optax.apply_updates(params_q, updates)

    return params_q, optimizer_state, loss


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    # cfg = OmegaConf.to_container(cfg)

    env = gym.make(cfg.env.env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    q_net = Q_Network_Flax(hidden_size=[cfg.net.hidden_dim for i in range(cfg.net.hidden_deep)], action_dim=action_dim)

    input_rng = jnp.ones((state_dim,))

    rng = jax.random.PRNGKey(cfg.net.seed)
    rng, rng_q = jax.random.split(rng)
    rng, rng_target = jax.random.split(rng)
    rng, rng_take_a = jax.random.split(rng)

    # define two networks by different rng, and sharing the same network structure
    params_q_net = q_net.init(rng_q, input_rng)
    params_tar_net = q_net.init(rng_target, input_rng)

    # optimizer
    optimizer = optax.adam(cfg.algo.lr)
    optimizer_state = optimizer.init(params_q_net)

    replay_buffer = ReplayBuffer(cfg.algo.replay_buffer_size)

    # ========== Training Loop ==========
    for episode in range(cfg.env.episodes):
        state, _ = env.reset()

        for step in range(cfg.env.max_steps):

            # TODO: Implement epsilon-greedy action selection
            action, rng_take_a = take_action(cfg, env, q_net, params_q_net, state, rng_take_a)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            replay_buffer.add(state, action, reward, next_state, done)

            if replay_buffer.size() > cfg.algo.mini_size:
                data = replay_buffer.sample(cfg.algo.batch_size)
                params_q_net, optimizer_state, loss = net_update(cfg, q_net, params_q_net, params_tar_net,
                                                                 optimizer_state,
                                                                 optimizer, data)  # TODO

            if step % cfg.algo.target_update_freq == 0:
                params_tar_net = params_q_net

            state = next_state
            if done:
                break

            # if episode % 5 == 0:
            print(f"Episode: {episode} / {cfg.env.episodes}, Step: {step} ")

    env = gym.make(cfg.env.env_name, render_mode='human')
    state, info = env.reset()
    done = False
    while not done:
        action, rng_take_a = take_action(cfg, env, q_net, params_q_net, state, rng_take_a, is_epsilon=False)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        state = next_state
        print(' Test ---- action, reward, obs, done: ', action, reward, state, done)
        time.sleep(0.05)


if __name__ == '__main__':
    main()
