import copy
from functools import partial
from typing import Any

import chex
import flax
import flax.linen as nn
import flax.training.train_state as train_state
import jax
import jax.numpy as jnp
import numpy as np
import optax
import ray
import tqdm
from jux.config import JuxBufferConfig
from jux.env import JuxEnv
from jux.state import State as JuxState

from dynamics import fresh_state, step_best
from evaluation import evaluate, make_replay
from observation import get_obs
from utils import Logger


class QNet(nn.Module):
    # returns predicted Q-values (for each agent)
    # current plan is do things VDN-style and just sum the local Q-values
    @nn.compact
    def __call__(self, obs):
        x = nn.Conv(8, kernel_size=(1, 1))(obs)
        # x = nn.relu(x)  # B, H, W, C
        # x = nn.GroupNorm(num_groups=8)(x)
        #
        # r = nn.Conv(64, kernel_size=(3, 3))(x)
        # w = r.mean(axis=(-2, -3))  # B, C
        # w = nn.Dense(8)(w)
        # w = nn.relu(w)
        # w = nn.Dense(64)(w)  # B, C
        # w = nn.sigmoid(w)
        #
        # x = w[..., None, None, :] * r  # B, H, W, C
        # x = nn.Conv(8, kernel_size=(1, 1))(x)
        return x


class TrainState(train_state.TrainState):
    slow_params: flax.core.FrozenDict[str, Any]
    slow_update_speed: float = 0.01

    def apply_gradients(self, *, grads, **kwargs):
        updates, new_opt_state = self.tx.update(
            grads, self.opt_state, self.params
        )
        new_params = optax.apply_updates(self.params, updates)
        new_slow_params = jax.tree_map(
            lambda p1, p2: self.slow_update_speed * p1
            + (1 - self.slow_update_speed) * p2,
            new_params,
            self.slow_params,
        )
        return self.replace(
            step=self.step + 1,
            params=new_params,
            slow_params=new_slow_params,
            opt_state=new_opt_state,
            slow_update_speed=self.slow_update_speed,
            **kwargs,
        )


vmapped_obs = jax.vmap(get_obs)
vmapped_step_best = jax.vmap(
    jax.jit(step_best, static_argnums=0), in_axes=(None, 0, 0)
)


def predict_target_q(env, ts, states: JuxState):
    obs = vmapped_obs(states)
    chex.assert_shape(obs, (obs.shape[0], 48, 48, 9))
    utility = ts.apply_fn({"params": ts.params}, obs)
    _, actions, _, _ = vmapped_step_best(env, states, utility)
    # scores = utility.at[states.units.pos.x, states.units.pos.y].get(
    #     mode="fill", fill_value=0
    # )
    # actions = jnp.zeros(
    #     (obs.shape[0], 2, env.buf_cfg.MAX_N_UNITS), dtype=jnp.int8
    # )
    # actions = actions.at[:, 0].set(
    #     jnp.argmax(scores[:, 0], axis=-1).astype(jnp.int8)
    # )
    # actions = actions.at[:, 1].set(
    #     jnp.argmin(scores[:, 1], axis=-1).astype(jnp.int8)
    # )

    # double q-learning style: select actions with fast, set target w/ slow
    slow_utility = ts.apply_fn({"params": ts.slow_params}, obs)
    chex.assert_shape(slow_utility, (obs.shape[0], 48, 48, 8))
    chex.assert_shape(actions, (obs.shape[0], 2, env.buf_cfg.MAX_N_UNITS))
    selected_utility = slow_utility.at[
        jnp.arange(obs.shape[0])[:, None, None],
        states.units.pos.x,
        states.units.pos.y,
        actions,
    ].get(
        mode="fill", fill_value=0.0
    )  # B, 2, U
    chex.assert_shape(
        selected_utility, (obs.shape[0], 2, env.buf_cfg.MAX_N_UNITS)
    )
    return selected_utility.mean(axis=(-1, -2))


@partial(jax.jit, static_argnums=2)
def td_loss(params, ts, env, states, noise, rng):
    obs = vmapped_obs(states)
    # utility = ts.apply_fn({"params": params}, obs)
    # next_states, actions, rewards, dones = vmapped_step_best(
    #     env,
    #     states,
    #     (
    #         jax.lax.stop_gradient(utility)
    #         + noise * jax.random.gumbel(rng, utility.shape)
    #     ),
    # )
    # scores = utility.at[states.units.pos.x, states.units.pos.y].get(
    #     mode="fill", fill_value=0
    # )
    # actions = jnp.zeros(
    #     (obs.shape[0], 2, env.buf_cfg.MAX_N_UNITS), dtype=jnp.int8
    # )
    # actions = actions.at[:, 0].set(
    #     jnp.argmax(scores[:, 0], axis=-1).astype(jnp.int8)
    # )
    # actions = actions.at[:, 1].set(
    #     jnp.argmin(scores[:, 1], axis=-1).astype(jnp.int8)
    # )

    # selected_utility = utility.at[
    #     jnp.arange(obs.shape[0])[:, None, None],
    #     states.units.pos.x,
    #     states.units.pos.y,
    #     actions,
    # ].get(mode="fill", fill_value=0.0)
    # q_preds = selected_utility.mean(axis=(-1, -2))
    #
    # # TODO restore
    # # q_target = reward + (1 - done) * predict_target_q(env, ts, next_state)
    # # next_q = predict_target_q(env, ts, next_states)
    # # chex.assert_shape(next_q, (obs.shape[0],))
    # # q_targets = rewards + (1 - dones) * 0.99 * next_q  # TODO restore
    # q_targets = ((actions == 2).sum(axis=(1, 2)) - (actions == 4).sum(axis=(1, 2))) / 1000
    # chex.assert_shape(q_targets, (obs.shape[0],))
    #
    # loss = jnp.mean((q_preds - jax.lax.stop_gradient(q_targets)) ** 2)
    # return (
    #     loss,
    #     (next_states, dones, q_preds, q_targets),
    # )
    utility = ts.apply_fn({"params": params}, obs)
    loss = (-utility[..., 2] + utility[..., 4]).mean() + 0.1 * (
        utility**2
    ).mean()
    return loss


jitted_step_best = jax.jit(step_best, static_argnums=0)


# returns a batched state with a size n_timesteps batch dimension
def explore(rng, env, n_timesteps, verbose=False) -> JuxState:
    rng, key = jax.random.split(rng)

    states = []
    curr = fresh_state(env, key)
    for _ in tqdm.trange(n_timesteps) if verbose else range(n_timesteps):
        states.append(curr)

        rng, key = jax.random.split(rng)
        curr, _, _, done = jitted_step_best(
            env, curr, jax.random.gumbel(key, (48, 48, 8))
        )
        if done:
            rng, key = jax.random.split(rng)
            curr = fresh_state(env, key)
    return jax.tree_map(lambda *xs: np.stack(xs, axis=0), *states)


def main():
    env = JuxEnv(buf_cfg=JuxBufferConfig(MAX_N_UNITS=250))
    rng = jax.random.PRNGKey(42)

    replay_buffer_size = 1000
    rng, key = jax.random.split(rng)
    replay_buffer = explore(key, env, replay_buffer_size, verbose=True)

    rng, key = jax.random.split(rng)
    model = QNet()
    params = model.init(rng, get_obs(fresh_state(env, rng)))["params"]

    ts = TrainState.create(
        apply_fn=model.apply,
        params=params,
        slow_params=params,
        tx=optax.adam(1e-3),
    )
    # logger = Logger()

    rng, key = jax.random.split(rng)

    minibatch_size = 128

    rng, key = jax.random.split(rng)
    # initial_params = copy.deepcopy(ts.params)
    replay_task = ray.remote(make_replay)

    tasks = []
    for i in tqdm.trange(1001):
        rng, key = jax.random.split(rng)
        idx = jax.random.choice(key, replay_buffer_size, (minibatch_size,))

        rng, key = jax.random.split(rng)
        states = jax.tree_map(lambda x: x[idx], replay_buffer)
        grads = jax.grad(td_loss)(ts.params, ts, env, states, 10.0, key)
        ts = ts.apply_gradients(grads=grads)
        #         noise = 99.9 * np.exp(-i / 100) + 0.1
        #         (
        #             vals,
        #             (next_states, dones, q_pred, q_target),
        #         ), grads = jax.value_and_grad(td_loss, has_aux=True)(
        #             ts.params,
        #             ts,
        #             env,
        #             states,
        #             noise,
        #             key,
        #         )
        #         jax.tree_map(
        #             lambda x, y: x.__setitem__(idx, y), replay_buffer, next_states
        #         )
        #
        #         logger.log(
        #             td_loss=vals.mean().item(),
        #             log_td_loss=np.log(vals).mean().item(),
        #             rollout_t=states.real_env_steps.mean().item(),
        #             noise=noise,
        #             q_pred=q_pred.mean().item(),
        #             q_target=q_target.mean().item(),
        #         )
        #
        #         grads = jax.tree_map(lambda x: x.mean(axis=0), grads)
        #         ts = ts.apply_gradients(grads=grads)
        #
        #         n_terminal = 0
        #         for j in range(minibatch_size):
        #             if dones[j]:
        #                 rng, key = jax.random.split(rng)
        #                 jax.tree_map(
        #                     lambda x, y: x.__setitem__(idx[j], y),
        #                     replay_buffer,
        #                     fresh_state(env, key),
        #                 )
        #                 n_terminal += 1
        #         logger.log(env_resets=n_terminal)
        #
        if i % 50 == 0:
            # rng, key = jax.random.split(rng)
            # temp_logger_1 = Logger()
            # evaluate(key, 10, temp_logger_1, model, ts.params, initial_params)
            # temp_logger_2 = Logger()
            # evaluate(key, 10, temp_logger_2, model, initial_params, ts.params)
            #
            # for k in temp_logger_1.cumulative_data.keys():
            #     logger.cumulative_data["p0_" + k].append(
            #         temp_logger_1.cumulative_data[k][-1]
            #     )
            #     logger.cumulative_data["p1_" + k].append(
            #         temp_logger_2.cumulative_data[k][-1]
            #     )
            #
            rng, key = jax.random.split(rng)
            tasks.append(replay_task.remote(
                key,
                f"replay_{i}",
                model,
                ts.params,
                ts.params,
            ))
            # logger.generate_plots()
    ray.get(tasks)


if __name__ == "__main__":
    main()
