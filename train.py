from functools import partial
from typing import Any

import flax
import flax.linen as nn
import flax.training.train_state as train_state
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tqdm
from jux.env import JuxEnv
from jux.state import State as JuxState

from dynamics import step_best
from observation import get_obs


class QNet(nn.Module):
    # returns predicted Q-values (for each agent)
    # current plan is do things VDN-style and just sum the local Q-values
    @nn.compact
    def __call__(self, obs):
        x = nn.Conv(32, kernel_size=(7, 7))(obs)
        x = nn.relu(x)
        x = nn.Conv(8, kernel_size=(1, 1))(x)
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


def choose_factory_spawn(rng, state: JuxState):
    spawns_mask = state.board.valid_spawns_mask
    selected = jax.random.choice(
        rng, 48 * 48, p=spawns_mask.ravel() / spawns_mask.sum()
    )
    coords = jnp.array([selected // 48, selected % 48], dtype=jnp.int8)
    return jnp.stack([coords, coords], axis=0)


def fresh_state(env, rng) -> JuxState:
    rng, key = jax.random.split(rng)
    seed = jax.random.randint(key, (), 0, 2**16)

    state = env.reset(seed)

    state, _ = env.step_bid(state, jnp.zeros(2), jnp.arange(2))

    def step_factory_placement(_i, args):
        rng_, state_ = args
        rng_, key_ = jax.random.split(rng_)
        action = choose_factory_spawn(key_, state_)
        new_s, _ = env.step_factory_placement(
            state_, action, jnp.array([150, 150]), jnp.array([150, 150])
        )
        return rng_, new_s

    rng, key = jax.random.split(rng)
    _, state = jax.lax.fori_loop(
        0,
        2 * state.board.factories_per_team.astype(jnp.int32),
        step_factory_placement,
        (key, state),
    )
    return state


def predict_target_q(env, ts, state: JuxState):
    obs = get_obs(state)
    utility = ts.apply_fn({"params": ts.params}, obs)
    _, actions, _, _ = step_best(env, state, utility)

    # double q-learning style: select actions with fast, set target w/ slow
    slow_utility = ts.apply_fn({"params": ts.slow_params}, obs)
    selected_utility = slow_utility.at[
        state.units.pos.x, state.units.pos.y, actions
    ].get(mode="fill", fill_value=0.0)
    team_utility = selected_utility.sum(axis=-1)
    return team_utility[0] - team_utility[1]


@partial(jax.jit, static_argnums=2)
def td_loss(params, ts, env, state, noise, rng):
    obs = get_obs(state)
    utility = ts.apply_fn({"params": params}, obs)
    next_state, actions, reward, done = step_best(
        env,
        state,
        (
            jax.lax.stop_gradient(utility)
            + noise * jax.random.gumbel(rng, utility.shape)
        ),
    )

    selected_utility = utility.at[
        state.units.pos.x, state.units.pos.y, actions
    ].get(mode="fill", fill_value=0.0)
    team_utility = selected_utility.sum(axis=-1)
    q_pred = team_utility[0] - team_utility[1]
    q_target = reward + (1 - done) * predict_target_q(env, ts, next_state)
    return (
        (q_pred - jax.lax.stop_gradient(q_target)) ** 2,
        (next_state, done),
    )


# def train_step(
#     env: JuxEnv, ts: TrainState, state: JuxState, noise, rng
# ) -> Tuple[TrainState, JuxState]:
#     rng, key = jax.random.split(rng)
#
#     grads, (next_state_, done_) = jax.grad(td_loss, has_aux=True)(ts.params)
#     next_state_ = jax.lax.cond(
#         done_,
#         lambda: fresh_state(env, rng),
#         lambda: next_state_,
#     )
#     return grads, next_state_


def main():
    env = JuxEnv()
    rng = jax.random.PRNGKey(42)

    replay_buffer_size = 1000
    rng, key = jax.random.split(rng)
    replay_buffer = jax.tree_map(
        lambda x: np.array(x),
        jax.vmap(fresh_state, in_axes=(None, 0))(
            env, jax.random.split(key, num=replay_buffer_size)
        ),
    )

    rng, key = jax.random.split(rng)
    model = QNet()
    params = model.init(rng, get_obs(fresh_state(env, rng)))["params"]

    ts = TrainState.create(
        apply_fn=model.apply,
        params=params,
        slow_params=params,
        tx=optax.adam(1e-3),
    )

    rng, key = jax.random.split(rng)

    jax_bs = jax.vmap(
        jax.grad(td_loss, has_aux=True), in_axes=(None, None, None, 0, None, 0)
    )

    minibatch_size = 128
    import time

    for _ in tqdm.trange(69):
        start_time = time.perf_counter()

        rng, key = jax.random.split(rng)
        idx = jax.random.choice(key, replay_buffer_size, (minibatch_size,))

        rng, key = jax.random.split(rng)
        states = jax.tree_map(lambda x: x[idx], replay_buffer)
        grads, (next_states, dones) = jax_bs(
            ts.params,
            ts,
            env,
            states,
            1.0,
            jax.random.split(key, num=minibatch_size),
        )

        grads = jax.tree_map(lambda x: x.mean(axis=0), grads)
        ts.apply_gradients(grads=grads)

        jax.tree_map(
            lambda x, y: x.__setitem__(idx, y), replay_buffer, next_states
        )
        for j in range(minibatch_size):
            if dones[j]:
                print("bughdsh")
                rng, key = jax.random.split(rng)
                jax.tree_map(
                    lambda x, y: x.__setitem__(idx[j], y),
                    replay_buffer,
                    fresh_state(env, key),
                )

        print(round(minibatch_size / (time.perf_counter() - start_time), 2))


if __name__ == "__main__":
    main()
