from functools import partial
from typing import NamedTuple

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
from jux.config import EnvConfig, JuxBufferConfig
from jux.env import JuxEnv
from jux.state import State as JuxState
from jux.utils import INT16_MAX
from rlax import lambda_returns

from action_selection import step_best
from observation import get_obs


class QNet(nn.Module):
    # returns predicted Q-values (for each agent)
    # current plan is do things VDN-style and just sum the local Q-values
    @nn.compact
    def __call__(self, obs):
        x = nn.Conv(32, kernel_size=(7, 7))(obs)
        x = nn.relu(x)
        x = nn.Conv(7, kernel_size=(1, 1))(x)
        return x


model = QNet()


class Rollout(NamedTuple):
    states: JuxState  # BATCH of states, s_t, for s in 1, ..., T+1
    actions: jax.Array  # BATCH of actions, a_t
    rewards: jax.Array  # BATCH of rewards received r_{t+1} from taking a_t
    terminal: jax.Array  # BATCH of terminal flags: d_{t+1}
    # d_{t+1} = 1 if r_{t+1} is the last reward received in the trajectory

    # note that the state array is one longer than the rest


def choose_factory_spawn(rng, state: JuxState):
    spawns_mask = state.board.valid_spawns_mask
    selected = jax.random.choice(
        rng, 48 * 48, p=spawns_mask.ravel() / spawns_mask.sum()
    )
    coords = jnp.array([selected // 48, selected % 48], dtype=jnp.int8)
    return jnp.stack([coords, coords], axis=0)


class Trainer:
    def __init__(self, env, gamma=0.99, lam=0.9, rollout_length=16):
        self.env = env
        self.gamma = gamma
        self.lam = lam
        self.rollout_length = rollout_length

    @partial(jax.jit, static_argnums=0)
    def td_loss(self, params, slow_params, rollout: Rollout):
        all_obs = jax.vmap(get_obs)(rollout.states)
        # B, H, W, C
        utility = model.apply(params, all_obs)
        utility_slow = model.apply(slow_params, all_obs)

        units_mask = rollout.states.board.units_map < INT16_MAX
        selected_utility = jnp.take_along_axis(
            utility * units_mask, rollout.actions[..., None]
        )
        selected_utility_slow = jnp.take_along_axis(
            utility_slow * units_mask, rollout.actions[..., None]
        )

        chex.assert_shape(
            [selected_utility, selected_utility_slow],
            (rollout.terminal.shape[0], 48, 48, 1),
        )

        q_pred = selected_utility.sum(axis=(1, 2, 3))
        q_pred_slow = selected_utility_slow.sum(axis=(1, 2, 3))

        chex.assert_shape([q_pred, q_pred_slow], (rollout.terminal.shape[0],))

        returns = lambda_returns(
            rollout.rewards,
            self.gamma * (1 - rollout.terminal),
            q_pred_slow[1:],
            lambda_=self.lam,
        )
        return jnp.mean((q_pred - returns) ** 2)

    @partial(jax.jit, static_argnums=0)
    def fresh_state(self, rng):
        rng, key = jax.random.split(rng)
        seed = jax.random.randint(key, (), 0, 2**16)

        state = self.env.reset(seed)

        state, _ = self.env.step_bid(state, jnp.zeros(2), jnp.arange(2))

        def step_factory_placement(_i, args):
            rng_, s = args
            rng_, key_ = jax.random.split(rng_)
            action = choose_factory_spawn(key_, s)
            new_s, _ = self.env.step_factory_placement(
                state, action, jnp.array([150, 150]), jnp.array([150, 150])
            )
            return rng_, new_s

        rng, key = jax.random.split(rng)
        _, state = jax.lax.fori_loop(
            0,
            state.board.factories_per_team,
            step_factory_placement,
            (key, state),
        )

    @partial(jax.jit, static_argnums=0)
    def collect_rollout(
        self, rng, params, start_state: JuxState, action_noise
    ):
        rollout_states: JuxState = jax.tree_util.tree_map(
            lambda x: jnp.zeros((self.rollout_length,) + x.shape, x.dtype),
            start_state,
        )
        rollout_actions = jnp.zeros(())

        curr_state = start_state
        for i in range(self.rollout_length):
            rollout_states = jax.tree_util.tree_map(
                lambda x, y: x.at[i].set(y), rollout_states, curr_state
            )

            utility = model.apply(params, get_obs(curr_state))
            curr_state, actions = step_best(curr_state, utility)

            # TODO: reset terminated environments (...or don't)
            done = (curr_state.n_factories == 0).any() | (
                curr_state.real_env_steps >= 1000
            )

            def get_fresh_state(rng_):
                rng_, key_ = jax.random.split(rng_)
                return rng_, self.fresh_state(rng)

            rng, curr_state = jax.lax.cond(
                done, get_fresh_state, (lambda _rng: _rng, curr_state), rng
            )
        return Rollout(
            states=rollout_states,
            actions=
        )
