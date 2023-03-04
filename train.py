from functools import partial
from typing import NamedTuple

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
from jux.env import JuxEnv
from jux.state import State as JuxState
from jux.utils import INT16_MAX
from rlax import lambda_returns

from action_handling import step_best
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
    def __init__(
        self, env: JuxEnv, model, gamma=0.99, lam=0.9, rollout_length=16
    ):
        self.env = env
        self.gamma = gamma
        self.model = model
        self.lam = lam
        self.rollout_length = rollout_length

    @partial(jax.jit, static_argnums=0)
    def td_loss(self, params, slow_params, rollout: Rollout):
        all_obs = jax.vmap(get_obs)(rollout.states)
        # B, H, W, C
        utility = self.model.apply(params, all_obs)
        utility_slow = self.model.apply(slow_params, all_obs)

        # TODO: rollout actions shape should be (batch dim, MAX_N_UNITS)
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
            state.board.factories_per_team.astype(jnp.int32),
            step_factory_placement,
            (key, state),
        )
        return state

    @partial(jax.jit, static_argnums=0)
    def collect_rollout(self, rng, params, start_state: JuxState):
        rollout_states: JuxState = jax.tree_util.tree_map(
            lambda x: jnp.zeros((self.rollout_length,) + x.shape, x.dtype),
            start_state,
        )
        rollout_actions = jnp.zeros(
            (self.rollout_length, 2, self.env.buf_cfg.MAX_N_UNITS)
        )
        rollout_rewards = jnp.zeros(self.rollout_length)
        rollout_terminal = jnp.zeros(self.rollout_length, dtype=jnp.bool_)

        def update(i, args):
            (
                rng_,
                state_,
                rollout_states_,
                rollout_actions_,
                rollout_rewards_,
                rollout_terminal_,
            ) = args
            rollout_states_ = jax.tree_util.tree_map(
                lambda x, y: x.at[i].set(y), rollout_states_, state_
            )

            utility = self.model.apply(params, get_obs(state_))
            state_, actions, reward, done = step_best(state_, utility)

            rollout_actions_ = rollout_actions_.at[i].set(actions)
            rollout_rewards_ = rollout_rewards_.at[i].set(reward)
            rollout_terminal_ = rollout_terminal_.at[i].set(done)

            rng_, key_ = jax.random.split(rng_)
            fresh_state = self.fresh_state(key_)
            state_ = jax.tree_map(
                lambda x, y: jnp.where(done, x, y), fresh_state, state_
            )
            return (
                rng_,
                state_,
                rollout_states_,
                rollout_actions_,
                rollout_rewards_,
                rollout_terminal_,
            )

        state = start_state
        (
            _, _,
            rollout_states,
            rollout_actions,
            rollout_rewards,
            rollout_terminal,
        ) = jax.lax.fori_loop(
            0,
            self.rollout_length,
            update,
            (
                rng,
                state,
                rollout_states,
                rollout_actions,
                rollout_rewards,
                rollout_terminal,
            ),
        )

        return (
            rollout_states,
            rollout_actions,
            rollout_rewards,
            rollout_terminal,
        )


def main():
    rng = jax.random.PRNGKey(42)
    rng, key = jax.random.split(rng)

    dummy_state = JuxEnv().reset(0)
    model = QNet()
    params = model.init(key, get_obs(dummy_state))

    trainer = Trainer(JuxEnv(), model)
    rollout = trainer.collect_rollout(rng, params, start_state=dummy_state)


if __name__ == "__main__":
    main()
