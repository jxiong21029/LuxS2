from functools import partial
from typing import Tuple

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
from jux.env import JuxEnv
from jux.state import State as JuxState
from jux.utils import INT16_MAX

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


def choose_factory_spawn(rng, state: JuxState):
    spawns_mask = state.board.valid_spawns_mask
    selected = jax.random.choice(
        rng, 48 * 48, p=spawns_mask.ravel() / spawns_mask.sum()
    )
    coords = jnp.array([selected // 48, selected % 48], dtype=jnp.int8)
    return jnp.stack([coords, coords], axis=0)


def predict_target_q(ts: TrainState, state: JuxState):
    obs = get_obs(state)
    utility = ts.apply_fn(ts.params, obs)
    _, actions, _, _ = step_best(state, utility)

    # double q-learning style: select actions with fast, set target w/ slow
    slow_utility = ts.apply_fn(ts.slow_params, obs)
    selected_utility = slow_utility.at[
        state.units.pos.x, state.units.pos.y, actions
    ].get(
        mode="fill", fill_value=0.0
    )  # (2, 1000)
    team_utility = selected_utility.sum(axis=-1)
    return team_utility[0] - team_utility[1]


@jax.jit
def step(ts: TrainState, state: JuxState) -> Tuple[TrainState, JuxState]:
    def td_loss(params):
        obs = get_obs(state)
        utility = ts.apply_fn(params, obs)
        next_state, actions, reward, done = step_best(state, utility)

        selected_utility = utility.at[
            state.units.pos.x, state.units.pos.y, actions
        ].get(
            mode="fill", fill_value=0.0
        )  # (2, 1000)
        team_utility = selected_utility.sum(axis=-1)
        q_pred = team_utility[0] - team_utility[1]
        q_target = jax.lax.cond(
            done, lambda: reward, lambda: predict_target_q(ts, next_state)
        )
        return (q_pred - q_target) ** 2, next_state, done

    grads, (next_state_, done_) = jax.grad(td_loss, has_aux=True)(ts.params)
    ts = ts.apply_gradients(grads=grads)
    return ts, next_state_, done_


def main():
    rng = jax.random.PRNGKey(42)
    rng, key = jax.random.split(rng)

    dummy_state = JuxEnv().reset(0)
    model = QNet()
    params = model.init(key, get_obs(dummy_state))


if __name__ == "__main__":
    main()
