from functools import partial
from typing import NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
from jux.config import EnvConfig, JuxBufferConfig
from jux.env import JuxEnv
from jux.state import State as JuxState
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
    obs: jax.Array  # BATCH of obs, o_t, for t in 1, ..., T
    actions: jax.Array  # BATCH of actions, a_t
    rewards: jax.Array  # BATCH of rewards received r_{t+1} from taking a_t
    terminal: jax.Array  # BATCH of terminal flags: d_{t+1}
    # d_{t+1} = 1 if r_{t+1} is the last reward received in the trajectory

    end_obs: jax.Array  # obs o_{T+1} AFTER the last action for value bootstrap


@jax.jit
def td_loss(params, slow_params, rollout, gamma=0.99, lam=0.9):
    q_pred = model.apply(params, rollout.obs)
    q_pred_slow = model.apply(slow_params, rollout.obs)

    returns = lambda_returns()


@partial(jax.jit, static_argnames=["rollout_steps"])
def collect_rollout(params, start_state: JuxState, rollout_steps=16):
    rollout_states: JuxState = jax.tree_util.tree_map(
        lambda x: jnp.zeros((rollout_steps,) + x.shape, x.dtype), start_state
    )
    curr_state = start_state
    for i in range(rollout_steps):
        rollout_states = jax.tree_util.tree_map(
            lambda x, y: x.at[i].set(y), rollout_states, curr_state
        )

        utility = model.apply(params, get_obs(curr_state))
        curr_state, actions = step_best(curr_state, utility)

        # TODO: reset terminated environments (...or don't)
