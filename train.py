from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp
import rlax
from flax.training.train_state import TrainState
from jux.config import EnvConfig, JuxBufferConfig
from jux.env import JuxEnv
from jux.state import State as JuxState

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


@partial(jax.jit, static_argnames=["rollout_steps"])
def rollout_and_learn(ts: TrainState, start_state: JuxState, rollout_steps=16):
    states = [start_state]
    for i in range(rollout_steps):
        action_scores = None
        states.append(step_best(start_state, action_scores))
