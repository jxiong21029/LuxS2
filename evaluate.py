import json

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from jux.config import JuxBufferConfig
from jux.env import JuxEnv
from jux.state import State as JuxState
from luxai_runner.utils import to_json

from dynamics import get_best_action
from observation import get_obs
from train import QNet


def choose_factory_spawn(rng, state: JuxState):
    spawns_mask = state.board.valid_spawns_mask
    selected = jax.random.choice(
        rng, 48 * 48, p=spawns_mask.ravel() / spawns_mask.sum()
    )
    coords = jnp.array([selected // 48, selected % 48], dtype=jnp.int8)
    return jnp.stack([coords, coords], axis=0)


def play(rng, model: nn.Module, params0, params1=None):
    if params1 is None:
        params1 = params0

    env = JuxEnv(buf_cfg=JuxBufferConfig(MAX_N_UNITS=250))

    rng, key = jax.random.split(rng)
    state = env.reset(jax.random.randint(key, (), 0, 2**31 - 1).item())

    replay = {
        "observations": [state.to_lux().get_compressed_obs()],
        "actions": [{}],
    }

    next_state, _ = env.step_bid(state, jnp.zeros(2), jnp.arange(2))
    replay["observations"].append(
        next_state.to_lux().get_change_obs(state.to_lux().get_compressed_obs())
    )
    replay["actions"].append(
        {
            "player_0": {"faction": "AlphaStrike", "bid": 0},
            "player_1": {"faction": "MotherMars", "bid": 0},
        }
    )
    state = next_state

    for i in range(2 * state.board.factories_per_team):
        rng, key = jax.random.split(rng)
        action = choose_factory_spawn(key, state)
        next_state, _ = env.step_factory_placement(
            state, action, jnp.array([150, 150]), jnp.array([150, 150])
        )

        replay["observations"].append(
            next_state.to_lux().get_change_obs(
                state.to_lux().get_compressed_obs()
            )
        )
        replay["actions"].append(
            {
                "player_0": {
                    "spawn": np.asarray(action[0]),
                    "water": 150,
                    "metal": 150,
                },
                "player_1": {
                    "spawn": np.asarray(action[1]),
                    "water": 150,
                    "metal": 150,
                },
            }
        )
        state = next_state

    rng, key = jax.random.split(rng)

    @jax.jit
    def get_action_scores(state_: JuxState):
        obs = get_obs(state_)

        q0 = model.apply(params0, obs)
        q1 = model.apply(params1, obs)
        mask = (
            jnp.zeros((48, 48), dtype=jnp.bool_)
            .at[state.units.pos.x[1], state.units.pos.y[1]]
            .set(True, mode="drop")
        )
        return jnp.where(mask[..., None], q1, q0)

    jitted = jax.jit(get_best_action, static_argnums=0)
    done = False
    while not done:
        print(state.real_env_steps)
        rng, key = jax.random.split(rng)

        action, _ = jitted(env, state, get_action_scores(state))
        next_state, (_, _, dones, _) = env.step_late_game(state, action)
        done = dones[0]

        replay["observations"].append(
            next_state.to_lux().get_change_obs(
                state.to_lux().get_compressed_obs()
            )
        )
        replay["actions"].append(action.to_lux(state))

        state = next_state

    replay = jax.tree_map(
        lambda x: np.asarray(x) if isinstance(x, jax.Array) else x,
        replay,
    )
    with open("eval_replay.json", "w") as f:
        json.dump(to_json(replay), f)


def main():
    rng = jax.random.PRNGKey(42)

    rng, key1, key2 = jax.random.split(rng, num=3)
    model = QNet()
    params0 = model.init(key1, jnp.zeros((48, 48, 9)))
    params1 = model.init(key2, jnp.zeros((48, 48, 9)))

    play(rng, model, params0, params1)


if __name__ == "__main__":
    main()
