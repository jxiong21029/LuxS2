import json

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jux.env import JuxEnv
from jux.state import State as JuxState
from luxai_runner.utils import to_json

from action_handling import get_best_action
from observation import get_obs
from train import QNet, TrainState, train_step

# rng, key = jax.random.split(rng)


def choose_factory_spawn(rng, state: JuxState):
    spawns_mask = state.board.valid_spawns_mask
    selected = jax.random.choice(
        rng, 48 * 48, p=spawns_mask.ravel() / spawns_mask.sum()
    )
    coords = jnp.array([selected // 48, selected % 48], dtype=jnp.int8)
    return jnp.stack([coords, coords], axis=0)


def main():
    env = JuxEnv()
    rng = jax.random.PRNGKey(42)

    state = env.reset(0)

    replay = {
        "observations": [state.to_lux().get_compressed_obs()],
        "actions": [{}],
    }

    state, _ = env.step_bid(state, jnp.zeros(2), jnp.arange(2))

    for i in range(2 * state.board.factories_per_team):
        rng, key = jax.random.split(rng)
        action = choose_factory_spawn(key, state)
        next_state, _ = env.step_factory_placement(
            state, action, jnp.array([150, 150]), jnp.array([150, 150])
        )

        replay["observations"].append(
            next_state.to_lux().get_change_obs(state.to_lux().get_compressed_obs())
        )
        replay["actions"].append(
            {
                "player_0": {
                    "spawn": np.asarray(action),
                    "water": 150,
                    "metal": 150,
                },
                "player_1": {
                    "spawn": np.asarray(action),
                    "water": 150,
                    "metal": 150,
                },
            }
        )

        state = next_state

    # rng, key = jax.random.split(rng)
    # model = QNet()
    # params = model.init(key, get_obs(state))["params"]
    #
    # ts = TrainState.create(
    #     apply_fn=model.apply,
    #     params=params,
    #     slow_params=params,
    #     tx=optax.adam(1e-3),
    # )

    done = False
    while not done:
        print(state.real_env_steps)
        rng, key = jax.random.split(rng)
        action, _ = get_best_action(
            env, state, jax.random.gumbel(key, (48, 48, 7))
        )
        next_state, (_, _, dones, _) = env.step_late_game(state, action)

        done = dones[0]

        replay["observations"].append(
            next_state.to_lux().get_change_obs(state.to_lux().get_compressed_obs())
        )
        replay["actions"].append(action.to_lux(state))

        state = next_state

    with open("eval_replay.json", "w") as f:
        json.dump(to_json(replay), f)


if __name__ == "__main__":
    main()
