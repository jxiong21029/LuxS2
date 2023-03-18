import json
from collections import defaultdict
from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from jux.actions import FactoryAction
from jux.config import JuxBufferConfig
from jux.env import JuxEnv
from jux.state import State as JuxState
from luxai_runner.utils import to_json

from dynamics import fresh_state, get_best_action, step_best
from observation import get_obs

jitted_get_best_action = jax.jit(get_best_action, static_argnums=0)
jitted_step_best = jax.jit(step_best, static_argnums=0)


@partial(jax.jit, static_argnums=0)
def get_action_scores(model, params0, params1, state: JuxState):
    obs = get_obs(state)

    q0 = model.apply({"params": params0}, obs)
    q1 = model.apply({"params": params1}, obs)
    mask = (
        jnp.zeros((48, 48), dtype=jnp.bool_)
        .at[state.units.pos.x[1], state.units.pos.y[1]]
        .set(True, mode="drop")
    )
    return jnp.where(mask[..., None], q1, q0), q0, q1


def evaluate(rng, n_episodes, logger, model, params0, params1=None):
    if params1 is None:
        params1 = params0
    rng, key = jax.random.split(rng)
    env = JuxEnv()

    for _ in range(n_episodes):
        metrics = defaultdict(float)
        state = fresh_state(env, key)

        done = False
        while not done:
            rng, key = jax.random.split(rng)

            combined_q, _, _ = get_action_scores(
                model, params0, params1, state
            )
            action, _ = jitted_get_best_action(env, state, combined_q)
            next_state, _, reward, done = jitted_step_best(
                env, state, combined_q
            )

            metrics["episode_reward"] += reward.item()
            metrics["spawned_light"] += (
                (
                    (action.factory_action == FactoryAction.BUILD_LIGHT)
                    & (
                        jnp.arange(env.buf_cfg.MAX_N_FACTORIES).reshape(1, -1)
                        < state.n_factories.reshape(-1, 1)
                    )
                )
                .sum()
                .item()
            )
            metrics["spawned_heavy"] += (
                (
                    (action.factory_action == FactoryAction.BUILD_HEAVY)
                    & (
                        jnp.arange(env.buf_cfg.MAX_N_FACTORIES).reshape(1, -1)
                        < state.n_factories.reshape(-1, 1)
                    )
                )
                .sum()
                .item()
            )
            # metrics["ice_mined"] += 0
            # metrics["ice_delivered"] += 0
            # metrics["ore_mined"] += 0
            # metrics["ore_delivered"] += 0

            state = next_state
        logger.push(dict(metrics))

    logger.step()


def choose_factory_spawn(rng, state: JuxState):
    spawns_mask = state.board.valid_spawns_mask
    selected = jax.random.choice(
        rng, 48 * 48, p=spawns_mask.ravel() / spawns_mask.sum()
    )
    coords = jnp.array([selected // 48, selected % 48], dtype=jnp.int8)
    return jnp.stack([coords, coords], axis=0)


def make_replay(rng, name, model: nn.Module, params0, params1=None):
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

    logit_data = []
    done = False
    while not done:
        rng, key = jax.random.split(rng)

        combined_q, q0, q1 = get_action_scores(model, params0, params1, state)
        action, _ = jitted_get_best_action(env, state, combined_q)
        next_state, (_, _, dones, _) = env.step_late_game(state, action)
        done = dones[0]

        logit_data.append(np.stack([q0, q1], axis=0))

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
    with open(f"{name}.json", "w") as f:
        json.dump(to_json(replay), f)
    np.savez_compressed(f"{name}_logit_data.npz", np.stack(logit_data, axis=0))
    print(f"created replay with {name=}")
