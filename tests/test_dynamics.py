import chex
import jax
import jax.numpy as jnp
import pytest
from jux.actions import (
    JuxAction,
    bid_action_from_lux,
    factory_placement_action_from_lux,
)
from jux.config import JuxBufferConfig
from jux.env import JuxEnv
from jux.map.position import direct2delta_xy
from jux.utils import load_replay

from dynamics import get_best_action, get_dig_mask, position_scores, step_best


@pytest.fixture(scope="session")
def sample_states():
    lux_env, lux_actions = load_replay("tests/sample_replay.json")
    jux_env, state = JuxEnv.from_lux(lux_env)

    lux_act = next(lux_actions)
    bid, faction = bid_action_from_lux(lux_act)
    state, _ = jux_env.step_bid(state, bid, faction)

    while state.real_env_steps < 0:
        lux_act = next(lux_actions)
        spawn, water, metal = factory_placement_action_from_lux(lux_act)

        state, _ = jux_env.step_factory_placement(state, spawn, water, metal)

    ret = [state]
    for i in range(800):
        lux_act = next(lux_actions)
        jux_act = JuxAction.from_lux(state, lux_act)
        state, _ = jux_env.step_late_game(state, jux_act)

        if i % 100 == 99:
            ret.append(state)

    assert len(ret) == 9

    return ret


def test_dig_mask_sufficient(sample_states):
    env = JuxEnv()
    jitted = jax.jit(
        chex.assert_max_traces(n=1)(get_dig_mask), static_argnums=0
    )
    for sample_state in sample_states:
        dig_mask = jitted(env, sample_state)

        assert dig_mask.shape == (2, 1000)

        for team in range(2):
            assert jnp.all(~dig_mask[sample_state.n_units[team] :])

            for i in range(sample_state.n_units[team]):
                x, y = (
                    sample_state.units.pos.x[team, i],
                    sample_state.units.pos.y[team, i],
                )
                can_dig = (
                    sample_state.board.ice[x, y]
                    | sample_state.board.ore[x, y]
                    | sample_state.board.rubble[x, y]
                ) & sample_state.units.power[team, i] >= (
                    6 if sample_state.units.unit_type[team, i] == 0 else 70
                )
                assert ~can_dig | dig_mask[team, i]


def test_resulting_pos_scores_valid(sample_states):
    env = JuxEnv()
    rng = jax.random.PRNGKey(42)
    jitted = jax.jit(
        chex.assert_max_traces(n=1)(position_scores), static_argnums=0
    )

    for sample_state in sample_states:
        rng, key = jax.random.split(rng)

        utility = jax.random.normal(key, shape=(48, 48, 8))
        scores, dig_best, pickup_best, dropoff_best = jitted(
            env, sample_state, utility
        )
        chex.assert_shape(scores, (2, env.buf_cfg.MAX_N_UNITS, 5))
        chex.assert_shape(dig_best, (2, env.buf_cfg.MAX_N_UNITS))
        chex.assert_shape(pickup_best, (2, env.buf_cfg.MAX_N_UNITS))
        chex.assert_shape(dropoff_best, (2, env.buf_cfg.MAX_N_UNITS))

        for team in range(2):
            n = sample_state.n_units[team]
            assert not jnp.isnan(scores[team, :n]).any()
            assert jnp.isnan(scores[team, n:]).all()

            assert not jnp.any(dig_best[team, n:])
            assert not jnp.any(pickup_best[team, n:])
            assert not jnp.any(dropoff_best[team, n:])

            assert not jnp.any(dig_best & pickup_best)
            assert not jnp.any(dig_best & dropoff_best)
            assert not jnp.any(pickup_best & dropoff_best)


# def test_action_maximization_smoke(sample_states):
#     env = JuxEnv()
#     jitted = jax.jit(
#         chex.assert_max_traces(n=1)(position_scores), static_argnums=0
#     )
#     for sample_state in sample_states:
#         scores, _, _ = jitted(env, sample_state, jnp.zeros((48, 48, 8)))
#         maximize_actions_callback(
#             EnvConfig(), JuxBufferConfig(), sample_state, scores,
#         )


def test_step_best_smoke(sample_states):
    env = JuxEnv()
    rng = jax.random.PRNGKey(42)
    jitted = jax.jit(chex.assert_max_traces(n=1)(step_best), static_argnums=0)
    for sample_state in sample_states:
        rng, key = jax.random.split(rng)

        scores = jax.random.normal(key, shape=(48, 48, 8))
        jitted(env, sample_state, scores)


def test_step_best_resource_rich_smoke(sample_states):
    env = JuxEnv()
    jitted = jax.jit(step_best, static_argnums=0)

    state = sample_states[-1]
    state = state._replace(
        board=state.board._replace(
            map=state.board.map._replace(
                rubble=jnp.zeros_like(state.board.map.rubble)
            )
        ),
        factories=state.factories._replace(
            cargo=state.factories.cargo._replace(
                stock=jnp.full_like(
                    state.factories.cargo.stock, fill_value=9999
                )
            )
        ),
    )
    done = False

    rng = jax.random.PRNGKey(42)
    while not done:
        rng, key = jax.random.split(rng)
        next_state, actions, reward, done = jitted(
            env, state, jax.random.normal(key, shape=(48, 48, 8))
        )
        assert next_state.real_env_steps == state.real_env_steps + 1

        if not done:
            # TODO restore
            # assert reward == 0
            state = next_state


def test_step_smaller_buffer():
    jitted = jax.jit(step_best, static_argnums=0)

    lux_env, lux_actions = load_replay("tests/sample_replay.json")
    jux_env, state = JuxEnv.from_lux(
        lux_env, buf_cfg=JuxBufferConfig(MAX_N_UNITS=250)
    )

    lux_act = next(lux_actions)
    bid, faction = bid_action_from_lux(lux_act)
    state, _ = jux_env.step_bid(state, bid, faction)

    assert state.units.pos.x.shape[1] == 250

    while state.real_env_steps < 0:
        lux_act = next(lux_actions)
        spawn, water, metal = factory_placement_action_from_lux(lux_act)

        state, _ = jux_env.step_factory_placement(state, spawn, water, metal)

    rng = jax.random.PRNGKey(42)
    for _ in range(10):
        rng, key = jax.random.split(rng)
        next_state, actions, reward, done = jitted(
            jux_env, state, jax.random.normal(key, shape=(48, 48, 8))
        )
        assert next_state.real_env_steps == state.real_env_steps + 1
        state = next_state


def test_actions_successful(sample_states):
    env = JuxEnv()
    rng = jax.random.PRNGKey(69420)
    jitted_get_best_action = jax.jit(get_best_action, static_argnums=0)
    jitted_step_best = jax.jit(step_best, static_argnums=0)

    n_collisions = 0
    for state in sample_states:
        rng, key = jax.random.split(rng)
        action_scores = jax.random.normal(key, (48, 48, 8))

        action, _ = jitted_get_best_action(env, state, action_scores)
        next_state, _, _, _ = jitted_step_best(env, state, action_scores)

        action: JuxAction

        curr_metal_value = (
            state.factories.cargo.metal.sum()
            + 10 * (state.units.pos.x < 127).sum()
        )
        next_metal_value = (
            next_state.factories.cargo.metal.sum()
            + 10 * (next_state.units.pos.x < 127).sum()
        )

        if next_metal_value < curr_metal_value:
            n_collisions += 1
            continue

        for team in range(2):
            for i in range(state.n_units[team]):
                if action.unit_action_queue.action_type[team, i, 0] == 0:
                    direction = action.unit_action_queue.direction[team, i][0]
                else:
                    direction = 0

                next_pos = (
                    state.units.pos.pos[team, i] + direct2delta_xy[direction]
                )
                assert jnp.array_equal(
                    next_state.units.pos.pos[team, i], next_pos
                )

    assert n_collisions < len(sample_states) // 2
