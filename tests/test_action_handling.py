import chex
import jax
import jax.numpy as jnp
import pytest
from jux.actions import (
    JuxAction,
    bid_action_from_lux,
    factory_placement_action_from_lux,
)
from jux.config import EnvConfig, JuxBufferConfig
from jux.env import JuxEnv
from jux.state import State as JuxState
from jux.utils import load_replay

from action_handling import (
    get_dig_mask,
    maximize_actions_callback,
    position_scores,
    step_best,
)


@pytest.fixture
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

        utility = jax.random.normal(key, shape=(48, 48, 7))
        scores, dig_best, dropoff_best = jitted(env, sample_state, utility)
        chex.assert_shape(scores, (2, env.buf_cfg.MAX_N_UNITS, 5))
        chex.assert_shape(dig_best, (2, env.buf_cfg.MAX_N_UNITS))
        chex.assert_shape(dropoff_best, (2, env.buf_cfg.MAX_N_UNITS))

        for team in range(2):
            n = sample_state.n_units[team]
            assert not jnp.isnan(scores[team, :n]).any()
            assert jnp.isnan(scores[team, n:]).all()

            assert not jnp.any(dig_best[team, n:])
            assert not jnp.any(dropoff_best[team, n:])

            assert not jnp.any(dig_best & dropoff_best)


def test_action_maximization_smoke(sample_states):
    env = JuxEnv()
    jitted = jax.jit(
        chex.assert_max_traces(n=1)(position_scores), static_argnums=0
    )
    for sample_state in sample_states:
        scores, _, _ = jitted(env, sample_state, jnp.zeros((48, 48, 7)))
        maximize_actions_callback(
            EnvConfig(), JuxBufferConfig(), sample_state, scores
        )


def test_step_best_smoke(sample_states):
    env = JuxEnv()
    rng = jax.random.PRNGKey(42)
    jitted = jax.jit(chex.assert_max_traces(n=1)(step_best), static_argnums=0)
    for sample_state in sample_states:
        rng, key = jax.random.split(rng)

        scores = jax.random.normal(key, shape=(48, 48, 7))
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
            env, state, jax.random.normal(key, shape=(48, 48, 7))
        )
        assert next_state.real_env_steps == state.real_env_steps + 1

        if done:
            assert reward in (0, 0.5, 1)
            break
        else:
            assert reward == 0
            state = next_state
