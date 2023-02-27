import jax.numpy as jnp
import numpy as np
import pytest
from jux.actions import (
    JuxAction,
    bid_action_from_lux,
    factory_placement_action_from_lux,
)
from jux.env import JuxEnv
from jux.utils import load_replay

from env.action_handling import get_dig_mask, maximize_actions_callback, position_scores


@pytest.fixture
def sample_state():
    lux_env, lux_actions = load_replay("replay.json")
    jux_env, state = JuxEnv.from_lux(lux_env)

    lux_act = next(lux_actions)
    bid, faction = bid_action_from_lux(lux_act)
    state, _ = jux_env.step_bid(state, bid, faction)

    while state.real_env_steps < 0:
        lux_act = next(lux_actions)
        spawn, water, metal = factory_placement_action_from_lux(lux_act)

        state, _ = jux_env.step_factory_placement(state, spawn, water, metal)

    for _ in range(800):
        lux_act = next(lux_actions)
        jux_act = JuxAction.from_lux(state, lux_act)
        state, _ = jux_env.step_late_game(state, jux_act)

    return state


def test_dig_mask(sample_state):
    dig_mask = get_dig_mask(sample_state)
    assert dig_mask.shape == (2, 1000)

    for team in range(2):
        assert jnp.all(~dig_mask[sample_state.n_units[team] :])

        for i in range(sample_state.n_units[team]):
            x, y = sample_state.units.pos.x[team, i], sample_state.units.pos.y[team, i]
            can_dig = (
                sample_state.board.ice[x, y]
                | sample_state.board.ore[x, y]
                | sample_state.board.rubble[x, y]
            )
            assert ~can_dig | dig_mask[team, i]


def test_resulting_pos_scores(sample_state):
    scores, dig_best, dropoff_best = position_scores(
        sample_state, jnp.zeros((48, 48, 7))
    )
    for team in range(2):
        n = sample_state.n_units[team]
        assert not jnp.isnan(scores[team, :n]).any()
        assert jnp.isnan(scores[team, n:]).all()

        assert not jnp.any(dig_best[team, n:])
        assert not jnp.any(dropoff_best[team, n:])

        assert not jnp.any(dig_best & dropoff_best)
        assert jnp.all(~(dig_best | dropoff_best) | (jnp.argmax(scores, axis=2) == 0))


def test_action_maximization(sample_state):
    scores, _, _ = position_scores(sample_state, np.zeros((48, 48, 7)))
    maximize_actions_callback(sample_state, scores)
