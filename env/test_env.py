import jax
import jax.numpy as jnp
import pytest
from jux.actions import (
    JuxAction,
    bid_action_from_lux,
    factory_placement_action_from_lux,
)
from jux.env import JuxEnv
from jux.utils import load_replay

from .action_masking import get_action_mask


@pytest.fixture
def sample_state():
    lux_env, lux_actions = load_replay("tests/test_replay.json")
    jux_env, state = JuxEnv.from_lux(lux_env)

    lux_act = next(lux_actions)
    bid, faction = bid_action_from_lux(lux_act)
    state, _ = jux_env.step_bid(state, bid, faction)

    while state.real_env_steps < 0:
        lux_act = next(lux_actions)
        spawn, water, metal = factory_placement_action_from_lux(lux_act)

        state, _ = jux_env.step_factory_placement(state, spawn, water, metal)

    for _ in range(25):
        lux_act = next(lux_actions)
        jux_act = JuxAction.from_lux(state, lux_act)
        state, _ = jux_env.step_late_game(state, jux_act)

    return state


def test_action_handling_compiles(sample_state):
    jitted = jax.jit(get_action_mask)
    jitted(sample_state)


def test_action_handling_correct(sample_state):
    action_mask = get_action_mask(sample_state)
    dig_mask = action_mask[..., 5]

    sufficient = (
        sample_state.board.ice
        | sample_state.board.ice
        | sample_state.board.rubble
        | sample_state.board.lichen
        > 0
    )

    # assert sufficient ==> dig_mask
    assert jnp.all(~sufficient | dig_mask)
