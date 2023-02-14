from jux.actions import (
    JuxAction,
    bid_action_from_lux,
    factory_placement_action_from_lux,
)
from jux.env import JuxEnv
from jux.utils import load_replay

from train import state_to_obs


def test_smokeless_obs():
    lux_env, lux_actions = load_replay(
        f"https://www.kaggleusercontent.com/episodes/{46215591}.json"
    )
    jux_env, state = JuxEnv.from_lux(lux_env)

    state_to_obs(state)

    # bidding phase
    lux_act = next(lux_actions)
    bid, faction = bid_action_from_lux(lux_act)

    state_to_obs(state)

    # placement phase
    state, _ = jux_env.step_bid(state, bid, faction)
    while state.real_env_steps < 0:
        lux_act = next(lux_actions)
        spawn, water, metal = factory_placement_action_from_lux(lux_act)

        state, _ = jux_env.step_factory_placement(
            state, spawn, water, metal
        )

    state_to_obs(state)

    # main phase
    for _ in range(25):
        lux_act = next(lux_actions)
        jux_act = JuxAction.from_lux(state, lux_act)
        state, _ = jux_env.step_late_game(
            state, jux_act
        )

    state_to_obs(state)
