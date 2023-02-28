from jux.actions import (
    JuxAction,
    bid_action_from_lux,
    factory_placement_action_from_lux,
)
from jux.env import JuxEnv
from jux.utils import load_replay

from train import state_to_obs


def test_obs_smoke():
    lux_env, lux_actions = load_replay("tests/test_replay.json")
    jux_env, state = JuxEnv.from_lux(lux_env)
    state_to_obs(state)

    # bidding phase
    lux_act = next(lux_actions)
    bid, faction = bid_action_from_lux(lux_act)
    state, _ = jux_env.step_bid(state, bid, faction)
    state_to_obs(state)

    # placement phase
    while state.real_env_steps < 0:
        lux_act = next(lux_actions)
        spawn, water, metal = factory_placement_action_from_lux(lux_act)

        state, _ = jux_env.step_factory_placement(state, spawn, water, metal)
    state_to_obs(state)

    # main phase
    for _ in range(25):
        lux_act = next(lux_actions)
        jux_act = JuxAction.from_lux(state, lux_act)
        state, _ = jux_env.step_late_game(state, jux_act)
    state_to_obs(state)


def test_obs_correct():
    lux_env, lux_actions = load_replay("tests/test_replay.json")
    jux_env, state = JuxEnv.from_lux(lux_env)
    # bidding phase
    lux_act = next(lux_actions)
    bid, faction = bid_action_from_lux(lux_act)
    state, _ = jux_env.step_bid(state, bid, faction)
    # placement phase
    while state.real_env_steps < 0:
        lux_act = next(lux_actions)
        spawn, water, metal = factory_placement_action_from_lux(lux_act)

        state, _ = jux_env.step_factory_placement(state, spawn, water, metal)

    # main phase
    for _ in range(25):
        lux_act = next(lux_actions)
        jux_act = JuxAction.from_lux(state, lux_act)
        state, _ = jux_env.step_late_game(state, jux_act)

        obs = state_to_obs(state)

        for team in range(2):
            for i in range(state.n_units[team]):
                unit_exists_channel = 3 + team
                assert (
                    obs[
                        state.units.pos.x[team, i],
                        state.units.pos.y[team, i],
                        unit_exists_channel,
                    ]
                    == 1
                )
