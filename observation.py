import jax.numpy as jnp
from jux.env import State as JuxState


# TODO: observations should include the units' action queues (in this case, simply the
#   last action taken).
def get_obs(state: JuxState):
    # for now: heuristic behavior for bidding, factory placement, and factory actions
    # only reward function is for supplying factories with water
    # simplified obs reflects all info necessary for this

    ret = jnp.zeros((48, 48, 9))

    # global: [sin t/50, cos t/50, log t, log (1000 - t),]

    # tile: ice, x, y, [ore, rubble]
    ret = ret.at[..., 0].set(jnp.arange(48).reshape(1, 48) / 48)
    ret = ret.at[..., 1].set(jnp.arange(48).reshape(48, 1) / 48)
    ret = ret.at[..., 2].set(state.board.ice)

    for team in range(2):
        # unit: light, [heavy,] cargo: ice, [ore, water, metal]; [power]
        # separate obs for ally and enemy
        unit_x = state.units.pos.x[team]
        unit_y = state.units.pos.y[team]
        cargo_ice = state.units.cargo.ice[team]
        ret = ret.at[unit_x, unit_y, 3 + team].set(1, mode="drop")
        ret = ret.at[unit_x, unit_y, 5 + team].set(
            cargo_ice / 100, mode="drop"
        )

        # factory: occupancy, [center, cargo: ice, ore, water, metal; lichen connected]
        # separate obs for ally and enemy
        factory_occp_x = state.factories.occupancy.x[team]
        factory_occp_y = state.factories.occupancy.y[team]
        ret = ret.at[factory_occp_x, factory_occp_y, 7 + team].set(
            1, mode="drop"
        )

    return ret
