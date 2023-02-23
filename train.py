import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from jux.actions import (
    ActionQueue,
    FactoryAction,
    JuxAction,
    UnitAction,
    UnitActionType,
)
from jux.config import EnvConfig, JuxBufferConfig
from jux.env import JuxEnv
from jux.state import State as JuxState
from jux.utils import INT8_MAX, INT16_MAX, load_replay


@jax.jit
def state_to_obs(state: JuxState):
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
        ret = ret.at[unit_x, unit_y, 5 + team].set(cargo_ice / 100, mode="drop")

        # factory: occupancy, [center, cargo: ice, ore, water, metal; lichen connected]
        # separate obs for ally and enemy
        factory_occp_x = state.factories.occupancy.x[team]
        factory_occp_y = state.factories.occupancy.y[team]
        ret = ret.at[factory_occp_x, factory_occp_y, 7 + team].set(1, mode="drop")

    return ret


# handles actions for both teams
@jax.jit
def action_arr_to_jux(
    state: JuxState, action_arr: jax.Array, env_cfg, buf_cfg
) -> JuxAction:
    batch_shape = (
        2,
        buf_cfg.MAX_N_UNITS,
        env_cfg.UNIT_ACTION_QUEUE_SIZE,
    )

    # heuristic factory behavior:
    # if have 10 metal / 100 power: build light
    # else: do nothing
    factory_action = jnp.where(
        (state.factories.cargo.metal >= 10) & (state.factories.power >= 100),
        FactoryAction.BUILD_LIGHT,
        FactoryAction.DO_NOTHING,
    )

    # Six unit actions: idle + 4 directions + interact
    # action_arr is an int8[HW]

    unit_mask = state.board.units_map != INT16_MAX

    unit_idxs = state.unit_id2idx.at[state.board.units_map].get(
        mode="fill", fill_value=INT16_MAX
    )

    interact_mask = unit_mask & (action_arr == 0)
    factory_interact_mask = interact_mask & (
        state.board.factory_occupancy_map != INT8_MAX
    )
    dig_mask = interact_mask & (
        state.board.ice | state.board.ore | (state.board.lichen > 0)
    )

    # [2, MAX_N_UNITS,  UNIT_ACTION_QUEUE_SIZE]
    action_types = jnp.full(batch_shape, fill_value=UnitActionType.DO_NOTHING)
    action_types = action_types.at[
        jnp.where(
            (action_arr == 5)
            & (
                state.board.ice
                | state.board.ore
                | (state.board.rubble > 0)
                | (state.board.lichen > 0)
            ),
            unit_idxs,
            INT16_MAX,
        )
    ].set(UnitActionType.DIG, mode="drop")

    unit_action_queue = UnitAction(
        action_type=None,
        direction=None,
        resource_type=None,
        amount=None,
        repeat=None,
        n=None,
    )

    JuxAction()

    # ret = JuxAction(
    #     factory_action=,
    # )


class QNet(nn.Module):
    # returns predicted Q-values (for each agent)
    # current plan is do things VDN-style and just sum the local Q-values
    @nn.compact
    def __call__(self, obs):
        x = nn.Conv(32, kernel_size=(7, 7))(obs)
        x = nn.relu(x)
        x = nn.Conv(7, kernel_size=(1, 1))(x)
        return x


def main():
    lux_env, lux_actions = load_replay(
        f"https://www.kaggleusercontent.com/episodes/{46215591}.json"
    )
    jux_env, state = JuxEnv.from_lux(lux_env)
    obs = state_to_obs(state)
    action_arr_to_jux(state, jnp.zeros(2))


if __name__ == "__main__":
    main()
