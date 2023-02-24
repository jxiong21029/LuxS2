import chex
import jax
import jax.numpy as jnp
from jux.env import State as JuxState
from jux.utils import INT8_MAX


def get_dig_mask(state: JuxState):
    # dig masking: can only dig resource OR rubble OR other team lichen
    lichen_team = state.factory_id2idx.at[state.board.lichen_strains, 0].get(
        mode="fill", fill_value=INT8_MAX
    )
    chex.assert_shape(lichen_team, (48, 48))
    chex.assert_type(lichen_team, jnp.int8)

    unit_team = jnp.full((48, 48), fill_value=INT8_MAX, dtype=jnp.int8)
    unit_team = unit_team.at[state.units.pos.x[0], state.units.pos.y[0]].set(
        0, mode="drop"
    )
    unit_team = unit_team.at[state.units.pos.x[1], state.units.pos.y[1]].set(
        1, mode="drop"
    )
    chex.assert_shape(unit_team, (48, 48))
    chex.assert_type(unit_team, jnp.int8)

    on_opponent_lichen = lichen_team == 1 - unit_team
    return jnp.where(
        state.board.lichen,
        on_opponent_lichen,
        state.board.ice | state.board.ore | state.board.rubble,
    )
