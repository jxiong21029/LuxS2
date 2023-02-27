import chex
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array as JaxArray
from jux.state import State as JuxState
from jux.utils import INT8_MAX
from scipy.optimize import linprog
from scipy.sparse import coo_array


@jax.jit
def get_dig_mask(state: JuxState):
    unit_x, unit_y = state.units.pos.x, state.units.pos.y  # (2, U)

    on_ice = state.board.ice.at[unit_x, unit_y].get(mode="fill", fill_value=False)
    on_ore = state.board.ore.at[unit_x, unit_y].get(mode="fill", fill_value=False)
    on_rubble = state.board.rubble.at[unit_x, unit_y].get(mode="fill", fill_value=0) > 0

    lichen_strains = state.board.lichen_strains.at[unit_x, unit_y].get(
        mode="fill", fill_value=INT8_MAX
    )
    lichen_team = state.factory_id2idx.at[lichen_strains, 0].get(
        mode="fill", fill_value=INT8_MAX
    )
    on_opponent_lichen = lichen_team == jnp.array([[1], [0]])

    chex.assert_shape(
        [unit_x, unit_y, on_ice, on_ore, on_rubble, on_opponent_lichen], (2, 1000)
    )

    dig_mask = on_ice | on_ore | on_rubble | on_opponent_lichen
    return dig_mask


@jax.jit
def resulting_position_scores(state: JuxState, action_scores: JaxArray):
    """
    Q predictions for each action -> max Q prediction for each resulting position

    action_scores: [MAX_N_AGENTS, 7]?
        0: idle
        1: move up
        2: move right
        3: move down
        4: move left
        5: dig
        6: dropoff
    """

    unit_x, unit_y = state.units.pos.x, state.units.pos.y  # (2, U)

    ret = jnp.full((2, 1000, 5), fill_value=np.nan)

    dig_mask = get_dig_mask(state)
    for team in range(2):
        scores = action_scores.at[unit_x[team], unit_y[team]].get(
            mode="fill", fill_value=np.nan
        )
        chex.assert_shape(scores, (1000, 7))

        scores = scores.at[..., 0].set(
            jnp.where(
                dig_mask[team] & (scores[..., 5] > scores[..., 0]),
                scores[..., 5],
                scores[..., 0],
            )
        )
        chex.assert_shape(scores, (1000, 7))

        factory_mask = state.board.factory_occupancy_map.at[
            unit_x[team], unit_y[team]
        ].get(mode="fill", fill_value=INT8_MAX)
        chex.assert_shape(factory_mask, (1000,))
        chex.assert_shape(scores, (1000, 7))

        scores = scores.at[..., 0].set(
            jnp.where(
                (factory_mask < INT8_MAX) & (scores[..., 6] > scores[..., 0]),
                scores[..., 6],
                scores[..., 0],
            )
        )
        chex.assert_shape(scores, (1000, 7))

        ret = ret.at[team].set(scores[..., :5])

    return ret


directions = np.array(
    [
        [0, 0],  # stay
        [0, -1],  # up
        [1, 0],  # right
        [0, 1],  # down
        [-1, 0],  # left
    ]
)


def best_joint_actions(state: JuxState, resulting_pos_scores: JaxArray):
    for team in range(2):
        n = state.n_units[team]
        unit_pos = np.asarray(state.units.pos.pos[team, :n])  # N, 2
        assert unit_pos.shape == (n, 2)

        scores = resulting_pos_scores[team, :n]  # N, 5

        c = np.asarray(scores).ravel()
        if team == 1:
            c = c * -1

        destinations = unit_pos[:, None] + directions  # N, 5, 2
        in_bounds = (
            (0 <= destinations[..., 0])
            & (destinations[..., 0] < 48)
            & (0 <= destinations[..., 1])
            & (destinations[..., 1] < 48)
        )  # N, 5 boolean arr
        flat_idx = 48 * destinations[..., 0] + destinations[..., 1]  # N, 5

        a_lt = coo_array(
            (
                np.ones(in_bounds.sum()),
                (flat_idx[in_bounds], np.arange(5 * n)[in_bounds.ravel()]),
            ),
            shape=(48 * 48, 5 * n),
        )
        b_lt = np.ones(48 * 48)

        i = np.arange(n).repeat(5)
        j = np.arange(5 * n)
        a_eq = coo_array((np.ones(5 * n), (i, j)))
        b_eq = np.ones(n)

        res = linprog(c, a_lt, b_lt, a_eq, b_eq, method="highs-ds")
        return res.x
