import chex
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array as JaxArray
from jux.actions import FactoryAction, JuxAction, UnitAction, UnitActionType
from jux.map.position import Direction
from jux.state import State as JuxState
from jux.unit_cargo import ResourceType
from jux.utils import INT8_MAX, INT16_MAX
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
def position_scores(state: JuxState, action_scores: JaxArray):
    """
    Q predictions for each action -> max Q prediction for each resulting position

    action_scores (48, 48, 7)
        0: idle
        1: move up
        2: move right
        3: move down
        4: move left
        5: dig
        6: dropoff
    """

    chex.assert_shape(action_scores, (48, 48, 7))
    unit_x, unit_y = state.units.pos.x, state.units.pos.y  # (2, U)
    ret = jnp.full((2, 1000, 5), fill_value=np.nan)
    dig_best = jnp.zeros((2, 1000), dtype=jnp.bool_)
    dropoff_best = jnp.zeros((2, 1000), dtype=jnp.bool_)

    dig_mask = get_dig_mask(state)
    for team in range(2):
        scores = action_scores.at[unit_x[team], unit_y[team]].get(
            mode="fill", fill_value=np.nan
        )

        dig_best = dig_best.at[team].set(
            dig_mask[team] & (scores[..., 5] > scores[..., 0])
        )
        scores = scores.at[..., 0].set(
            jnp.where(
                dig_best[team],
                scores[..., 5],
                scores[..., 0],
            )
        )

        factory_mask = state.board.factory_occupancy_map.at[
            unit_x[team], unit_y[team]
        ].get(mode="fill", fill_value=INT8_MAX)
        chex.assert_shape(factory_mask, (1000,))

        dropoff_best = dropoff_best.at[team].set(
            (factory_mask < INT8_MAX) & (scores[..., 6] > scores[..., 0])
        )
        scores = scores.at[..., 0].set(
            jnp.where(
                dropoff_best[team],
                scores[..., 6],
                scores[..., 0],
            )
        )
        chex.assert_shape(scores, (1000, 7))

        ret = ret.at[team].set(scores[..., :5])
    return ret, dig_best, dropoff_best


directions = np.array(
    [
        [0, 0],  # stay
        [0, -1],  # up
        [1, 0],  # right
        [0, 1],  # down
        [-1, 0],  # left
    ]
)


def maximize_actions_callback(state: JuxState, resulting_pos_scores: np.ndarray):
    try:
        ret = np.zeros((2, resulting_pos_scores.shape[1]), dtype=np.int8)
        for team in range(2):
            n = state.n_units[team]
            if n == 0:
                continue
            unit_pos = np.asarray(state.units.pos.pos[team, :n])  # N, 2
            assert unit_pos.shape == (n, 2)

            scores = resulting_pos_scores[team, :n]  # N, 5

            c = scores.ravel()
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

            i = np.arange(n).repeat(5)[in_bounds.ravel()]
            j = np.arange(5 * n)[in_bounds.ravel()]
            a_eq = coo_array((np.ones(i.shape), (i, j)), shape=(n, 5 * n))
            b_eq = np.ones(n)

            result = linprog(c, a_lt, b_lt, a_eq, b_eq, method="highs-ds")
            coeffs = result.x.reshape(scores.shape)
            ret[team, np.arange(n)] = np.argmax(coeffs, axis=1)
        return ret
    except AttributeError:
        breakpoint()


@jax.jit
def step_best(state: JuxState, action_scores: JaxArray):
    chex.assert_shape(action_scores, (48, 48, 7))

    scores, dig_best, dropoff_best = position_scores(state, action_scores)
    selected_idx = jax.pure_callback(
        maximize_actions_callback,
        jax.ShapeDtypeStruct(shape=(2, 1000), dtype=jnp.int8),
        state,
        scores,
        vectorized=False,
    )  # (2, MAX_N_UNITS)

    # heuristic factory behavior: simply build light robots when possible
    factory_action = jnp.where(
        (state.factories.cargo.metal >= 10)
        & (state.factories.power >= 100)
        & (
            state.board.units_map[state.factories.pos.x, state.factories.pos.y]
            == INT16_MAX
        ),
        int(FactoryAction.BUILD_LIGHT),
        int(FactoryAction.DO_NOTHING),
    ).astype(jnp.int8)

    # reconstruct non-movement actions
    action_types = jnp.zeros((2, 1000, 20), dtype=jnp.int8)
    action_types = action_types.at[..., 0].set(
        jnp.where(dig_best, int(UnitActionType.DIG), action_types[..., 0])
    )
    action_types = action_types.at[..., 0].set(
        jnp.where(dropoff_best, int(UnitActionType.TRANSFER), action_types[..., 0])
    )
    action_types = action_types.at[..., 0].set(
        jnp.where(
            (1 <= selected_idx) & (selected_idx <= 4),
            int(UnitActionType.MOVE),
            action_types[..., 0],
        )
    )
    direction = jnp.zeros((2, 1000, 20), dtype=jnp.int8)
    direction = direction.at[..., 0].set(
        jnp.where(selected_idx == 1, int(Direction.UP), direction[..., 0])
    )
    direction = direction.at[..., 0].set(
        jnp.where(selected_idx == 2, int(Direction.RIGHT), direction[..., 0])
    )
    direction = direction.at[..., 0].set(
        jnp.where(selected_idx == 3, int(Direction.DOWN), direction[..., 0])
    )
    direction = direction.at[..., 0].set(
        jnp.where(selected_idx == 4, int(Direction.LEFT), direction[..., 0])
    )

    resource_type = jnp.full((2, 1000, 20), fill_value=ResourceType.ice, dtype=jnp.int8)
    amount = jnp.zeros((2, 1000, 20), dtype=jnp.int16)
    amount = amount.at[..., 0].set(state.units.cargo.ice.astype(jnp.int16))
    repeat = jnp.ones((2, 1000, 20), dtype=jnp.int16)
    n = jnp.ones((2, 1000, 20), dtype=jnp.int16)

    unit_action_queue = UnitAction(
        action_type=action_types,
        direction=direction,
        resource_type=resource_type,
        amount=amount,
        repeat=repeat,
        n=n,
    )

    unit_action_queue_count = jnp.ones((2, 1000), dtype=jnp.int8)
    unit_action_queue_update = jnp.zeros((2, 1000), dtype=jnp.bool_)
    for team in range(2):
        unit_action_queue_update = unit_action_queue_update.at[team].set(
            jnp.where(jnp.arange(1000) < state.n_units[team], True, False)
        )

    action = JuxAction(
        factory_action=factory_action,
        unit_action_queue=unit_action_queue,
        unit_action_queue_count=unit_action_queue_count,
        unit_action_queue_update=unit_action_queue_update,
    )
    return state._step_late_game(action)
