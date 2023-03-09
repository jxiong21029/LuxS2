from typing import Tuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array as JaxArray
from jux.actions import FactoryAction, JuxAction, UnitAction, UnitActionType
from jux.config import EnvConfig, JuxBufferConfig
from jux.env import JuxEnv
from jux.map.position import Direction
from jux.state import State as JuxState
from jux.unit import UnitType
from jux.unit_cargo import ResourceType
from jux.utils import INT8_MAX, INT16_MAX
from scipy.optimize import linprog
from scipy.sparse import coo_array


def get_dig_mask(env: JuxEnv, state: JuxState):
    unit_x, unit_y = state.units.pos.x, state.units.pos.y

    on_ice = state.board.ice.at[unit_x, unit_y].get(
        mode="fill", fill_value=False
    )
    on_ore = state.board.ore.at[unit_x, unit_y].get(
        mode="fill", fill_value=False
    )
    on_rubble = (
        state.board.rubble.at[unit_x, unit_y].get(mode="fill", fill_value=0)
        > 0
    )

    lichen_strains = state.board.lichen_strains.at[unit_x, unit_y].get(
        mode="fill", fill_value=INT8_MAX
    )
    lichen_team = state.factory_id2idx.at[lichen_strains, 0].get(
        mode="fill", fill_value=INT8_MAX
    )
    on_opponent_lichen = lichen_team == jnp.array([[1], [0]])

    power_req = jnp.where(
        state.units.unit_type == int(UnitType.LIGHT),
        env.env_cfg.ROBOTS[0].DIG_COST,
        env.env_cfg.ROBOTS[1].DIG_COST,
    )
    # power_req = power_req + jnp.where(state.units.next_action())
    chex.assert_shape(
        state.units.action_queue.data.action_type,
        (2, env.buf_cfg.MAX_N_UNITS, 20),
    )

    action_queue_rewrite_costs = jnp.where(
        state.units.unit_type == int(UnitType.LIGHT),
        env.env_cfg.ROBOTS[0].ACTION_QUEUE_POWER_COST,
        env.env_cfg.ROBOTS[1].ACTION_QUEUE_POWER_COST,
    )

    power_req = power_req + jnp.where(
        state.units.action_queue.data.action_type[..., 0]
        == int(UnitActionType.DIG),
        0,
        action_queue_rewrite_costs,
    )
    chex.assert_shape(power_req, (2, env.buf_cfg.MAX_N_UNITS))

    has_enough_power = state.units.power >= power_req

    chex.assert_shape(unit_x, (2, env.buf_cfg.MAX_N_UNITS))
    chex.assert_shape(unit_y, (2, env.buf_cfg.MAX_N_UNITS))
    chex.assert_shape(on_ice, (2, env.buf_cfg.MAX_N_UNITS))
    chex.assert_shape(on_ore, (2, env.buf_cfg.MAX_N_UNITS))
    chex.assert_shape(on_rubble, (2, env.buf_cfg.MAX_N_UNITS))
    chex.assert_shape(on_opponent_lichen, (2, env.buf_cfg.MAX_N_UNITS))
    chex.assert_shape(has_enough_power, (2, env.buf_cfg.MAX_N_UNITS))

    return has_enough_power & (
        on_ice | on_ore | on_rubble | on_opponent_lichen
    )


def get_dropoff_mask(env: JuxEnv, state: JuxState):
    on_factory = (
        state.board.factory_occupancy_map.at[
            state.units.pos.x, state.units.pos.y
        ].get(mode="fill", fill_value=INT8_MAX)
        < INT8_MAX
    )

    action_queue_rewrite_costs = jnp.where(
        state.units.unit_type == int(UnitType.LIGHT),
        env.env_cfg.ROBOTS[0].ACTION_QUEUE_POWER_COST,
        env.env_cfg.ROBOTS[1].ACTION_QUEUE_POWER_COST,
    )

    power_req = jnp.where(
        state.units.action_queue.data.action_type[..., 0]
        == int(UnitActionType.TRANSFER),
        0,
        action_queue_rewrite_costs,
    )
    chex.assert_shape(power_req, (2, env.buf_cfg.MAX_N_UNITS))
    has_enough_power = state.units.power >= power_req

    chex.assert_shape(on_factory, (2, env.buf_cfg.MAX_N_UNITS))
    chex.assert_shape(has_enough_power, (2, env.buf_cfg.MAX_N_UNITS))

    return on_factory & has_enough_power


def position_scores(env: JuxEnv, state: JuxState, action_scores: JaxArray):
    """
    Q predictions for each action -> max Q for each resulting position

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
    unit_x, unit_y = state.units.pos.x, state.units.pos.y
    ret = jnp.full((2, env.buf_cfg.MAX_N_UNITS, 5), fill_value=np.nan)

    dig_best = jnp.zeros((2, env.buf_cfg.MAX_N_UNITS), dtype=jnp.bool_)
    dropoff_best = jnp.zeros((2, env.buf_cfg.MAX_N_UNITS), dtype=jnp.bool_)
    # boolean arrays where dig/dropoff are the best STATIONARY moves

    chex.assert_shape(unit_x, (2, env.buf_cfg.MAX_N_UNITS))
    chex.assert_shape(unit_y, (2, env.buf_cfg.MAX_N_UNITS))

    dig_mask = get_dig_mask(env, state)
    dropoff_mask = get_dropoff_mask(env, state)

    for team in range(2):
        tt = 1 if team == 0 else -1
        scores = action_scores.at[unit_x[team], unit_y[team]].get(
            mode="fill", fill_value=np.nan
        )

        dig_best = dig_best.at[team].set(
            dig_mask[team] & (tt * scores[..., 5] > tt * scores[..., 0])
        )
        scores = scores.at[..., 0].set(
            jnp.where(
                dig_best[team],
                scores[..., 5],
                scores[..., 0],
            )
        )
        dropoff_best = dropoff_best.at[team].set(
            dropoff_mask[team] & (tt * scores[..., 6] > tt * scores[..., 0])
        )
        scores = scores.at[..., 0].set(
            jnp.where(
                dropoff_best[team],
                scores[..., 6],
                scores[..., 0],
            )
        )
        chex.assert_shape(scores, (env.buf_cfg.MAX_N_UNITS, 7))

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


def maximize_actions_callback(
    env_cfg: EnvConfig,
    buf_cfg: JuxBufferConfig,
    state: JuxState,
    resulting_pos_scores: np.ndarray,
):
    ret = np.zeros((2, buf_cfg.MAX_N_UNITS), dtype=np.int8)
    for team in range(2):
        n = state.n_units[team]
        if n == 0:
            continue
        unit_pos = np.asarray(
            state.units.pos.pos[team, :n], dtype=np.int64
        )  # N, 2
        assert unit_pos.shape == (n, 2)

        scores = resulting_pos_scores[team, :n]  # N, 5

        # negative because the LP solver minimizes, while we want maximum score
        c = -scores.ravel()
        if team == 1:  # unless we want minimum score
            c *= -1

        destinations = unit_pos[:, None] + directions  # N, 5, 2
        in_bounds = (
            (0 <= destinations[..., 0])
            & (destinations[..., 0] < 48)
            & (0 <= destinations[..., 1])
            & (destinations[..., 1] < 48)
        ).ravel()  # (N*5,) boolean arr -- for each move, bool for in bounds

        flat_idx = (
            48 * destinations[..., 0] + destinations[..., 1]
        ).ravel()  # (N*5,)
        # for each move, int in [0, 48^2) -- flattened index of destination

        a_lt = coo_array(
            (
                np.ones(in_bounds.sum()),
                (flat_idx[in_bounds], np.arange(5 * n)[in_bounds]),
            ),
            shape=(48 * 48, 5 * n),
        )
        b_lt = np.ones(48 * 48)

        destination_rubble = np.zeros((n, 5), dtype=int)
        destination_rubble.ravel()[in_bounds] = state.board.rubble[
            destinations[..., 0].ravel()[in_bounds],
            destinations[..., 1].ravel()[in_bounds],
        ]
        chex.assert_shape(destination_rubble, (n, 5))
        unit_types = np.tile(state.units.unit_type[team, :n, None], reps=(5,))
        chex.assert_shape(unit_types, (n, 5))
        power_req = np.where(
            unit_types == int(UnitType.LIGHT),
            np.floor(
                env_cfg.ROBOTS[0].MOVE_COST
                + env_cfg.ROBOTS[0].RUBBLE_MOVEMENT_COST * destination_rubble
            ),
            np.floor(
                env_cfg.ROBOTS[1].MOVE_COST
                + env_cfg.ROBOTS[1].RUBBLE_MOVEMENT_COST * destination_rubble
            ),
        )
        chex.assert_shape(power_req, (n, 5))

        # we want to set rewrite costs to 0 if type==move and direction correct

        action_queue_rewrite_costs = np.where(
            state.units.unit_type[team, :n] == int(UnitType.LIGHT),
            env_cfg.ROBOTS[0].ACTION_QUEUE_POWER_COST,
            env_cfg.ROBOTS[1].ACTION_QUEUE_POWER_COST,
        )
        chex.assert_shape(action_queue_rewrite_costs, (n,))

        move_rewrite_costs = np.tile(
            action_queue_rewrite_costs[..., None], reps=(5,)
        )
        chex.assert_shape(move_rewrite_costs, (n, 5))

        correct_action_type = state.units.action_queue.data.action_type[
            team, :n, 0
        ] == int(UnitActionType.MOVE)
        chex.assert_shape(correct_action_type, (n,))

        move_directions = state.units.action_queue.data.direction[team, :n, 0]
        move_rewrite_costs[
            np.arange(n)[correct_action_type],
            move_directions[correct_action_type],
        ] = 0

        power_req += move_rewrite_costs
        not_enough_power = (
            np.asarray(state.units.power[team, :n, None]) < power_req
        )
        chex.assert_shape(not_enough_power, (n, 5))

        not_enough_power[..., 0] = False

        # n rows for the (each unit must select exactly one move) constraint
        # 1 final row for the (no out-of-bounds moves may be selected)
        num_oob = (~in_bounds).sum()
        a_eq = coo_array(
            (
                np.ones(5 * n + num_oob + not_enough_power.sum()),
                (
                    np.concatenate(
                        [
                            np.arange(n).repeat(5),
                            np.full(num_oob, fill_value=n),
                            np.full(not_enough_power.sum(), fill_value=n + 1),
                        ]
                    ),
                    np.concatenate(
                        [
                            np.arange(5 * n),
                            (~in_bounds).nonzero()[0],
                            not_enough_power.ravel().nonzero()[0],
                        ]
                    ),
                ),
            ),
            shape=(n + 2, 5 * n),
        )
        b_eq = np.ones(n + 2)
        b_eq[-2] = 0  # sum of OOB moves must be zero
        b_eq[-1] = 0  # sum of moves w/o enough power must be zero

        result = linprog(c, a_lt, b_lt, a_eq, b_eq, method="highs-ds")
        coeffs = result.x.reshape(scores.shape)
        ret[team, np.arange(n)] = np.argmax(coeffs, axis=1)
    return ret


def get_best_action(env, state, action_scores):
    chex.assert_shape(action_scores, (48, 48, 7))

    scores, dig_best, dropoff_best = position_scores(env, state, action_scores)
    selected_idx = jax.pure_callback(
        maximize_actions_callback,
        jax.ShapeDtypeStruct(
            shape=(2, env.buf_cfg.MAX_N_UNITS), dtype=jnp.int8
        ),
        env.env_cfg,
        env.buf_cfg,
        state,
        scores,
        vectorized=False,
    )  # (2, MAX_N_UNITS)

    # heuristic factory behavior: build light robots if possible
    # otherwise, water if it increases lichen and either
    #   (water > 150 or 2 * water > steps remaining)

    _, n_watered_locations, _ = state._cache_water_info(None)

    factory_action = jnp.where(
        (state.factories.cargo.metal >= env.env_cfg.ROBOTS[0].METAL_COST)
        & (state.factories.power >= env.env_cfg.ROBOTS[0].POWER_COST)
        & (
            state.board.units_map[state.factories.pos.x, state.factories.pos.y]
            == INT16_MAX
        ),
        int(FactoryAction.BUILD_LIGHT),
        jnp.where(
            (n_watered_locations > 0)
            & (
                (state.factories.cargo.water > 150)
                | (
                    2 * state.factories.cargo.water
                    > 1000 - state.real_env_steps
                )
            ),
            int(FactoryAction.WATER),
            int(FactoryAction.DO_NOTHING),
        ),
    ).astype(jnp.int8)

    # reconstruct non-movement actions
    action_types = jnp.zeros((2, env.buf_cfg.MAX_N_UNITS, 20), dtype=jnp.int8)
    action_types = action_types.at[..., 0].set(
        jnp.where(
            (selected_idx == 0) & dig_best,
            int(UnitActionType.DIG),
            action_types[..., 0],
        )
    )
    action_types = action_types.at[..., 0].set(
        jnp.where(
            (selected_idx == 0) & dropoff_best,
            int(UnitActionType.TRANSFER),
            action_types[..., 0],
        )
    )
    action_types = action_types.at[..., 0].set(
        jnp.where(
            (1 <= selected_idx) & (selected_idx <= 4),
            int(UnitActionType.MOVE),
            action_types[..., 0],
        )
    )
    direction = jnp.zeros((2, env.buf_cfg.MAX_N_UNITS, 20), dtype=jnp.int8)
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

    resource_type = jnp.full(
        (2, env.buf_cfg.MAX_N_UNITS, 20),
        fill_value=ResourceType.ice,
        dtype=jnp.int8,
    )
    amount = jnp.zeros((2, env.buf_cfg.MAX_N_UNITS, 20), dtype=jnp.int16)
    amount = amount.at[..., 0].set(state.units.cargo.ice.astype(jnp.int16))
    repeat = jnp.ones((2, env.buf_cfg.MAX_N_UNITS, 20), dtype=jnp.int16)
    n = jnp.ones((2, env.buf_cfg.MAX_N_UNITS, 20), dtype=jnp.int16)

    unit_action_queue = UnitAction(
        action_type=action_types,
        direction=direction,
        resource_type=resource_type,
        amount=amount,
        repeat=repeat,
        n=n,
    )

    unit_action_queue_count = jnp.ones(
        (2, env.buf_cfg.MAX_N_UNITS), dtype=jnp.int8
    )
    unit_action_queue_update = jnp.where(
        jnp.arange(env.buf_cfg.MAX_N_UNITS)[None, :] < state.n_units[:, None],
        True,
        False,
    )
    chex.assert_shape(unit_action_queue_update, (2, env.buf_cfg.MAX_N_UNITS))

    queue_front_equal = jax.tree_map(
        lambda a, b: a[..., 0] == b[..., 0],
        unit_action_queue,
        state.units.action_queue.data,
    )

    queue_front_equal = jax.tree_util.tree_reduce(
        jnp.logical_and, queue_front_equal
    )
    chex.assert_shape(queue_front_equal, (2, env.buf_cfg.MAX_N_UNITS))

    unit_action_queue_update = unit_action_queue_update & ~queue_front_equal

    jux_action = JuxAction(
        factory_action=factory_action,
        unit_action_queue=unit_action_queue,
        unit_action_queue_count=unit_action_queue_count,
        unit_action_queue_update=unit_action_queue_update,
    )

    action_arr = jnp.where((selected_idx == 0) & dig_best, 5, selected_idx)
    action_arr = jnp.where((selected_idx == 0) & dropoff_best, 6, action_arr)
    return jux_action, action_arr


def step_best(
    env: JuxEnv, state: JuxState, action_scores: JaxArray
) -> Tuple[JuxState, jnp.int8, jnp.float32, jnp.bool_]:
    action, action_arr = get_best_action(env, state, action_scores)

    new_state = state._step_late_game(action)

    # old_potential = state.n_units[0] - state.n_units[1]
    # new_potential = new_state.n_units[0] - new_state.n_units[1]
    # reward = new_potential - old_potential

    done = (new_state.n_factories == 0).any() | (
        new_state.real_env_steps >= 1000
    )
    reward = jnp.where(
        done,
        jnp.where(
            state.team_lichen_score()[0] == state.team_lichen_score()[1],
            0.5,
            (
                state.team_lichen_score()[0] > state.team_lichen_score()[1]
            ).astype(float),
        ),
        0.0,
    )

    return new_state, action_arr, reward, done
