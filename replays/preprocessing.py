from collections import defaultdict

import numpy as np
from luxai_s2.actions import (
    Action,
    DigAction,
    MoveAction,
    PickupAction,
    SelfDestructAction,
    TransferAction,
    validate_actions,
)
from luxai_s2.env import ActionsByType
from luxai_s2.state import State
from luxai_s2.unit import Unit, UnitType


def get_obs(state: State, elo, final_lichen_diff):
    assert state.real_env_steps >= 0

    # metadata: sin t/50, cos t/50, log t, log(1000 - t), elo (x2), lichen diff
    #   t = real env timestep
    #   (potentially: lichen totals, resource totals, bot counts)

    meta = np.array(
        [
            np.sin(state.real_env_steps / 50),
            np.cos(state.real_env_steps / 50),
            np.log1p(state.real_env_steps),
            np.log1p(1000 - state.real_env_steps),
            elo / 1000,
            np.log1p(np.abs(final_lichen_diff)),
        ]
    )

    # tile: ice, ore, rubble, x, y, lichen team (x2), lichen amount (x2)
    tiles = np.zeros((35, 48, 48), dtype=np.float32)
    tiles[0] = state.board.ice
    tiles[1] = state.board.ore
    tiles[2] = state.board.rubble / 100
    tiles[3] = (np.arange(48) / 48).reshape(48, 1) / 48
    tiles[4] = (np.arange(48) / 48).reshape(1, 48) / 48

    for i, player_id in enumerate(state.factories):
        for factory_id in state.factories[player_id]:
            mask = (
                state.board.lichen_strains
                == state.factories[player_id][factory_id].num_id
            )
            tiles[5 + i, mask] = 1
            tiles[7 + i] += state.board.lichen * mask / 100

    # unit (x2): light, heavy, ice, ore, water, metal, power
    for i, player_id in enumerate(state.units):
        for unit in state.units[player_id].values():
            if unit.unit_type == UnitType.LIGHT:
                tiles[9 + i, unit.pos.x, unit.pos.y] = 1
            else:
                tiles[11 + i, unit.pos.x, unit.pos.y] = 1

            tiles[13 + i, unit.pos.x, unit.pos.y] = np.log1p(unit.cargo.ice)
            tiles[15 + i, unit.pos.x, unit.pos.y] = np.log1p(unit.cargo.ore)
            tiles[17 + i, unit.pos.x, unit.pos.y] = np.log1p(unit.cargo.water)
            tiles[19 + i, unit.pos.x, unit.pos.y] = np.log1p(unit.cargo.metal)
            tiles[21 + i, unit.pos.x, unit.pos.y] = np.log1p(unit.power)

    # factory (x2): center, ice, ore, water, metal, power
    for i, player_id in enumerate(state.factories):
        for fact in state.factories[player_id].values():
            tiles[23 + i, fact.pos.x, fact.pos.y] = 1
            tiles[25 + i, fact.pos.x, fact.pos.y] = np.log1p(fact.cargo.ice)
            tiles[27 + i, fact.pos.x, fact.pos.y] = np.log1p(fact.cargo.ore)
            tiles[29 + i, fact.pos.x, fact.pos.y] = np.log1p(fact.cargo.water)
            tiles[31 + i, fact.pos.x, fact.pos.y] = np.log1p(fact.cargo.metal)
            tiles[33 + i, fact.pos.x, fact.pos.y] = np.log1p(fact.power)

    return meta, tiles


def get_actions(env_cfg, state: State, actions):
    # action type:
    # idle (1) (also encodes recharge)
    # move NESW (4)
    # transfer CNESW (5) * resource type (ice, ore, water, metal) * amount
    # pickup (1) * resource type (ice, ore, water, metal) * amount
    # dig (1)
    # self destruct (1)

    # store actions by type
    actions_by_type: ActionsByType = defaultdict(list)
    for agent in actions.keys():
        for unit in state.units[agent].values():
            unit_a: Action = unit.next_action()
            if unit_a is None:
                continue
            actions_by_type[unit_a.act_type].append((unit, unit_a))
        for factory in state.factories[agent].values():
            if len(factory.action_queue) > 0:
                unit_a: Action = factory.action_queue.pop(0)
                actions_by_type[unit_a.act_type].append((factory, unit_a))

    # validate all actions against current state, discard invalid actions
    actions_by_type = validate_actions(env_cfg, state, actions_by_type)

    action_types = np.zeros((48, 48), dtype=np.int8)
    action_resources = np.zeros((48, 48), dtype=np.int32)
    action_amounts = np.zeros((48, 48), dtype=np.int32)

    seen_unit_ids = set()

    for v in actions_by_type.values():
        for unit, action in v:
            unit: Unit
            seen_unit_ids.add(unit.unit_id)
            x, y = unit.pos.x, unit.pos.y

            if isinstance(action, MoveAction):
                assert 0 <= action.move_dir < 5
                action_types[x, y] = action.move_dir

            elif isinstance(action, TransferAction):
                assert 0 <= action.transfer_dir < 5
                action_types[x, y] = 5 + action.transfer_dir
                action_resources[x, y] = action.resource
                action_amounts[x, y] = action.transfer_amount

            elif isinstance(action, PickupAction):
                action_types[x, y] = 10
                action_resources[x, y] = action.resource
                action_amounts[x, y] = action.pickup_amount

            elif isinstance(action, DigAction):
                action_types[x, y] = 11

            elif isinstance(action, SelfDestructAction):
                action_types[x, y] = 12

    for player_id in state.units:
        for unit_id, unit in state.units[player_id].items():
            if unit_id not in seen_unit_ids:
                action_types[unit.pos.x, unit.pos.y] = 0

    return action_types, action_resources, action_amounts
