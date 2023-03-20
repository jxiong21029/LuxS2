import numpy as np
from luxai_s2.state import State
from luxai_s2.unit import UnitType


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
