import luxai_s2.wrappers.sb3
import numpy as np
from luxai_s2 import LuxAI_S2
from luxai_s2.config import EnvConfig
from luxai_s2.utils import my_turn_to_place_factory

from lux.factory import Factory
from lux.kit import GameState, obs_to_game_state
from lux.utils import direction_to


def factory_location_score(state: GameState, x, y):
    factory_locs = []
    for agent in state.factories.keys():
        for unit_id in state.factories[agent]:
            factory: Factory = state.factories[agent][unit_id]
            factory_locs.append([factory.pos[0], factory.pos[1]])
    factory_locs = np.array(factory_locs)

    x = np.asarray(x)
    y = np.asarray(y)

    p = -1.0

    ice_rates = np.zeros(x.shape)
    ore_rates = np.zeros(y.shape)
    for ax in range(48):
        for ay in range(48):
            # we assume that the amount of a resource that each factory will obtain
            # from some tile is propto dist ** p, where currently p = -1.0
            dist = np.abs(x - ax) + np.abs(y - ay)
            if factory_locs.size > 0:
                other_dists = np.abs(factory_locs[:, 0] - ax) + np.abs(
                    factory_locs[:, 1] - ay
                )

                # the proportion of a resource a factory at x, y would be able to obtain
                # from ax, ay, given the locations of existing factories
                proportion = np.power(dist + 1, p) / (
                    np.power(dist + 1, p) + np.power(other_dists + 1, p).sum()
                )
            else:
                proportion = 1

            rate = proportion * np.power(dist + 1, p)
            if state.board.ice[ax, ay]:
                ice_rates += rate
            if state.board.ore[ax, ay]:
                ore_rates += rate

    scores = np.minimum(ice_rates, ore_rates)
    return scores


class Agent:
    def __init__(self, player: str, env_cfg: EnvConfig):
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.env_cfg = env_cfg

    def setup_act(self, step: int, obs):
        if step == 0:
            return {"faction": "MotherMars", "bid": 0}

        state = obs_to_game_state(step, self.env_cfg, obs)

        is_my_turn = my_turn_to_place_factory(
            state.teams[self.player].place_first, step
        )
        if is_my_turn and state.teams[self.player].factories_to_place > 0:
            test_x, test_y = np.where(obs["board"]["valid_spawns_mask"])
            scores = factory_location_score(state, test_x, test_y)
            best = np.argmax(scores)

            return dict(
                spawn=np.array([test_x[best], test_y[best]]), metal=150, water=150
            )
        return {}

    def act(self, step: int, obs):
        # slightly modified starter code

        actions = {}
        game_state = obs_to_game_state(step, self.env_cfg, obs)
        factories = game_state.factories[self.player]

        factory_tiles, factory_units = [], []
        for unit_id, factory in factories.items():
            if (
                factory.power >= self.env_cfg.ROBOTS["LIGHT"].POWER_COST
                and factory.cargo.metal >= self.env_cfg.ROBOTS["LIGHT"].METAL_COST
            ):
                actions[unit_id] = factory.build_light()
            elif factory.cargo.water - factory.water_cost(game_state) >= 100:
                actions[unit_id] = factory.water()
            factory_tiles += [factory.pos]
            factory_units += [factory]
        factory_tiles = np.array(factory_tiles)

        units = game_state.units[self.player]
        ice_map = game_state.board.ice
        ice_tile_locations = np.argwhere(ice_map == 1)

        for unit_id, unit in units.items():
            # track the closest factory
            if len(factory_tiles) > 0:
                factory_distances = np.abs(factory_tiles - unit.pos).sum()
                closest_factory_tile = factory_tiles[np.argmin(factory_distances)]
                adjacent_to_factory = (
                    np.mean((closest_factory_tile - unit.pos) ** 2) == 0
                )

                if unit.cargo.ice < 80:
                    ice_tile_distances = np.mean(
                        (ice_tile_locations - unit.pos) ** 2, 1
                    )
                    closest_ice_tile = ice_tile_locations[np.argmin(ice_tile_distances)]
                    if np.all(closest_ice_tile == unit.pos):
                        if unit.power >= unit.dig_cost(
                            game_state
                        ) + unit.action_queue_cost(game_state):
                            actions[unit_id] = [unit.dig(repeat=0, n=1)]
                    else:
                        direction = direction_to(unit.pos, closest_ice_tile)
                        move_cost = unit.move_cost(game_state, direction)
                        if (
                            move_cost is not None
                            and unit.power
                            >= move_cost + unit.action_queue_cost(game_state)
                        ):
                            actions[unit_id] = [unit.move(direction, repeat=0, n=1)]
                else:
                    direction = direction_to(unit.pos, closest_factory_tile)
                    if adjacent_to_factory:
                        if unit.power >= unit.action_queue_cost(game_state):
                            actions[unit_id] = [
                                unit.transfer(direction, 0, unit.cargo.ice, repeat=0)
                            ]
                    else:
                        move_cost = unit.move_cost(game_state, direction)
                        if (
                            move_cost is not None
                            and unit.power
                            >= move_cost + unit.action_queue_cost(game_state)
                        ):
                            actions[unit_id] = [unit.move(direction, repeat=0, n=1)]
        return actions
