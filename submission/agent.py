import sys

import numpy as np
from lux.config import EnvConfig
from lux.utils import direction_to, my_turn_to_place_factory
from luxai_s2.state import State
from preprocessing import get_obs


class Agent:
    def __init__(self, player: str, env_cfg: EnvConfig) -> None:
        self.player = player
        self.opp_player = (
            "player_1" if self.player == "player_0" else "player_0"
        )
        np.random.seed(0)
        self.env_cfg: EnvConfig = env_cfg

    def early_setup(self, step: int, obs, remainingOverageTime: int = 60):
        if step == 0:
            return dict(faction="AlphaStrike", bid=0)
        else:
            # factory placement period
            game_state = State.from_obs(obs, self.env_cfg)
            print(game_state.board.__dict__.keys(), file=sys.stderr)

            # how many factories you have left to place
            factories_to_place = game_state.teams[
                self.player
            ].factories_to_place
            # whether it is your turn to place a factory
            my_turn_to_place = my_turn_to_place_factory(
                game_state.teams[self.player].place_first, step
            )
            if factories_to_place > 0 and my_turn_to_place:
                potential_spawns = np.array(
                    list(
                        zip(*np.where(obs["board"]["valid_spawns_mask"] == 1))
                    )
                )
                spawn_loc = potential_spawns[
                    np.random.randint(0, len(potential_spawns))
                ]
                return dict(spawn=spawn_loc, metal=150, water=150)
            return dict()

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        actions = dict()

        obs = State.from_obs(obs, self.env_cfg)

        return actions
