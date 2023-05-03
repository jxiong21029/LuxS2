import sys

import numpy as np
import torch
import torch.nn as nn
from luxai_s2.state import State

from early_game.placement import SetupEstimator
from lux.config import EnvConfig
from lux.unit import Unit
from lux.utils import my_turn_to_place_factory
from preprocessing import get_obs

BOARD_SIZE = (48, 48)


def display(*args, **kwargs) -> None:
    """Display something for debugging purposes."""
    print(*args, **kwargs, file=sys.stderr)


def valid_factories(state: State) -> np.ndarray:
    """Return a list of valid factory placement locations."""
    return np.array(list(zip(*np.where(state.board.valid_spawns_mask == 1))))


class Agent:
    def __init__(
        self,
        model: nn.Module,
        model_ckpt_filename: str,
        player: str,
        env_cfg: EnvConfig,
    ) -> None:
        self.model = model
        self.model.load_state_dict(torch.load(model_ckpt_filename))

        self.player = player
        self.opp_player = (
            "player_1" if self.player == "player_0" else "player_0"
        )
        np.random.seed(0)
        self.env_cfg: EnvConfig = env_cfg

    def early_setup(self, step: int, obs):
        if step == 0:
            return dict(faction="AlphaStrike", bid=0)
        else:
            # factory placement period
            game_state = State.from_obs(obs, self.env_cfg)

            # how many factories you have left to place
            factories_to_place = game_state.teams[
                self.player
            ].factories_to_place
            # whether it is your turn to place a factory
            my_turn_to_place = my_turn_to_place_factory(
                game_state.teams[self.player].place_first, step
            )
            if factories_to_place > 0 and my_turn_to_place:
                potential_spawns = valid_factories(game_state)

                scores = SetupEstimator().fit().predict(game_state.board)
                indices = np.ravel_multi_index(potential_spawns.T, BOARD_SIZE)
                spawn_loc = potential_spawns[np.argmax(scores[indices])]
                return dict(spawn=spawn_loc, metal=150, water=150)
            return dict()

    def act(self, obs):
        actions = {}

        state = State.from_obs(obs, self.env_cfg)

        lichen_diff = 100 if self.player == "player_0" else -100
        meta, board = get_obs(state, 1700, lichen_diff)

        board = torch.cat(
            [torch.broadcast_to(meta, (meta.shape[0], 48, 48)), board]
        )
        logits = self.model(board)

        for unit_id, unit in state.units[self.player]:
            unit: Unit

            idx = np.argmax(logits[unit.pos[0], unit.pos[1]])
            actions[unit_id] = unit.move

        return actions
