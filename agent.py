import sys

import numpy as np
import torch
import torch.nn as nn
from luxai_s2.state import State

import lux.unit
from lux.config import EnvConfig
from lux.utils import my_turn_to_place_factory
from preprocessing import get_obs


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
            state = State.from_obs(obs, self.env_cfg)
            print(state.board.__dict__.keys(), file=sys.stderr)

            # how many factories you have left to place
            factories_to_place = state.teams[self.player].factories_to_place
            # whether it is your turn to place a factory
            my_turn_to_place = my_turn_to_place_factory(
                state.teams[self.player].place_first, step
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
                return {"spawn": spawn_loc, "metal": 150, "water": 150}
            return {}

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
            unit: lux.unit.Unit

            idx = np.argmax(logits[unit.pos[0], unit.pos[1]])
            actions[unit_id] = unit.move

        return actions
