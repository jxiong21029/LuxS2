import numpy as np
from luxai_s2 import LuxAI_S2
from luxai_s2.config import EnvConfig
from luxai_s2.utils import my_turn_to_place_factory

from lux.kit import obs_to_game_state


class Agent:
    def __init__(self, player: str, env_cfg: EnvConfig):
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.env_cfg = env_cfg

    def setup_act(self, step: int, obs):
        if step == 0:
            return {"faction": "MotherMars", "bid": 0}

        actions = {}
        game_state = obs_to_game_state(step, self.env_cfg, obs)

        is_my_turn = my_turn_to_place_factory(
            game_state.teams[self.player].place_first, step
        )
        if is_my_turn and game_state.teams[self.player].factories_to_place > 0:
            candidate_positions = np.array(
                list(zip(*np.where(obs["board"]["valid_spawns_mask"])))
            )

        # if game_state.teams[self.player].water

        return actions

    def act(self, step: int, obs):
        actions = {}
        game_state = obs_to_game_state(step, self.env_cfg, obs)
        return actions


def evaluate():
    pass
