import json
import os
import re

from torch import tensor
from torch.utils.data import Dataset


BOARD_RESOURCES = ["rubble", "ice", "ore"]


def position(encoded):
    return tuple(map(int, encoded.split(",")))


def decode_board(board):
    # dictionary 'x,y' -> value
    if isinstance(board, dict):
        output = [[0] * 48 for _ in range(48)]
        for key, value in board.items():
            x, y = position(key)
            output[x][y] = value
        return output
    # list [x][y] -> value
    else:
        return board


def decode_factories(factories):
    player_0 = [[0] * 48 for _ in range(48)]
    player_1 = [[0] * 48 for _ in range(48)]

    for player, array in [("player_0", player_0), ("player_1", player_1)]:
        for _, data in factories[player].items():
            x, y = data["pos"]
            array[x][y] = 1

    return player_0, player_1


def start_state(data):
    for step in data["steps"]:
        obs = step[0]["observation"]["obs"]
        state = json.loads(obs)
        if state["real_env_steps"] == 0:
            return state
    return None


class StateWinners(Dataset):
    def __init__(self, directory):
        self.directory = directory
        self.filenames = [
            filename
            for filename in os.listdir(directory)
            if re.match(r"^\d*\.json$", filename)
        ]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index]
        with open(os.path.join(self.directory, filename)) as f:
            data = json.load(f)
        start = start_state(data)

        assert start is not None, "no start state found"

        board = start["board"]

        factories = decode_factories(start["factories"])

        layers = [
            decode_board(board.get(key, {})) for key in BOARD_RESOURCES
        ] + list(factories)

        reward_0, reward_1 = data["rewards"]
        winner = int(reward_0 < reward_1)

        return tensor(layers), winner
