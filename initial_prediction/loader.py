import json
import os
import re

from torch import tensor
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

from luxai_s2 import LuxAI_S2
import luxai_s2


BOARD_RESOURCES = ["rubble", "ice", "ore"]


def position(encoded):
    return tuple(map(int, encoded.split(",")))


def decode_board(boards):
    final_board = None
    for board in boards:
        if isinstance(board, dict):
            for key, value in board.items():
                x, y = position(key)
                final_board[x][y] = value
        else:
            final_board = [row[:] for row in board]
    
    return final_board
    

def decode_factories(factories):
    player_0 = [[0] * 48 for _ in range(48)]
    player_1 = [[0] * 48 for _ in range(48)]
    water = [[0] * 48 for _ in range(48)]
    metal = [[0] * 48 for _ in range(48)]

    for player, array in [("player_0", player_0), ("player_1", player_1)]:
        for _, data in factories[player].items():
            x, y = data["pos"]
            array[x][y] = 1
            water[x][y] = data["cargo"]["water"]
            metal[x][y] = data["cargo"]["metal"]

    return player_0, player_1, water, metal

def start_state(data):
    states = []
    for step in data["steps"]:
        obs = step[0]["observation"]["obs"]
        state = json.loads(obs)
        states.append(state)
        if state["real_env_steps"] == 0:
            return states
    return None


class StateWinners(Dataset):
    def __init__(self, directory):
        self.directory = directory
        self.filenames = [
            filename
            for filename in os.listdir(directory)
            if re.match(r"^\d*\.json$", filename)
        ]

        # necessary preprocessing needed for resnet
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index]
        with open(os.path.join(self.directory, filename)) as f:
            data = json.load(f)
        setup_states = start_state(data)

        assert setup_states is not None, "no start state found"
        start = setup_states[-1]

        factories = decode_factories(start["factories"])

        layers = [
            decode_board([state["board"].get(key, {}) for state in setup_states]) for key in BOARD_RESOURCES
        ] + list(factories)

        reward_0, reward_1 = data["rewards"]
        winner = int(reward_0 < reward_1)

        return self.preprocess(tensor(layers).float()), winner
