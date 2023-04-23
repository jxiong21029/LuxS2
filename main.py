import json
from argparse import Namespace

from luxai_s2.state import State
from model import UNet

from agent import Agent
from lux.config import EnvConfig
from lux.kit import process_obs, to_json


def read_input():
    """
    Reads input from stdin
    """
    try:
        return input()
    except EOFError as eof:
        raise SystemExit(eof)


def main():
    env_cfg = None
    i = 0

    agents = {}
    prev_obs = {}

    while True:
        inputs = read_input()
        obs = json.loads(inputs)

        observation = Namespace(
            **dict(
                step=obs["step"],
                obs=json.dumps(obs["obs"]),
                remainingOverageTime=obs["remainingOverageTime"],
                player=obs["player"],
                info=obs["info"],
            )
        )
        if i == 0:
            env_cfg = obs["info"]["env_cfg"]
        i += 1

        step = observation.step

        player = observation.player
        if step == 0:
            env_cfg = EnvConfig.from_dict(env_cfg)

            model = UNet()
            agents[player] = Agent(
                model, "whatever_filename_TODO.pth", player, env_cfg
            )

            prev_obs[player] = dict()

        agent = agents[player]
        obs = process_obs(
            player, prev_obs[player], step, json.loads(observation.obs)
        )
        prev_obs[player] = obs
        agent.step = step

        if obs["real_env_steps"] < 0:
            actions = agent.early_setup(step, obs)
        else:
            actions = agent.act(obs)

        print(json.dumps(to_json(actions)))


if __name__ == "__main__":
    main()
