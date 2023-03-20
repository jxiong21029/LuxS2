import glob
import json
import os

import blosc2
import numpy as np
import tqdm
from luxai_s2 import LuxAI_S2

from observation import get_obs


def convert_replay(filename):
    with open(filename, "rb") as f:
        replay = json.loads(blosc2.decompress(f.read()).decode())

    info_filename = os.path.splitext(filename)[0] + "_info.json"
    with open(info_filename, "r") as f:
        info = json.load(f)

    elo = (
        info["agents"][0]["updatedScore"] + info["agents"][1]["updatedScore"]
    ) / 2

    if (
        replay["steps"][-1][0]["observation"]["reward"] >= 0
        and replay["steps"][-1][1]["observation"]["reward"] >= 0
    ):
        lichen_diff = (
            replay["steps"][-1][0]["observation"]["reward"]
            - replay["steps"][-1][1]["observation"]["reward"]
        )
    else:
        lichen_diff = (
            replay["steps"][-2][0]["observation"]["reward"]
            - replay["steps"][-2][1]["observation"]["reward"]
        )

    env = LuxAI_S2()
    env.reset(seed=replay["configuration"]["seed"])

    metas = []
    tiles = []
    for step in replay["steps"][1:]:
        p0, p1 = step
        actions = {"player_0": p0["action"], "player_1": p1["action"]}
        env.step(actions)

        if env.state.real_env_steps < 0:
            continue

        new_meta, new_tiles = get_obs(env.state, elo, lichen_diff)
        metas.append(new_meta)
        tiles.append(new_tiles)

    dataset_meta = np.stack(metas, axis=0)
    dataset_tiles = np.stack(tiles, axis=0)

    replay_id = os.path.splitext(os.path.split(filename)[1])[0]
    os.makedirs("cooked/", exist_ok=True)

    with open(f"cooked/{replay_id}_meta.dat", "wb") as f:
        f.write(blosc2.compress(dataset_meta))
    with open(f"cooked/{replay_id}_tiles.dat", "wb") as f:
        f.write(blosc2.compress(dataset_tiles))


def main():
    for filename in tqdm.tqdm(list(glob.glob("raw/*.dat"))):
        convert_replay(filename)


if __name__ == "__main__":
    main()


