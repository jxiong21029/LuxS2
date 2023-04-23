import glob
import json
import os
import random

import blosc2
import numpy as np
import tqdm
import zarr
from luxai_s2 import LuxAI_S2
from submission.preprocessing import get_actions, get_obs


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

    obs_meta = []
    obs_tiles = []
    action_types = []
    action_resources = []
    action_amounts = []
    for step in replay["steps"][1:]:
        p0, p1 = step
        actions = {"player_0": p0["action"], "player_1": p1["action"]}

        if env.state.real_env_steps >= 0:
            new_meta, new_tiles = get_obs(env.state, elo, lichen_diff)
            (
                new_action_types,
                new_action_resources,
                new_action_amounts,
            ) = get_actions(env.env_cfg, env.state, actions)

            obs_meta.append(new_meta)
            obs_tiles.append(new_tiles)
            action_types.append(new_action_types)
            action_resources.append(new_action_resources)
            action_amounts.append(new_action_amounts)

        env.step(actions)

    group = zarr.open_group("../replay_data.zarr", mode="r+")

    obs_meta = np.stack(obs_meta, axis=0)
    obs_tiles = np.stack(obs_tiles, axis=0)
    action_types = np.stack(action_types, axis=0)
    action_resources = np.stack(action_resources, axis=0)
    action_amounts = np.stack(action_amounts, axis=0)

    if group.attrs["length"] + obs_meta.shape[0] > group["obs_meta"].shape[0]:
        for k, arr in group.arrays():
            old_shape = arr.shape
            arr.resize((old_shape[0] * 2,) + old_shape[1:])

    l1 = group.attrs["length"]
    l2 = l1 + obs_meta.shape[0]

    group["obs_meta"][l1:l2] = obs_meta
    group["obs_tiles"][l1:l2] = obs_tiles
    group["action_types"][l1:l2] = action_types
    group["action_resources"][l1:l2] = action_resources
    group["action_amounts"][l1:l2] = action_amounts
    group["terminal"][l2 - 1] = True

    group.attrs["length"] = l2


def main():
    group = zarr.open_group("../replay_data.zarr", mode="w")
    group.attrs["length"] = 0
    group.zeros(
        "obs_meta", shape=(1024, 6), chunks=(2**18, None), dtype=np.float32
    )
    group.zeros(
        "obs_tiles",
        shape=(1024, 35, 48, 48),
        chunks=(32, None, None, None),
        dtype=np.float32,
    )
    group.zeros(
        "action_types",
        shape=(1024, 48, 48),
        chunks=(1024, None, None),
        dtype=np.int8,
    )
    group.zeros(
        "action_resources",
        shape=(1024, 48, 48),
        chunks=(1024, None, None),
        dtype=np.int32,
    )
    group.zeros(
        "action_amounts",
        shape=(1024, 48, 48),
        chunks=(1024, None, None),
        dtype=np.int32,
    )
    group.zeros("terminal", shape=(1024,), chunks=(2**20,), dtype=bool)

    filenames = list(glob.glob("raw/*.bl2"))
    random.seed(42)
    random.shuffle(filenames)

    print(f"replays available: {len(filenames)}")
    for i, filename in enumerate(tqdm.tqdm(filenames)):
        # still testing, so only 50 replays for now
        if i == 50:
            break

        convert_replay(filename)
        if i % 20 == 0:
            print(group.info)
            for k, arr in group.arrays():
                print(arr.info)

    print(group.info)
    for k, arr in group.arrays():
        print(arr.info)


if __name__ == "__main__":
    main()
