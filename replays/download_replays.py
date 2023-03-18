import datetime
import json
import os
import random
import time
from collections import defaultdict

import blosc
import numpy as np
import pandas as pd
import polars as pl
import requests

MIN_ELO = 1600

episodes_df = pl.read_csv("Episodes.csv")
episodes_df = episodes_df.filter(pl.col("CompetitionId") == 45040)
episodes_df = episodes_df.to_pandas()
print(f"Episodes.csv: {len(episodes_df)} rows after filtering")

epagents_df = pl.read_csv(
    "EpisodeAgents.csv",
    dtypes={"Reward": pl.Float32},
    n_threads=8,
    low_memory=True,
)
epagents_df = epagents_df.filter(
    pl.col("EpisodeId").is_in(episodes_df["Id"].to_list())
)
epagents_df = epagents_df.to_pandas()
epagents_df["InitialConfidence"] = (
    epagents_df["InitialConfidence"].replace("", np.nan).astype(float)
)
epagents_df["InitialScore"] = (
    epagents_df["InitialScore"].replace("", np.nan).astype(float)
)
print(f"EpisodeAgents.csv: {len(epagents_df)} rows after filtering")

episodes_df = episodes_df.set_index(["Id"])
episodes_df["CreateTime"] = pd.to_datetime(episodes_df["CreateTime"])
episodes_df["EndTime"] = pd.to_datetime(episodes_df["EndTime"])
epagents_df.fillna(0, inplace=True)
epagents_df = epagents_df.sort_values(by=["Id"], ascending=False)

max_df = (
    epagents_df.sort_values(by=["EpisodeId"], ascending=False)
    .groupby("SubmissionId")
    .head(1)
    .drop_duplicates()
    .reset_index(drop=True)
)
max_df = max_df[MIN_ELO <= max_df.UpdatedScore]
max_df = pd.merge(
    left=episodes_df, right=max_df, left_on="Id", right_on="EpisodeId"
)
subid_to_elo = pd.Series(
    max_df.UpdatedScore.values, index=max_df.SubmissionId
).to_dict()
print(f"{len(subid_to_elo)} submissions with elo over {MIN_ELO}")

sub_to_episodes = defaultdict(list)
excl = {30512132, 30496738, 30512387, 30512390}
for key, value in sorted(
    subid_to_elo.items(), key=lambda kv: kv[1], reverse=True
):
    if key not in excl:  # we can filter subs like this
        eps = sorted(
            epagents_df[epagents_df["SubmissionId"].isin([key])][
                "EpisodeId"
            ].values,
            reverse=True,
        )
        sub_to_episodes[key] = eps
    else:
        print(f"skipped {key}")
candidates = len(
    set([item for sublist in sub_to_episodes.values() for item in sublist])
)
print(f"{candidates} episodes for these {len(subid_to_elo)} submissions")

all_files = []
for root, dirs, files in os.walk("./", topdown=False):
    all_files.extend(files)
seen_episodes = [
    int(f.split(".")[0])
    for f in all_files
    if "." in f and f.split(".")[0].isdigit() and f.split(".")[1] == "json"
]
remaining = np.setdiff1d(
    [item for sublist in sub_to_episodes.values() for item in sublist],
    seen_episodes,
)
print(f"{len(remaining)} of these {candidates} episodes not yet saved")
print("Total of {} games in existing library".format(len(seen_episodes)))


def create_info_json(epid_):
    create_seconds = int(
        (
            episodes_df[episodes_df.index == epid_]["CreateTime"].values[0]
        ).item()
        / 1e9
    )
    end_seconds = int(
        (
            episodes_df[episodes_df.index == epid_]["CreateTime"].values[0]
        ).item()
        / 1e9
    )

    agents = []
    for index, row in (
        epagents_df[epagents_df["EpisodeId"] == epid_]
        .sort_values(by=["Index"])
        .iterrows()
    ):
        agent = {
            "id": int(row["Id"]),
            "state": int(row["State"]),
            "submissionId": int(row["SubmissionId"]),
            "reward": float(row["Reward"]),
            "index": int(row["Index"]),
            "initialScore": float(row["InitialScore"]),
            "initialConfidence": float(row["InitialConfidence"]),
            "updatedScore": float(row["UpdatedScore"]),
            "updatedConfidence": float(row["UpdatedConfidence"]),
        }
        agents.append(agent)

    info = {
        "id": int(epid_),
        "createTime": {"seconds": int(create_seconds)},
        "endTime": {"seconds": int(end_seconds)},
        "agents": agents,
    }

    return info


EPISODE_URL = (
    "https://www.kaggle.com/api/i/competitions.EpisodeService/GetEpisodeReplay"
)


def save_episode(epid_):
    re = requests.post(EPISODE_URL, json={"episodeId": int(epid_)})

    replay = re.json()
    with open(f"{epid_}.dat", "wb") as f:
        f.write(blosc.compress(json.dumps(replay).encode()))

    info = create_info_json(epid_)
    with open(f"{epid_}_info.dat", "wb") as f:
        f.write(blosc.compress(json.dumps(info).encode()))


t = 1

start_time = datetime.datetime.now()
count = 0

subid_to_epid = {k: list(v) for k, v in sub_to_episodes.items()}
for k in subid_to_epid:
    random.shuffle(subid_to_epid[k])
seen_episodes = set(seen_episodes)

subids = list(subid_to_epid.keys())
while True:
    subid = random.choice(subids)

    if len(subid_to_epid[subid]) == 0:
        subids.remove(subid)
        continue
    epid = subid_to_epid[subid].pop()
    if epid in seen_episodes:
        continue

    save_episode(epid)

    t += 1
    count += 1
    if os.path.exists(os.path.join(f"{epid}.dat")):
        print(
            f"{count: >4}: saved episode #{epid} from submission "
            f"{subid}(elo={round(subid_to_elo[subid])})"
        )
        seen_episodes.add(epid)
    else:
        print(f"error saving episode {epid}")

    if t > (datetime.datetime.now() - start_time).seconds:
        time.sleep(t - (datetime.datetime.now() - start_time).seconds)

print(f"Episodes saved: {count}")
