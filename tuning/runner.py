import itertools
import torch
import zarr

from logger import Logger
from model import LuxAIModel
from torch.utils.data import DataLoader
from train import LuxAIDataset, Trainer
from tuning.tuning import Tuner

DATASET = "replays/replay_data_zarr/replay_data.zarr"
WORKERS = 1
EPOCHS = 10

def trial(config):
    missing = (
        {"lr", "weight_decay", "batch_size", "trial"}
        - config.keys(),
    )
    assert not missing, f"missing keys: {missing}"

    data = LuxAIDataset(zarr.open(DATASET))
    loader = DataLoader(
        data,
        batch_size=config["batch_size"],
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        num_workers=WORKERS,
    )
    network = LuxAIModel()
    logger = Logger()
    trainer = Trainer(
        loader,
        network,
        logger,
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )

    for _ in range(EPOCHS):
        trainer.train()
        yield logger


def main():
    spec = {
        "batch_size": "nuisance",
        "lr": "nuisance",
        "weight_decay": "science",
        "trial": "id",
    }

    tuner = Tuner(spec, trial_fn=trial, metric="loss", mode="min")
    params = {
        "batch_size": [16, 32],
        "lr": [1e-3, 1e-2],
        "weight_decay": [1e-5, 1e-4],
    }

    for config in itertools.product(*params.values()):
        config = dict(zip(params.keys(), config))
        tuner.add(config)

    tuner.run()
