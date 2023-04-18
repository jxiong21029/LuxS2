import itertools
import torch
import zarr

from model import LuxAIModel
from torch.utils.data import DataLoader
from train import LuxAIDataset, Trainer
from tuning.logger import Logger
from tuning.tuning import Tuner

DATASET = "replays/replay_data_zarr/replay_data.zarr"
LOADER_WORKERS = 1
EPOCHS = 1

def trial(config):
    missing = ({"lr", "weight_decay", "batch_size"} - config.keys())
    assert not missing, f"missing keys: {missing}"

    array = zarr.open(DATASET)
    # for testing
    array["action_amounts"] = array["action_amounts"][:1000]
    data = LuxAIDataset(array)
    loader = DataLoader(
        data,
        batch_size=config["batch_size"],
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        num_workers=LOADER_WORKERS,
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
    }

    tuner = Tuner(
        spec,
        trial_fn=trial,
        metric="loss",
        mode="min",
        throw_on_exception=True,
        trial_gpus=1,
    )
    params = {
        "batch_size": [16],
        "lr": [1e-3, 1e-2],
        "weight_decay": [1e-5],
    }

    for config in itertools.product(*params.values()):
        config = dict(zip(params.keys(), config))
        tuner.add(config)

    tuner.run()

if __name__ == "__main__":
    main()
