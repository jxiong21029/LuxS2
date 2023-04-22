import itertools

import torch
import zarr
from torch.utils.data import DataLoader

from model import LuxAIModel, UNet
from train import LuxAIDataset, Trainer
from tuning.logger import Logger
from tuning.tuning import Tuner

DATASET = "replays/replay_data_zarr/replay_data.zarr"
LOADER_WORKERS = 1
EPOCHS = 4


# extraneous parmeter slots
# useful for model-specific params
extras = {
    "extra_1": "nuisance",
    "extra_2": "nuisance",
    "extra_3": "nuisance",
}


def trial(config):
    missing = (
        ({"lr", "weight_decay", "batch_size"} | extras.keys()) - config.keys()
    )
    assert not missing, f"missing keys: {missing}"

    array = zarr.open(DATASET)
    data = LuxAIDataset(array, max_size=1000)
    loader = DataLoader(
        data,
        batch_size=config["batch_size"],
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        num_workers=LOADER_WORKERS,
    )

    kwargs = {
        name: value
        for (name, value) in (
            (None, None)
            if config[key] is None
            else config[key]
            for key in extras.keys()
        )
        if value is not None
    }

    print(config["model"], kwargs)

    network = config["model"](**kwargs)

    logger = Logger()
    trainer = Trainer(
        loader,
        network,
        logger,
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )

    trainer.evaluate()
    for _ in range(EPOCHS):
        trainer.train()
        trainer.evaluate()
        yield logger


def main():
    spec = {
        "model": "science",
        "batch_size": "nuisance",
        "lr": "nuisance",
        "weight_decay": "nuisance",
    }

    tuner = Tuner(
        spec | extras,
        trial_fn=trial,
        metric="loss",
        mode="min",
        throw_on_exception=True,
        trial_gpus=1,
    )

    models = {
        "unet": (UNet, {"extra_1": ("out", [20, 30, 40])}),
        "linear": (LuxAIModel, {}),
    }

    params = {
        "batch_size": [16],
        "lr": [1e-3, 1e-2],
        "weight_decay": [1e-5],
    }

    for _, (model, options) in models.items():
        # i.e. if models = {'linear': (A, {'extra_1': ('param', [1, 2])})}
        # we want to produce
        # {
        #     'extra_1': [('param', 1), ('param', 2)]
        #     'extra_2': [None]
        #     'extra_3': [None]
        # }
        additional = {
            key: [None] if name is None else [
                (name, choice)
                for choice in choices
            ]
            for key, (name, choices) in (
                (key, options.get(key, (None, [])))
                for key in extras.keys()
            )
        }

        all_params = params | additional
        print(all_params)
        for options in itertools.product(*all_params.values()):
            tuner.add(dict(zip(all_params.keys(), options)) | {"model": model})

    tuner.run()


if __name__ == "__main__":
    main()
