import torch
import zarr
from torch.utils.data import DataLoader

from logger import Logger
from models import LinearModel, UNet
from train import LuxAIDataset, Trainer
from tuning import Tuner

DATASET = "replays/replay_data_zarr/replay_data.zarr"
LOADER_WORKERS = 1
EPOCHS = 10


def checkpoint_filename(config):
    return "_".join(
        f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}"
        for k, v in config.items()
    )


def trial(config):
    array = zarr.open(DATASET)
    dataset = LuxAIDataset(array)

    train_dataset, valid_dataset = torch.utils.data.random_split(
        dataset, lengths=[0.9, 0.1]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        num_workers=LOADER_WORKERS,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=32,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    if config["model"] == "linear":
        network = LinearModel()
    elif config["model"] == "unet":
        network = UNet()
    else:
        assert config["model"] == "lraspp"
        raise ValueError()
        # network = LRASPP(out=20)

    logger = Logger()
    trainer = Trainer(
        network,
        logger,
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )

    trainer.evaluate(valid_loader, verbose=True)
    for _ in range(EPOCHS):
        trainer.train(train_loader, verbose=True)
        trainer.evaluate(valid_loader, verbose=True)

        torch.save(
            trainer.network.state_dict(),
            f"checkpoints/{checkpoint_filename(config)}.pth",
        )

        yield logger


def main():
    spec = {
        "model": "science",
        "lr": "nuisance",
        "weight_decay": "nuisance",
    }

    tuner = Tuner(
        spec,
        trial_fn=trial,
        metric="valid_loss",
        mode="min",
        throw_on_exception=True,
        trial_gpus=1,
    )

    for model in ("unet",):
        for lr in (
            10**-3,
            10**-2.5,
            10**-2,
        ):
            for weight_decay in (1e-5,):
                tuner.add(
                    {"model": model, "lr": lr, "weight_decay": weight_decay}
                )

    tuner.run()


if __name__ == "__main__":
    main()
