import numpy as np
import torch
import torch.nn as nn
import tqdm
import zarr
import zarr.core
from torch.utils.data import DataLoader, Dataset

from model import LuxAIModel, UNet, LRaspp
from tuning.logger import Logger

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)


class LuxAIDataset(Dataset):
    def __init__(self, array: zarr.core.Array, max_size=None):
        self.array: zarr.core.Array = array
        self.max_size = max_size

    def __len__(self):
        if self.max_size is None:
            return self.array.attrs["length"]
        return min(self.max_size, self.array.attrs["length"])

    def __getitem__(self, idx):
        unit_mask = (
            np.sum(self.array["obs_tiles"][idx][9:13, :, :], 0) > 0
        ).astype(int)

        # we want a mask for when action type is 5-10
        resource_mask = (5 <= self.array["action_types"][idx]) * (
            self.array["action_types"][idx] <= 10
        )
        meta = torch.zeros((self.array["obs_meta"][idx].shape[0], 48, 48))
        for i in range(self.array["obs_meta"][idx].shape[0]):
            meta[i] = torch.full((48, 48), self.array["obs_meta"][idx][i])
        board = torch.cat((torch.tensor(self.array["obs_tiles"][idx]), meta), 0)

        return (
            board,
            unit_mask,
            resource_mask,
            self.array["action_types"][idx],
            self.array["action_resources"][idx],
            self.array["action_amounts"][idx],
        )


class Trainer:
    def __init__(
        self,
        dataloader,
        network,
        logger,
        lr=1e-3,
        weight_decay=0,
    ):
        self.dataloader = dataloader
        self.network = network
        self.network.to(device)
        self.logger = logger
        self.params = {
            "lr": lr,
            "weight_decay": weight_decay,
        }

        self.optimizer = torch.optim.Adam(
            network.parameters(), lr=lr, weight_decay=weight_decay
        )

    def train(self, verbose=False):
        self.network.train()
        for example in (
            tqdm.tqdm(self.dataloader, desc="train")
            if verbose
            else self.dataloader
        ):
            self.optimizer.zero_grad()
            (
                board,
                unit_mask,
                resource_mask,
                action_types,
                action_resources,
                action_amounts,
            ) = example
            (
                predicted_types,
                predicted_resources,
                predicted_quantities,
            ) = self.network(board.to(device))
            # predicted_types is batch_size x 48 x 48 x 13 (13 action types)
            # predicted_resources is batch_size x 48 x 48 x 4 (4 resources)
            # predicted_quantites is batch_size x 48 x 48 x 1 (regression)

            ce_loss = torch.mean(
                nn.functional.cross_entropy(
                    predicted_types.transpose(1, 3).transpose(2, 3),
                    action_types.long().to(device),
                    reduction="none",
                )
                * unit_mask.to(device)
            ) + torch.mean(
                nn.functional.cross_entropy(
                    predicted_resources.transpose(1, 3).transpose(2, 3),
                    (
                        action_resources.long().to(device)
                        # * resource_mask.to(device)
                    ),
                    reduction="none",
                )
                * resource_mask.to(device)
            )
            mse_loss = torch.mean(
                torch.square(
                    (predicted_quantities - action_amounts.to(device))
                    * unit_mask.to(device)
                    / 1000
                )
            )
            loss = ce_loss + mse_loss

            loss.backward()
            self.optimizer.step()

            self.logger.push(loss=loss)

        self.logger.step()

    def evaluate(self, verbose=False):
        self.network.eval()
        type_correct, total_type = 0, 0
        for example in (
            tqdm.tqdm(self.dataloader, desc="eval")
            if verbose
            else self.dataloader
        ):
            (
                board,
                unit_mask,
                _,
                action_types,
                _,
                _,
            ) = example
            (
                predicted_types,
                _,
                _,
            ) = self.network(board.to(device))
            predicted_types = torch.argmax(predicted_types, 3)
            type_correct += torch.sum(
                (predicted_types == action_types.to(device).long())
                * unit_mask.to(device)
            )
            total_type += torch.sum((unit_mask > 0).long())
        self.logger.log(accuracy=type_correct / total_type)


if __name__ == "__main__":
    # keys: action_amounts, action_resources, action_types, obs_meta, obs_tiles
    array = zarr.open("replays/replay_data_zarr/replay_data.zarr")
    dataset = LuxAIDataset(array, max_size=5000)
    train_dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        num_workers=1,
    )
    network = LRaspp()

    trainer = Trainer(train_dataloader, network, Logger())
    for _ in range(1):
        trainer.train(verbose=True)
        trainer.evaluate(verbose=True)
        print(trainer.logger.data)
