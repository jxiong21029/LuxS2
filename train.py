import numpy as np
import torch
import torch.nn as nn
import zarr
from torch.utils.data import DataLoader, Dataset

from model import LuxAIModel
from tuning.logger import Logger

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)


class LuxAIDataset(Dataset):
    def __init__(self, array):
        self.array = array

    def __len__(self):
        return self.array["action_amounts"].shape[0]

    def __getitem__(self, idx):
        unit_mask = (
            np.sum(self.array["obs_tiles"][idx][9:13, :, :], 0) > 0
        ).astype(int)
        return (
            self.array["obs_tiles"][idx],
            unit_mask,
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

    def accuracy(self):
        self.network.eval()
        type_correct, total_type = 0, 0
        for example in self.dataloader:
            (
                board,
                unit_mask,
                action_types,
                action_resources,
                action_amounts,
            ) = example
            (
                predicted_types,
                predicted_resources,
                predicted_quantities,
            ) = self.network(board.to(device))
            predicted_types, predicted_resources = torch.argmax(
                predicted_types, 3
            ), torch.argmax(predicted_resources, 3)
            type_correct += torch.sum((predicted_types == action_types.to(device).long() * unit_mask.to(device)))
            total_type += torch.sum((unit_mask > 0).long())
        self.logger.push(accuracy=type_correct / total_type)
        self.logger.step()
            

    def train(self):
        self.network.train()
        for example in self.dataloader:
            self.optimizer.zero_grad()
            (
                board,
                unit_mask,
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
            # predicted_resources is batch size x 48 x 48 x 4 (4 possible resources)
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
                    action_resources.long().to(device),
                    reduction="none",
                )
                * unit_mask.to(device)
            )
            mse_loss = torch.mean(
                torch.square(predicted_quantities - action_amounts.to(device))
            )
            l = ce_loss + mse_loss

            l.backward()
            self.optimizer.step()

            self.logger.push(loss=l)

        self.logger.step()


if __name__ == "__main__":
    # keys: action_amounts, action_resources, action_types, obs_meta, obs_tiles
    array = zarr.open("replays/replay_data_zarr/replay_data.zarr")
    dataset = LuxAIDataset(array)
    train_dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        num_workers=1,
    )
    network = LuxAIModel()

    trainer = Trainer(train_dataloader, network, Logger())
    trainer.train()
    trainer.accuracy()
