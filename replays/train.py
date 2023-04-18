import zarr
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import LuxAIModel

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


def train(dataloader, network, lr=1e-5, weight_decay=0, epochs=10):
    network.train()
    optimizer = torch.optim.Adam(
        network.parameters(), lr=lr, weight_decay=weight_decay
    )
    network.to(device)

    for i in range(epochs):
        total = 0
        for example in dataloader:
            optimizer.zero_grad()
            (
                board,
                unit_mask,
                action_types,
                action_resources,
                action_amounts,
            ) = example
            regression_true = torch.cat(
                (
                    torch.unsqueeze(action_resources, 3),
                    torch.unsqueeze(action_amounts, 3),
                ),
                3,
            )
            predicted_types, predicted_quantities = network(board.to(device))
            # predicted_types is batch_size x 48 x 48 x 13 (13 action types)
            # predicted_quantities is batch size x 48 x 48 x 2 (one channel for action_resources, one for action_amounts)

            ce_loss = torch.mean(
                nn.functional.cross_entropy(
                    predicted_types.transpose(1, 3).transpose(2, 3),
                    action_types.long().to(device),
                    reduction="none",
                )
                * unit_mask.to(device)
            )
            mse_loss = torch.mean(
                torch.square(predicted_quantities - regression_true.to(device))
            )
            l = ce_loss + mse_loss
            total += l

            l.backward()
            optimizer.step()

        print("Epoch={}, Total Loss={}".format(i, total))


if __name__ == "__main__":
    # keys: action_amounts, action_resources, action_types, obs_meta, obs_tiles
    array = zarr.open("replay_data_zarr/replay_data.zarr")
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

    train(train_dataloader, network)
