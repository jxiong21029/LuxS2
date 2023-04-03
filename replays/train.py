import zarr
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)

class LuxAIDataset(Dataset):
    def __init__(self, array):
        self.array = array
    
    def __len__(self):
        return self.array["action_amounts"].shape[0]

    def __getitem__(self, idx):
        return self.array["obs_tiles"][idx], self.array["action_types"][idx], self.array["action_resources"][idx], self.array["action_amounts"][idx]
    
def train(dataloader, network, lr=1e-5, weight_decay=0, epochs=10):
    ce_loss = nn.CrossEntropyLoss()
    network.train()
    optimizer = optim.Adam(
        network.parameters(), lr=lr, weight_decay=weight_decay
    )
    network.to(device)

    for i in range(epochs):
        total = 0
        for example in dataloader:
            board, action_types, action_resources, action_amounts = example
            regression_output = torch.cat((torch.unsqueeze(action_resources, 1), torch.unsqueeze(action_amounts, 1)), 1)
            predicted_types, predicted_quantities = network(board.to(device))
            # predicted_types should be batch_size x 48 x 48 x 13 (13 action types)
            # predicted_quantities should be batch size x 2 x 48 x 48 (one channel for action_resources, one for action_amounts)

            l = ce_loss(predicted_types.transpose(1, 3).transpose(2, 3), action_types) + torch.sum(torch.square(predicted_quantities - regression_output))
            total += l
        print("Epoch={}, Total Loss={}".format(i, total))

# keys: action_amounts, action_resources, action_types, obs_meta, obs_tiles
array = zarr.open("replay_data_zarr/replay_data.zarr")
dataset = LuxAIDataset(array)
train_dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

