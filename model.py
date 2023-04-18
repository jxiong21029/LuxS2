import torch
from torch import nn

class LuxAIModel(nn.Module):
    def __init__(self):
        super(LuxAIModel, self).__init__()

        # baseline architecture: CNN with 1x1 kernel followed by linear layer (for both regression and "segmentation")
        self.types_conv = nn.Conv2d(35, 13, 1)
        self.resources_conv = nn.Conv2d(35, 4, 1)
        self.quantities_conv = nn.Conv2d(35, 1, 1)
    
    def forward(self, board):
        # board is of shape batch_size x 35 x 48 x 48
        predicted_types = self.types_conv(board).transpose(1, 3).transpose(1, 2) # shape is now batch_size x 48 x 48 x 13
        predicted_resources = self.resources_conv(board).transpose(1, 3).transpose(1, 2) # shape is now batch_size x 48 x 48 x 4
        predicted_quantities = self.quantities_conv(board).transpose(1, 3).transpose(1, 2)[:, :, :, 0] # shape is now batch_size x 48 x 48

        return predicted_types, predicted_resources, predicted_quantities

