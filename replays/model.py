import torch
from torch import nn

class LuxAIModel(nn.Module):
    def __init__(self):
        super(LuxAIModel, self).__init__()

        # baseline architecture: CNN with 1x1 kernel followed by linear layer (for both regression and "segmentation")
        self.types_conv = nn.Conv2d(35, 1, 1)
        self.types_linear = nn.Linear(1, 13)
        self.quantities_conv = nn.Conv2d(35, 2, 1)
        self.quantities_linear = nn.Linear(2, 2)
    
    def forward(self, board):
        # board is of shape batch_size x 35 x 48 x 48
        predicted_types = self.types_conv(board).transpose(1, 3).transpose(1, 2) # shape is now batch_size x 48 x 48 x 1
        predicted_types = self.types_linear(predicted_types) # shape is now batch_size x 48 x 48 x 13

        predicted_quantities = self.quantities_conv(board).transpose(1, 3).transpose(1, 2) # shape is now batch_size x 48 x 48 x 2
        predicted_quantities = self.quantities_linear(predicted_quantities) # shape is still batch_size x 48 x 48 x 2

        return predicted_types, predicted_quantities

