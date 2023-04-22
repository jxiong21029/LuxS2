import torch
import torchvision
from torch import nn


class LuxAIModel(nn.Module):
    def __init__(self):
        super(LuxAIModel, self).__init__()

        # baseline architecture: CNN with 1x1 kernel followed by linear layer
        # (for both regression and "segmentation")
        self.types_conv = nn.Conv2d(41, 13, 1)
        self.resources_conv = nn.Conv2d(41, 5, 1)
        self.quantities_conv = nn.Conv2d(41, 1, 1)

    def forward(self, board):
        # board is of shape batch_size x 35 x 48 x 48
        predicted_types = (
            self.types_conv(board).transpose(1, 3).transpose(1, 2)
        )  # shape is now batch_size x 48 x 48 x 13
        predicted_resources = (
            self.resources_conv(board).transpose(1, 3).transpose(1, 2)
        )  # shape is now batch_size x 48 x 48 x 5
        predicted_quantities = (
            self.quantities_conv(board)
            .transpose(1, 3)
            .transpose(1, 2)[:, :, :, 0]
        )  # shape is now batch_size x 48 x 48

        return predicted_types, predicted_resources, predicted_quantities

class UNet(nn.Module):
    def __init__(self, out=20):
        super(UNet, self).__init__()

        self.unet = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=41, out_channels=out, pretrained=False)
        self.types_linear = nn.Linear(out, 13)
        self.resources_linear = nn.Linear(out, 5)
        self.quantities_linear = nn.Linear(out, 1)

    def forward(self, board):
        output = self.unet(board).transpose(1, 3).transpose(1, 2)
        # print(output.size())
        predicted_types = self.types_linear(output)
        predicted_resources = self.resources_linear(output)
        predicted_quantites = self.quantities_linear(output)[:, :, :, 0]
        # print(predicted_types.size())
        return predicted_types, predicted_resources, predicted_quantites

class LRaspp(nn.Module):
    def __init__(self, out=20):
        super(LRaspp, self).__init__()

        self.lraspp = torchvision.models.segmentation.lraspp_mobilenet_v3_large(num_classes=out)
        self.lraspp.backbone['0'][0] = nn.Conv2d(41, 16, 3, stride=2, padding=1, bias=False)
        self.types_linear = nn.Linear(out, 13)
        self.resources_linear = nn.Linear(out, 5)
        self.quantities_linear = nn.Linear(out, 1)

    def forward(self, board):
        output = self.lraspp(board)['out'].transpose(1, 3).transpose(1, 2)
        predicted_types = self.types_linear(output)
        predicted_resources = self.resources_linear(output)
        predicted_quantites = self.quantities_linear(output)[:, :, :, 0]
        return predicted_types, predicted_resources, predicted_quantites