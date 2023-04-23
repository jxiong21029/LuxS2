import torch
import torchvision
from torch import nn


class UNet(nn.Module):
    def __init__(self, out=20):
        super(UNet, self).__init__()

        self.unet = torch.hub.load(
            "mateuszbuda/brain-segmentation-pytorch",
            "unet",
            in_channels=41,
            out_channels=out,
            pretrained=False,
        )
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


class LRASPP(nn.Module):
    def __init__(self, out=20):
        super(LRASPP, self).__init__()

        self.lraspp = (
            torchvision.models.segmentation.lraspp_mobilenet_v3_large(
                num_classes=out
            )
        )
        self.lraspp.backbone["0"][0] = nn.Conv2d(
            41, 16, 3, stride=2, padding=1, bias=False
        )
        self.types_linear = nn.Linear(out, 13)
        self.resources_linear = nn.Linear(out, 5)
        self.quantities_linear = nn.Linear(out, 1)

    def forward(self, board):
        output = self.lraspp(board)["out"].transpose(1, 3).transpose(1, 2)
        predicted_types = self.types_linear(output)
        predicted_resources = self.resources_linear(output)
        predicted_quantites = self.quantities_linear(output)[:, :, :, 0]
        return predicted_types, predicted_resources, predicted_quantites
