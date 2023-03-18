import torch
import torch.nn as nn
from torchvision import models, transforms


class Model(nn.Module):
    def __init__(self, resnet):
        super(Model, self).__init__()

        self.resnet = resnet
        self.fc = nn.Linear(1000, 2)

    def forward(self, x):
        return self.fc(self.resnet(x))
