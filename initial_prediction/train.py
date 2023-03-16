import torch
from torch.utils.data import DataLoader
from torch import optim
import torch.nn as nn
from torchvision import models
from loader import StateWinners
from model import Model

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)

def train(dataloader, network, lr=0.0001, weight_decay=0, epochs=10):
    network.train()
    optimizer = optim.Adam(network.parameters(), lr=lr, weight_decay=weight_decay)
    loss = nn.CrossEntropyLoss()
    network.to(device)

    for i in range(epochs):
        total = 0
        for example in dataloader:
            network.zero_grad()
            initial_states, winners = example
            output = network(initial_states.to(device))

            l = loss(output, winners.to(device))
            total += l
            l.backward()
            optimizer.step()
        print(total)

def accuray(dataloader, network):
    network.eval()
    correct, total = 0, 0
    for example in dataloader:
        initial_states, winners = example
        predict = torch.argmax(network(initial_states.to(device)), 1)
        for i in range(predict.size()[0]):
            if predict[i] == winners[i]:
                correct += 1
            total += 1
    return correct / total

resnet = models.resnet18(pretrained=False)
network = Model(resnet)

dataset = StateWinners("replays_baseline")
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
train_dataloader, test_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True), DataLoader(test_dataset, batch_size=16, shuffle=True)

train(train_dataloader, network)
print(accuray(test_dataloader, network))

torch.save(resnet.state_dict(), 'saved/resnet_new.pth')
torch.save(network.fc.state_dict(), 'saved/linear_new.bin')