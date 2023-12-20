import torch
import torch.nn as nn


class NN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU()
        )

        self.layer2 = nn.Sequential(
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        x = torch.flatten(x, 1)
        out = self.layer1(x)
        out = self.layer2(out)

        return out
