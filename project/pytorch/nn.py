import torch
import torch.nn as nn


class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(28 * 28, 28 * 14),
            nn.ReLU()
        )

        self.layer2 = nn.Sequential(
            nn.Linear(28 * 14, 10)
        )

    def forward(self, x):
        x = torch.flatten(x, 1)
        out = self.layer1(x)
        out = self.layer2(out)

        return out
