import torch
import torch.nn as nn
import torch.nn.functional as F


class Latent2Class(nn.Module):
    def __init__(self, cls=1000) -> None:
        super().__init__()
        self.cls = nn.Sequential(
            nn.Linear(2048, 1000),
            nn.Softmax()
        )

    def forward(self, x):
        x = x.permute(1, 0, 2)
        x = x.reshape(x.shape[0], -1)
        prob = self.cls(x)
        return prob
