# model.py
import torch
import torch.nn as nn
from M03_fcnn_fourier.fourier import FourierFeatureEncoding



class SimpleFCNN(nn.Module):
    """
    初版最简单 FCNN:
    7 -> 128 -> 128 -> 1
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(7, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
        )
    def forward(self, x):
        return self.net(x)


class SimpleFourier(nn.Module):
    """
    FCNN + Fourier:
    7 -> 128 -> 128 -> 1
    """
    def __init__(self):
        super().__init__()
        self.fourier = FourierFeatureEncoding(
            in_dim=7,
            num_frequencies=64,
            scale=10.0
        )
        self.net = nn.Sequential(
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 1)
        )
    def forward(self, x):
        x = self.fourier(x)
        return self.net(x)

