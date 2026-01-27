# pinn_mvp/model.py
import torch
import torch.nn as nn


class SimplePINN(nn.Module):
    """
    初版最简单 PINN：
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
