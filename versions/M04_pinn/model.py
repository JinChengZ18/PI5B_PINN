# model.py
import torch
import torch.nn as nn


class SimplePINN(nn.Module):
    """
    M04-PINN:
    在 M02_fcnn_anneal 基础上增加 Laplacian 计算
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

    def laplacian(self, x):
        """
        ∇²T，仅对 x,y,z 三个维度
        """
        T = self.forward(x)

        grads = torch.autograd.grad(
            T,
            x,
            grad_outputs=torch.ones_like(T),
            create_graph=True,
            retain_graph=True,
        )[0]

        lap = 0.0
        for i in range(3):
            grad_i = grads[:, i:i+1]
            grad2 = torch.autograd.grad(
                grad_i,
                x,
                grad_outputs=torch.ones_like(grad_i),
                create_graph=True,
                retain_graph=True,
            )[0][:, i:i+1]
            lap += grad2

        return lap
