import torch
from torch import nn


class HuberLoss(nn.Module):
    def __init__(self, delta=0.1):
        super(HuberLoss, self).__init__()
        self.delta = delta

    def forward(self, pred, target):
        error = pred - target
        abs_error = torch.abs(error)
        loss = torch.where(
            abs_error <= self.delta,
            0.5 * error ** 2,
            self.delta * (abs_error - 0.5 * self.delta),
        )
        return torch.mean(loss)
