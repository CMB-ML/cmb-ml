import torch
import torch.nn as nn


class LinearCombLayer(nn.Module):
    def __init__(self):
        # TODO: Rename "kernel" to "weight"
        # self.weight is what Petroff labelled the 'kernel'; it's just a scalar value
        super().__init__()
        self.kernel = nn.Parameter(torch.ones(1), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, x):
        x = self.kernel * x + self.bias
        return x
