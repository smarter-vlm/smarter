import torch
from torch import nn


class CLayer(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, im_repr, q_repr):
        return torch.cat([im_repr, q_repr], dim=1)


class QFLayer(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, im_repr, q_repr):
        return torch.cat([im_repr, q_repr], dim=1)
