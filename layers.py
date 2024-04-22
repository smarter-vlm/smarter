import torch
from torch import nn


class QFLayer(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, im_repr, q_repr):
        return torch.cat([im_repr.mean(1), q_repr.mean(1)], dim=1)
