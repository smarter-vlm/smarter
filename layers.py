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
        self.intermediate = QFIntermediate()

    def forward(self, im_repr, q_repr):
        x = torch.cat([im_repr, q_repr], dim=1)
        x = self.intermediate(x)
        return x

class QFIntermediate(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(3*768, 768) # TODO DR shapes/hidden sizes
        self.intermediate_act_fn = nn.GELU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states