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
        self.dense = nn.Linear(256, 256)  # TODO DR shapes/hidden sizes
        self.intermediate_act_fn = nn.GELU()
        self.layer_norm = nn.LayerNorm(256, eps=1e-12)
        self.dropout = nn.Dropout(0.1)
        self.dense_final = nn.Linear(256, 256)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = self.dense(hidden_states)
        x = self.intermediate_act_fn(x)

        x = self.dense_final(x)
        x = self.dropout(x)
        x = self.layer_norm(hidden_states + x)
        return x
