import torch.nn as nn
from torch import Tensor


class FlattenHead(nn.Module):
    def __init__(self, nf, target_window, head_dropout):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x) -> Tensor:
        return self.dropout(self.linear(self.flatten(x)))
