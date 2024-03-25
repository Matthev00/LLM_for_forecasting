import torch.nn as nn
from torch import Tensor
import torch
from embed_layers import TokenEmbedder, PatchEmbedder
from NormalizeLayer import NormalizeLayer
from ReprogrammingLayer import ReprogrammingLayer
from FlattenHead import FlattenHead


class TimeLLM(nn.Module):
    def __init__(self):
        super(TimeLLM, self).__init__()

    def forward(self, x):
        return self.forecast(x)

    def forecast(self, x):
        return x
