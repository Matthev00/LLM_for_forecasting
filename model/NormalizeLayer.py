import torch.nn as nn
from torch import Tensor
import torch


class NormalizeLayer(nn.Module):
    def __init__(
        self,
        num_features: int,
        eps=1e-5,
        affine=False,
        subtract_last=False,
        non_norm=False,
    ):
        super(NormalizeLayer, self).__init__()

        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        self.non_norm = non_norm
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str) -> Tensor:
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        pass

    def _get_statistics(self, x):
        pass

    def _normalize(self, x):
        pass

    def _denormalize(self, x):
        pass
