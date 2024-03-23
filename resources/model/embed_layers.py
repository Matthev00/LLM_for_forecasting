from torch import nn, Tensor
import torch


class ReplicationPad1d(nn.Module):
    def __init__(self, padding) -> None:
        super(ReplicationPad1d, self).__init__()
        self.padding = padding

    def forward(self, x: Tensor) -> Tensor:
        replicate_padding = input[:, :, -1].unsqueeze(-1).repeat(1, 1, self.padding[-1])
        output = torch.cat([x, replicate_padding], dim=-1)
        return output


class TokenEmbedder(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedder, self).__init__()
        self.tokenconv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=3,
            padding=1,
            padding_mode="circular",
            bias=False,
        )
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )

    def forward(self, x: Tensor) -> Tensor:
        return self.tokenconv(x.permute(0, 2, 1)).transpose(1, 2)
