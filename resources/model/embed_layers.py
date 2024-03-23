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


class PatchEmbedder(nn.Module):
    def __init__(self, d_model, patch_len, stride, dropout):
        super(PatchEmbedder, self).__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = ReplicationPad1d((0, stride))
        self.value_embedding = TokenEmbedder(patch_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2] * x.shape[3]))
        x = self.value_embedding(x)
        x = self.dropout(x)
        return x, n_vars
