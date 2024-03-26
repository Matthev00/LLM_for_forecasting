import torch.nn as nn
from torch import Tensor
import torch


class ReprogrammingLayer(nn.Module):
    def __init__(
        self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1
    ):
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        self.query = nn.Linear(d_model, d_keys, *n_heads)
        self.key = nn.Linear(d_llm, d_keys * n_heads)
        self.value = nn.Linear(d_llm, d_keys * n_heads)
        self.out = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding) -> Tensor:
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query(target_embedding).view(B, L, H, -1)
        source_embedding = self.key(source_embedding).view(S, H, -1)
        value_embedding = self.value(value_embedding).view(S, H, -1)

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)

        out = out.reshape(B, L, -1)

        return self.out_projection(out)

    def reprogramming(
        self, target_embedding, source_embedding, value_embedding
    ) -> Tensor:
        B, L, H, E = target_embedding.shape

        scale = 1.0 / sqrt(E)

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding
