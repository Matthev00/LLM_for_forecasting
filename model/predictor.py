from embed_layers import PatchEmbedder
from NormalizeLayer import NormalizeLayer
from ReprogrammingLayer import ReprogrammingLayer
from FlattenHead import FlattenHead

from transformers import GPT2Config, GPT2Model, GPT2Tokenizer
import torch.nn as nn
from torch import Tensor


class TimeLLM(nn.Module):
    def __init__(self, configs=None):
        super(TimeLLM, self).__init__()

        self.configs = configs

        self._set_llm_model()
        self._set_tokenizer()
        self._set_pad_token()
        self._freeze_llm()
        self.description = configs.content

        # layers
        self.dropout = nn.Dropout(configs.dropout)

        self.patch_embedder = PatchEmbedder(
            configs.d_model, configs.patch_len, configs.stride, configs.dropout
        )

        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        vocab_size = self.word_embeddings.shape[0]
        num_tokens = 1000
        self.mapping_layer = nn.Linear(vocab_size, num_tokens)

        self.reprogramming_layer = ReprogrammingLayer(
            configs.d_model, configs.n_heads, configs.d_ff, configs.d_llm
        )

        self.patch_nums = int(
            (configs.seq_len - configs.patch_len) / configs.stride + 2
        )
        self.head_nf = configs.d_ff * self.patch_nums

        self.output_projection = FlattenHead(
            configs.enc_in, self.head_nf, configs.pred_len, head_dropout=configs.dropout
        )

        self.normalize_layer = NormalizeLayer(configs.enc_in, affine=False)

    def forward(self, x) -> Tensor:
        return self.forecast(x)

    def forecast(self, x) -> Tensor:
        return x

    def _set_llm_model(self):
        config = GPT2Config.from_pretrained("openai-community/gpt2")
        config.num_hidden_layers = self.configs.n_layers
        config.output_attentions = True
        config.output_hidden_states = True
        try:
            self.llm_model = GPT2Model.from_pretrained(
                "openai-community/gpt2",
                config=config,
                trust_remote_code=True,
                local_files_only=True,
            )
        except EnvironmentError:
            print("Local model files not found. Attempting to download...")
            self.llm_model = GPT2Model.from_pretrained(
                "openai-community/gpt2",
                config=config,
                trust_remote_code=True,
                local_files_only=False,
            )

    def _set_tokenizer(self):
        try:
            self.tokenizer = GPT2Tokenizer.from_pretrained(
                "openai-community/gpt2",
                trust_remote_code=True,
                local_files_only=True,
            )
        except EnvironmentError:
            print("Local tokenizer files not found. Attempting to download...")
            self.tokenizer = GPT2Tokenizer.from_pretrained(
                "openai-community/gpt2",
                trust_remote_code=True,
                local_files_only=False,
            )

    def _set_pad_token(self):
        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = "[PAD]"
            self.tokenizer.add_special_tokens({"pad_token": pad_token})
            self.tokenizer.pad_token = pad_token

    def _freeze_llm(self):
        for param in self.llm_model.parameters():
            param.requires_grad = False


TimeLLM = TimeLLM()
