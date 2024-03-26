from embed_layers import TokenEmbedder, PatchEmbedder
from NormalizeLayer import NormalizeLayer
from ReprogrammingLayer import ReprogrammingLayer
from FlattenHead import FlattenHead

from transformers import GPT2Config, GPT2Model, GPT2Tokenizer
import torch.nn as nn
from torch import Tensor
import torch


class TimeLLM(nn.Module):
    def __init__(self, configs=None):
        super(TimeLLM, self).__init__()

        # init

        config = GPT2Config.from_pretrained("openai-community/gpt2")
        # config.num_hidden_layers = configs.n_layers
        config.output_attentions = True
        config.output_hidden_states = True

        self._set_llm_model(config)
        self._set_tokenizer()

    def forward(self, x):
        return self.forecast(x)

    def forecast(self, x):
        return x

    def _set_llm_model(self, config):
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


TimeLLM = TimeLLM()