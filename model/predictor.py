from embed_layers import PatchEmbedder
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

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec) -> Tensor:
        return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec) -> Tensor:
        x_enc = self.normalize_layer(x_enc, mode="norm")

        B, T, N = x_enc.size()
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
        prompt = self._generate_prompt(x_enc)
        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()


    def _calculate_lags(x: Tensor) -> Tensor:
        return x

    def _generate_prompt(self, x_enc) -> List[str]:

        min_values = torch.min(x_enc, dim=1)[0]
        max_value = torch.max(x_enc, dim=1)[0]
        medians = torch.max(x_enc, dim=1).values
        lags = self._calculate_lags(x_enc)
        trends = x_enc.diff(dim=1).sum(dim=1)

        prompt = []
        for batch in range(x_enc.size[0]):
            min_values_str = str(min_values[batch].tolist()[0])
            max_values_str = str(max_value[batch].tolist()[0])
            medians_str = str(medians[batch].tolist()[0])
            lags_str = str(lags[batch].tolist())
            prompt_ = (
                f"<|start_prompt|>Dataset description: {self.description}"
                f"Task description: forecast the next {str(self.configs.pred_len)} steps given the previous {str(self.seq_len)} steps information; "
                "Input statistics: "
                f"min value {min_values_str}, "
                f"max value {max_values_str}, "
                f"median value {medians_str}, "
                f"the trend of input is {'upward' if trends[batch] > 0 else 'downward'}, "
                f"top 5 lags are : {lags_str}<|<end_prompt>|>"
            )
            prompt.append(prompt_)
        return prompt

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
