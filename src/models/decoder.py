import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from ..utils import generate_input


class FrozenDecoder(nn.Module):
    """Замороженная LLM, декодирующая (e, m) в логиты за один проход."""

    def __init__(self, decoder_name, tokenizer, dtype):
        super().__init__()
        self.decoder = AutoModelForCausalLM.from_pretrained(decoder_name, torch_dtype=dtype)
        self.pad_token_id = tokenizer.pad_token_id
        self._freeze()

    @classmethod
    def from_config(cls, config: dict, tokenizer) -> "FrozenDecoder":
        from ..runtime import resolve_dtype

        return cls(config["model"]["path"], tokenizer, resolve_dtype(config["model"]["dtype"]))

    @property
    def pad_emb(self):
        return self.decoder.get_input_embeddings().weight[self.pad_token_id]

    def _freeze(self):
        for p in self.decoder.parameters():
            p.requires_grad = False
        self.decoder.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self._freeze()
        return self

    def forward(self, e, m, answer_lengths, answer_attention_mask):
        vectors = torch.stack([e, m], dim=1).to(dtype=self.decoder.dtype)
        max_len = answer_attention_mask.size(1)
        current_input = generate_input(
            vectors, answer_lengths, max_len, self.pad_emb, vectors.device
        )
        return self.decoder(
            inputs_embeds=current_input, attention_mask=answer_attention_mask
        ).logits
