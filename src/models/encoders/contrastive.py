import torch
import torch.nn as nn
from transformers import AutoModel

from ...registry import ENCODERS
from ...runtime import resolve_dtype
from .base import EncoderOutput, ProtoEncoder


@ENCODERS.register("contrastive")
class ContrastiveEncoder(ProtoEncoder):

    def __init__(self, encoder_name, output_dim, dtype, pooling="last"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name, torch_dtype=dtype)
        hidden = self.encoder.config.hidden_size
        self.pooling = pooling

        self.e_proj = nn.Linear(hidden, output_dim, dtype=dtype)
        self.m_proj = nn.Linear(hidden, output_dim, dtype=dtype)
        self.mu = nn.Linear(hidden, 1, dtype=dtype)      # среднее распределения длины
        self.std = nn.Linear(hidden, 1, dtype=dtype)     # лог-дисперсия распределения длины

    @classmethod
    def from_config(cls, config: dict, output_dim=None) -> "ContrastiveEncoder":
        """Собрать из model.encoder конфига."""
        if output_dim is None:
            raise ValueError("ContrastiveEncoder.from_config требует output_dim (H декодера).")
        enc = config["model"]["encoder"]
        return cls(
            enc.get("path", config["model"]["path"]),
            output_dim,
            resolve_dtype(config["model"]["dtype"]),
            pooling=enc.get("pooling", "last"),
        )

    def _pool(self, hidden, attention_mask):
        """Свести последовательность в один вектор: mean / last / cls."""
        if self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            x = (hidden * mask).sum(dim=1)
            return x / mask.sum(dim=1).clamp(min=1e-6)
        if self.pooling == "cls":
            return hidden[:, 0, :]
        lengths = attention_mask.sum(dim=1) - 1
        return hidden[torch.arange(hidden.size(0), device=hidden.device), lengths]

    def encode(self, batch: dict) -> EncoderOutput:
        dev = next(self.parameters()).device
        input_ids = batch["instruction"]["input_ids"].to(dev)
        attention_mask = batch["instruction"]["attention_mask"].to(dev)

        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        x = self._pool(hidden, attention_mask)

        e = self.e_proj(x)
        m = self.m_proj(x)
        mu = self.mu(x)
        std = self.std(x)
        return EncoderOutput(e=e, m=m, extra={"mu": mu, "std": std})
