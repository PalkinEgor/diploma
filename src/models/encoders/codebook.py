import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from ...registry import ENCODERS
from ...runtime import resolve_dtype
from .base import EncoderOutput, ProtoEncoder


class GumbelVectorQuantizer(nn.Module):
    """Квантайзер: вектор -> дискретный код из G групп по V вариантов."""

    def __init__(self, dim, G, V, code_books_init, dtype, tau):
        super().__init__()
        assert dim % G == 0, f"dim ({dim}) must be divisible by G ({G})"

        self.G = G
        self.V = V
        self.code_dim = dim // G
        self.tau = tau
        self.proj = nn.Linear(dim, G * V, dtype=dtype)

        centroids = code_books_init.reshape(V, G, dim // G)  # (V, G, d/G)
        centroids = centroids.permute(1, 0, 2)               # (G, V, d/G)
        self.code_books = nn.Parameter(centroids.to(dtype=dtype))

        self.out_proj = nn.Linear(G * self.code_dim, dim, dtype=dtype)

    def forward(self, x):
        B, _ = x.shape

        logits = self.proj(x).view(B, self.G, self.V)                 # (B, G, V)
        probs = F.gumbel_softmax(logits, tau=self.tau, hard=True, dim=-1)
        codes = torch.einsum("bgv,gvd->bgd", probs, self.code_books)  # (B, G, code_dim)
        codes = codes.reshape(B, self.G * self.code_dim)
        out = self.out_proj(codes)

        # diversity loss — энтропийный штраф на равномерность использования книг
        soft_probs = F.softmax(logits, dim=-1)
        avg_probs = soft_probs.mean(dim=0)
        diversity_loss = (avg_probs * torch.log(avg_probs + 1e-7)).sum()
        diversity_loss = diversity_loss / (self.G * self.V)

        return out, diversity_loss


@ENCODERS.register("codebook")
class CodebookEncoder(ProtoEncoder):
    """Энкодер инструкция -> (e, m) на кодовых книгах (Gumbel-Softmax)."""

    def __init__(self, encoder_name, G, V, e_code_books_init, m_code_books_init,
                 tau, dtype, m_vector, mean_pooling):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(encoder_name, torch_dtype=dtype)
        dim = self.encoder.config.hidden_size
        self.e_quantizer = GumbelVectorQuantizer(dim, G, V, e_code_books_init, dtype, tau)
        self.m_vector = m_vector
        self.mean_pooling = mean_pooling
        if m_vector:
            self.m_quantizer = GumbelVectorQuantizer(dim, G, V, m_code_books_init, dtype, tau)
        else:
            self.m_proj = nn.Linear(dim, dim, dtype=dtype)

    @classmethod
    def from_config(cls, config: dict, init=None) -> "CodebookEncoder":
        """Собрать из ``model.encoder`` конфига."""
        if init is None:
            raise ValueError("CodebookEncoder.from_config требует init (CodeBooksInit).")
        enc = config["model"]["encoder"]
        cb = enc["code_books"]
        return cls(
            enc.get("path", config["model"]["path"]),
            cb["G"],
            cb["V"],
            init.e_code_books,
            init.m_code_books,
            cb["tau"],
            resolve_dtype(config["model"]["dtype"]),
            cb["m_vector"],
            enc.get("mean_pooling", False),
        )

    def _pool(self, hidden, attention_mask):
        """Свести последовательность в один вектор: mean-pooling или последний токен."""
        if self.mean_pooling:
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            x = (hidden * mask).sum(dim=1)
            return x / mask.sum(dim=1).clamp(min=1e-6)
        lengths = attention_mask.sum(dim=1) - 1
        return hidden[torch.arange(hidden.size(0), device=hidden.device), lengths]

    def encode(self, batch: dict) -> EncoderOutput:
        dev = next(self.parameters()).device
        input_ids = batch["instruction"]["input_ids"].to(dev)
        attention_mask = batch["instruction"]["attention_mask"].to(dev)

        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        x = self._pool(hidden, attention_mask)

        e, e_diversity = self.e_quantizer(x)
        if self.m_vector:
            m, m_diversity = self.m_quantizer(x)
            diversity = e_diversity + m_diversity
        else:
            m = self.m_proj(x)
            diversity = e_diversity

        return EncoderOutput(e=e, m=m, aux_losses={"diversity": diversity})
