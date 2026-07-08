import torch
import torch.nn as nn
import torch.nn.functional as F

from ...registry import ENCODERS
from .base import EncoderOutput, ProtoEncoder


class _GaussianEncoder(nn.Module):
    """MLP-энкодер домена: вход -> (mu, logvar) в латентном пространстве."""

    def __init__(self, in_dim, latent_dim, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU())
        self.mu = nn.Linear(hidden, latent_dim)
        self.logvar = nn.Linear(hidden, latent_dim)

    def forward(self, x):
        h = self.net(x)
        return self.mu(h), self.logvar(h)


class _Decoder(nn.Module):
    """MLP-декодер домена: латент -> реконструкция входа домена."""

    def __init__(self, latent_dim, out_dim, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim)
        )

    def forward(self, z):
        return self.net(z)


def _reparam(mu, logvar):
    std = torch.exp(0.5 * logvar)
    return mu + torch.randn_like(std) * std


def _kl_to_prior(mu, logvar):
    return 0.5 * torch.sum(torch.exp(logvar) + mu ** 2 - 1.0 - logvar, dim=-1).mean()


def _wasserstein2(mu1, logvar1, mu2, logvar2):
    std1 = torch.exp(0.5 * logvar1)
    std2 = torch.exp(0.5 * logvar2)
    return (((mu1 - mu2) ** 2).sum(-1) + ((std1 - std2) ** 2).sum(-1)).mean()


@ENCODERS.register("cada_vae")
class CadaVaeEncoder(ProtoEncoder):
    """CADA-VAE: согласование пространств инструкции и прото-токена через общий латент."""

    def __init__(self, teacher_dim, proto_dim, latent_dim, hidden=512):
        super().__init__()
        assert proto_dim % 2 == 0, "proto_dim = 2*H (конкатенация e и m)"
        self.H = proto_dim // 2

        self.E_T = _GaussianEncoder(teacher_dim, latent_dim, hidden)
        self.D_T = _Decoder(latent_dim, teacher_dim, hidden)
        self.E_P = _GaussianEncoder(proto_dim, latent_dim, hidden)
        self.D_P = _Decoder(latent_dim, proto_dim, hidden)
        self.teacher_embeddings = {}

    @classmethod
    def from_config(cls, config: dict, teacher_dim=None, proto_dim=None) -> "CadaVaeEncoder":
        if teacher_dim is None or proto_dim is None:
            raise ValueError("CadaVaeEncoder.from_config требует teacher_dim и proto_dim.")
        enc = config["model"]["encoder"]
        return cls(teacher_dim, proto_dim, enc["latent_dim"], enc.get("hidden_dim", 512))

    def _lookup_teacher(self, instructions, device):
        return torch.stack([self.teacher_embeddings[i] for i in instructions]).to(device).float()

    def encode(self, batch: dict) -> EncoderOutput:
        dev = next(self.parameters()).device
        q = self._lookup_teacher(batch["metainfo"]["instructions"], dev)  
        e_t = batch["targets"]["e"].to(dev).float()
        m_t = batch["targets"]["m"].to(dev).float()
        p = torch.cat([e_t, m_t], dim=-1)                                 

        mu_T, logvar_T = self.E_T(q)
        mu_P, logvar_P = self.E_P(p)
        z_T = _reparam(mu_T, logvar_T)
        z_P = _reparam(mu_P, logvar_P)

        recon_q = self.D_T(z_T)     # реконструкция внутри домена VAE_T
        recon_p = self.D_P(z_P)     # реконструкция внутри домена VAE_P
        cross_p = self.D_P(z_T)     # генеративный путь: q -> z_T -> p
        cross_q = self.D_T(z_P)     # обратное согласование: p -> z_P -> q

        aux = {
            "recon": F.mse_loss(recon_q, q) + F.mse_loss(recon_p, p),
            "kl": _kl_to_prior(mu_T, logvar_T) + _kl_to_prior(mu_P, logvar_P),
            "cross_align": F.mse_loss(cross_p, p) + F.mse_loss(cross_q, q),
            "dist_align": _wasserstein2(mu_T, logvar_T, mu_P, logvar_P),
        }

        e = cross_p[:, : self.H]
        m = cross_p[:, self.H:]
        return EncoderOutput(e=e, m=m, aux_losses=aux, latent=z_T)
