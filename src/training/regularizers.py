from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F

from ..registry import REGULARIZERS


class Regularizer(ABC):

    def __init__(self, weight: float = 1.0):
        self.weight = weight

    @abstractmethod
    def __call__(self, proto: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @classmethod
    def from_config(cls, config: dict) -> "Regularizer":
        return cls(**{k: v for k, v in config.items() if k != "name"})


@REGULARIZERS.register("cosine_anchor")
class CosineAnchorRegularizer(Regularizer):

    def __init__(self, weight: float = 1.0, seed: int = 0):
        super().__init__(weight)
        self.seed = seed
        self._proj = None

    def _projection(self, D, H, device):
        if self._proj is None or tuple(self._proj.shape) != (D, H):
            g = torch.Generator().manual_seed(self.seed)
            P = torch.randn(D, H, generator=g) / (D ** 0.5)
            self._proj = P.to(device=device, dtype=torch.float32)
        return self._proj

    def __call__(self, proto, teacher):
        e = proto.float()
        t = teacher.float()
        P = self._projection(t.shape[-1], e.shape[-1], e.device)
        t_proj = t @ P
        cos = F.cosine_similarity(e, t_proj, dim=-1)
        return (1.0 - cos).mean()


@REGULARIZERS.register("relation_distillation")
class RelationDistillationRegularizer(Regularizer):
    """Дистилляция попарных отношений: MSE между матрицами косинусной близости."""

    def __call__(self, proto, teacher):
        e = F.normalize(proto.float(), dim=-1)
        t = F.normalize(teacher.float(), dim=-1)
        Se = e @ e.t()
        St = t @ t.t()
        return F.mse_loss(Se, St)


def build_regularizers(configs):
    regs = []
    for cfg in configs or []:
        reg_cls = REGULARIZERS.get(cfg["name"])
        regs.append(reg_cls.from_config(cfg))
    return regs
