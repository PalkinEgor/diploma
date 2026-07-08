from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.nn as nn


@dataclass
class EncoderOutput:
    """Результат энкодера для батча."""

    e: torch.Tensor
    m: torch.Tensor
    aux_losses: Dict[str, torch.Tensor] = field(default_factory=dict)
    latent: Optional[torch.Tensor] = None
    extra: Dict[str, torch.Tensor] = field(default_factory=dict)

    def stacked(self) -> torch.Tensor:
        return torch.stack([self.e, self.m], dim=1)


class ProtoEncoder(nn.Module, ABC):

    @abstractmethod
    def encode(self, batch: dict) -> EncoderOutput:
        """Закодировать батч инструкций в прото-токены ответа."""
        raise NotImplementedError

    def forward(self, batch: dict) -> EncoderOutput:
        return self.encode(batch)

    @classmethod
    @abstractmethod
    def from_config(cls, config: dict) -> "ProtoEncoder":
        """Собрать энкодер из раздела model.encoder конфига."""
        raise NotImplementedError
