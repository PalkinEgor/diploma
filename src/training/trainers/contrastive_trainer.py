"""Тренер архитектуры №3 (контрастивная регрессия континуальных прото-токенов).

Лосс = CE-реконструкция (через замороженный декодер) + контрастивное
восстановление прото-токенов против целей этапа 1 (InfoNCE и/или DBLL) +
предсказание длины ответа (gaussian). Веса слагаемых — в ``training.loss_weights``.

Контрастивные функции портированы из ``legacy/train_encoder.py`` (``info_nce_loss``,
``dbll``, ``gaussian_loss``); ``dbll`` векторизован (эквивалентно поэлементному
BCE по матрице расстояний), ``gaussian_loss`` приведён к согласованным формам
``(B, 1)`` (в legacy было непреднамеренное броадкастинг-расширение).
"""

import torch
import torch.nn.functional as F

from ...models.decoder import FrozenDecoder
from ...models.encoders.contrastive import ContrastiveEncoder
from ...registry import EXPERIMENTS
from ...runtime import build_adamw, build_dataloader, get_device, load_tokenizer, setup_logger
from .base import BaseTrainer


def info_nce_loss(e_pred, m_pred, e_target, m_target, temperature=0.07):
    """InfoNCE по батчу: позитив — совпадающий индекс, негативы — остальные."""
    e_pred = F.normalize(e_pred, dim=-1)
    m_pred = F.normalize(m_pred, dim=-1)
    e_target = F.normalize(e_target, dim=-1)
    m_target = F.normalize(m_target, dim=-1)

    e_sim = (e_pred @ e_target.T) / temperature
    m_sim = (m_pred @ m_target.T) / temperature
    positives = torch.arange(e_pred.size(0), device=e_pred.device)

    e_loss = F.cross_entropy(e_sim, positives)
    m_loss = F.cross_entropy(m_sim, positives)
    return 0.5 * (e_loss + m_loss)


def dbll(e_pred, m_pred, e_target, m_target, margin=1.0):
    """Distance-based logistic loss (векторизованный эквивалент legacy-цикла)."""
    B = e_pred.size(0)
    identity = torch.eye(B, device=e_pred.device)
    scale = 1.0 + torch.exp(torch.tensor(-margin, device=e_pred.device))

    D_e = torch.cdist(e_pred, e_target)                       # (B, B) евклид
    pred_e = scale / (1.0 + torch.exp(D_e - margin))
    e_loss = F.binary_cross_entropy(pred_e, identity)

    D_m = torch.cdist(m_pred, m_target)
    pred_m = scale / (1.0 + torch.exp(D_m - margin))
    m_loss = F.binary_cross_entropy(pred_m, identity)

    return 0.5 * (e_loss + m_loss)


def gaussian_loss(target, mu, std):
    """NLL гауссианы для длины (``std`` — лог-дисперсия), формы ``(B, 1)``."""
    return (0.5 * (std + ((target - mu) ** 2) / torch.exp(std))).mean()


@EXPERIMENTS.register("contrastive")
class ContrastiveTrainer(BaseTrainer):
    """Обучение контрастивного энкодера: реконструкция + восстановление (e, m) + длина."""

    def compute_loss(self, output, batch, logits):
        w = self.config["training"]["loss_weights"]
        logs = {}

        labels = batch["answer"]["labels"].to(self.device)
        ce = self.reconstruction_ce(logits, labels)
        total = w.get("reconstruction", 1.0) * ce
        logs["ce_loss"] = ce.item()

        e_pred, m_pred = output.e.float(), output.m.float()
        e_t = batch["targets"]["e"].to(self.device).float()
        m_t = batch["targets"]["m"].to(self.device).float()

        if w.get("infonce", 0) > 0:
            info = info_nce_loss(e_pred, m_pred, e_t, m_t)
            total = total + w["infonce"] * info
            logs["infonce"] = info.item()

        if w.get("dbll", 0) > 0:
            d = dbll(e_pred, m_pred, e_t, m_t)
            total = total + w["dbll"] * d
            logs["dbll"] = d.item()

        if w.get("length", 0) > 0:
            lengths = torch.tensor(
                batch["answer"]["lengths"], device=self.device, dtype=torch.float32
            ).unsqueeze(1)
            g = gaussian_loss(lengths, output.extra["mu"].float(), output.extra["std"].float())
            total = total + w["length"] * g
            logs["length"] = g.item()

        logs["loss"] = total.item()
        return total, logs

    @classmethod
    def from_config(cls, config: dict) -> "ContrastiveTrainer":
        logger = setup_logger(config["logging"]["save_log_path"])
        tokenizer = load_tokenizer(config["model"]["path"])
        device = get_device()

        decoder = FrozenDecoder.from_config(config, tokenizer).to(device)
        output_dim = decoder.decoder.config.hidden_size  # прото-токен = скрытый размер декодера
        encoder = ContrastiveEncoder.from_config(config, output_dim=output_dim).to(device)
        optimizer = build_adamw(encoder.parameters(), config["training"])

        dataloader = build_dataloader(config, tokenizer, task_type="end2end")
        return cls(encoder, decoder, dataloader, optimizer, tokenizer, device, config, logger)
