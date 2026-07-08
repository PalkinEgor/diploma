"""Тренер архитектуры №2 (CADA-VAE).

Лосс = взвешенные 4 слагаемых CADA-VAE из ``EncoderOutput.aux_losses``
(recon / kl / cross_align / dist_align) + грунтующая CE-реконструкция через
замороженный декодер (чтобы генеративный путь ``q -> z_T -> D_P -> (e, m)``
действительно декодировался в текст).

Teacher-эмбеддинги инструкций (вход ``VAE_T``) предпосчитываются один раз при
сборке (:class:`proto_tokens.training.teacher.TeacherEmbedder`, режим instruction)
и кладутся в энкодер словарём ``инструкция -> вектор``.
"""

import torch

from ...models.decoder import FrozenDecoder
from ...models.encoders.cada_vae import CadaVaeEncoder
from ...registry import EXPERIMENTS
from ...runtime import build_adamw, build_dataloader, get_device, load_tokenizer, setup_logger
from ..teacher import TeacherEmbedder
from .base import BaseTrainer


@EXPERIMENTS.register("cada_vae")
class CadaVaeTrainer(BaseTrainer):
    """Обучение CADA-VAE-энкодера: 4 лосса согласования + CE-реконструкция."""

    def compute_loss(self, output, batch, logits):
        w = self.config["training"]["loss_weights"]
        aux = output.aux_losses

        labels = batch["answer"]["labels"].to(self.device)
        ce = self.reconstruction_ce(logits, labels)

        total = w.get("ce", 1.0) * ce
        for name in ("recon", "kl", "cross_align", "dist_align"):
            total = total + w.get(name, 1.0) * aux[name]

        logs = {"loss": None, "ce_loss": ce.item()}
        logs.update({name: aux[name].item() for name in ("recon", "kl", "cross_align", "dist_align")})
        logs["loss"] = total.item()
        return total, logs

    @classmethod
    def from_config(cls, config: dict) -> "CadaVaeTrainer":
        logger = setup_logger(config["logging"]["save_log_path"])
        tokenizer = load_tokenizer(config["model"]["path"])
        device = get_device()

        dataloader = build_dataloader(config, tokenizer, task_type="end2end")

        # Предпосчёт teacher-эмбеддингов инструкций (один раз, кэш инструкция -> вектор).
        logger.info("precomputing teacher embeddings for instructions...")
        embedder = TeacherEmbedder.from_config(config["teacher"])
        instructions = list(dict.fromkeys(dataloader.dataset.instructions))
        embs = embedder.encode(instructions)
        teacher_embeddings = {
            ins: embs[i].detach().cpu().float() for i, ins in enumerate(instructions)
        }
        teacher_dim = embs.shape[-1]

        decoder = FrozenDecoder.from_config(config, tokenizer).to(device)
        proto_dim = 2 * decoder.decoder.config.hidden_size  # [e; m]

        encoder = CadaVaeEncoder.from_config(config, teacher_dim=teacher_dim, proto_dim=proto_dim)
        encoder.teacher_embeddings = teacher_embeddings
        encoder = encoder.to(device)

        optimizer = build_adamw(encoder.parameters(), config["training"])
        return cls(encoder, decoder, dataloader, optimizer, tokenizer, device, config, logger)
