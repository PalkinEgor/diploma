"""Общий тренерный цикл этапа 2 (одинаковый для всех трёх архитектур).

База отвечает за то, что не зависит от конкретной архитектуры: цикл эпох,
оптимизатор (только энкодер), лосс реконструкции через замороженный декодер
(:class:`proto_tokens.models.decoder.FrozenDecoder`), метрики, логирование и
сохранение обученного энкодера.

Наследники (codebook / cada_vae / contrastive) переопределяют только
:meth:`compute_loss` — как из :class:`~proto_tokens.models.encoders.base.EncoderOutput`
и логитов реконструкции собрать итоговый лосс со своими весами (diversity /
KL+CA+DA / InfoNCE+DBLL) — и :meth:`from_config` (сборка своего энкодера).
"""

from abc import abstractmethod
from collections import defaultdict

import torch
import torch.nn.functional as F

from ...base import Runnable
from ...metrics import Metrics
from ...models.encoders.base import EncoderOutput


class BaseTrainer(Runnable):
    """Общий цикл обучения энкодера ``инструкция -> (e, m)``."""

    def __init__(self, encoder, decoder, dataloader, optimizer, tokenizer, device, config, logger):
        self.encoder = encoder
        self.decoder = decoder
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.tokenizer = tokenizer
        self.device = device
        self.config = config
        self.logger = logger
        self.n_epochs = config["training"]["n_epochs"]

    # ------------------------------------------------------------------ #
    # Общие строительные блоки (переиспользуются наследниками)
    # ------------------------------------------------------------------ #
    def reconstruction_ce(self, logits, labels):
        """CE-лосс реконструкции ответа (паддинг игнорируется)."""
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=self.tokenizer.pad_token_id,
        )

    def decode(self, output: EncoderOutput, answer: dict):
        """Логиты замороженного декодера по прото-токенам энкодера."""
        return self.decoder(
            output.e,
            output.m,
            answer["lengths"],
            answer["attention_mask"].to(self.device),
        )

    @staticmethod
    def token_accuracies(logits, labels, lengths):
        """Список токен-уровневых точностей по примерам батча (по длинам ответов)."""
        pred = logits.argmax(dim=-1)
        accs = []
        for i in range(len(lengths)):
            L = lengths[i]
            accs.append(Metrics.calculate_accuracy(labels[i, :L], pred[i, :L]))
        return accs

    @abstractmethod
    def compute_loss(self, output: EncoderOutput, batch: dict, logits):
        """Итоговый лосс архитектуры.

        Returns:
            Кортеж ``(loss_tensor, logs_dict)``, где ``logs_dict`` — скаляры для
            усреднения по эпохе (например ``{'loss', 'ce_loss', ...}``).
        """
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Цикл
    # ------------------------------------------------------------------ #
    def run(self):
        self.encoder.train()
        self.decoder.train()  # остаётся eval/заморожен (см. FrozenDecoder.train)

        self.logger.info("start training")
        for epoch in range(self.n_epochs):
            agg = defaultdict(float)
            accs = []
            n_batches = 0

            for batch in self.dataloader:
                labels = batch["answer"]["labels"].to(self.device)

                self.optimizer.zero_grad()
                output = self.encoder.encode(batch)
                logits = self.decode(output, batch["answer"])
                loss, logs = self.compute_loss(output, batch, logits)
                loss.backward()
                self.optimizer.step()

                for k, v in logs.items():
                    agg[k] += v
                accs.extend(self.token_accuracies(logits, labels, batch["answer"]["lengths"]))
                n_batches += 1

            summary = "; ".join(f"{k}: {agg[k] / max(n_batches, 1):.4f}" for k in agg)
            self.logger.info(
                "Epoch %d/%d | %s | Accuracy: %.4f",
                epoch + 1, self.n_epochs, summary, sum(accs) / max(len(accs), 1),
            )

        self._save_model()
        self.logger.info("finish training")
        return self.encoder

    def _save_model(self):
        """Сохранить веса энкодера, если задан ``logging.save_model_path``."""
        path = self.config.get("logging", {}).get("save_model_path")
        if path:
            torch.save(self.encoder.state_dict(), path)
            self.logger.info("saved encoder to %s", path)
