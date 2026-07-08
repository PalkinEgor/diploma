"""Тренер архитектуры №1 (кодовые книги + Gumbel-Softmax).

Лосс = CE-реконструкция + ``diversity_loss_weight * diversity`` (энтропийный
штраф на равномерность использования кодовых книг). Перед обучением кодовые
книги k-means-инициализируются из подобранных NAR-векторов
(:class:`proto_tokens.training.init_codebooks.CodeBooksInit`).

Портировано из ``legacy/new_proj/trainer.py`` (``CodeBookFit``) и entry-point
``legacy/new_proj/entry_points/code_book_train.py`` под общий
:class:`~proto_tokens.training.trainers.base.BaseTrainer`.
"""

from ...data import get_dataset
from ...models.decoder import FrozenDecoder
from ...models.encoders.codebook import CodebookEncoder
from ...registry import EXPERIMENTS
from ...runtime import (
    build_adamw,
    build_dataloader,
    get_device,
    load_tokenizer,
    resolve_dtype,
    setup_logger,
)
from ..init_codebooks import CodeBooksInit
from .base import BaseTrainer


@EXPERIMENTS.register("codebook")
class CodebookTrainer(BaseTrainer):
    """Обучение codebook-энкодера: CE-реконструкция + diversity."""

    def compute_loss(self, output, batch, logits):
        labels = batch["answer"]["labels"].to(self.device)
        ce = self.reconstruction_ce(logits, labels)
        diversity = output.aux_losses["diversity"]
        weight = self.config["training"]["diversity_loss_weight"]
        loss = ce + weight * diversity
        return loss, {
            "loss": loss.item(),
            "ce_loss": ce.item(),
            "diversity_loss": diversity.item(),
        }

    @classmethod
    def from_config(cls, config: dict) -> "CodebookTrainer":
        logger = setup_logger(config["logging"]["save_log_path"])
        tokenizer = load_tokenizer(config["model"]["path"])
        device = get_device()

        # K-means инициализация кодовых книг из подобранных NAR-векторов.
        dataset = get_dataset(config["dataset"]["type"], "end2end", config["dataset"]["path"])
        cb = config["model"]["encoder"]["code_books"]
        init = CodeBooksInit(
            dataset,
            cb["V"],
            config["training"]["seed"],
            resolve_dtype(config["model"]["dtype"]),
            norm=cb.get("normalize", False),
        )
        init.init_codebooks()

        encoder = CodebookEncoder.from_config(config, init).to(device)
        decoder = FrozenDecoder.from_config(config, tokenizer).to(device)
        optimizer = build_adamw(encoder.parameters(), config["training"])

        dataloader = build_dataloader(config, tokenizer, task_type="end2end")
        return cls(encoder, decoder, dataloader, optimizer, tokenizer, device, config, logger)
