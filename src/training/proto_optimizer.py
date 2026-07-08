import json
import torch
from ..base import Runnable
from ..metrics import Metrics
from ..registry import EXPERIMENTS
from ..runtime import (
    DEFAULT_OPT,
    build_dataloader,
    load_frozen_causal_lm,
    load_tokenizer,
    setup_logger,
)
from ..utils import generate_input


class NARfit:
    
    def __init__(self, model, tokenizer, device, hyperparams, regularizers=None, teacher_embedder=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.hyperparams = hyperparams
        self.regularizers = regularizers or []
        self.teacher_embedder = teacher_embedder
        self.pad_emb = model.get_input_embeddings().weight[tokenizer.pad_token_id].to(self.device)

    def train_batch(self, batch, maxiter, threshold):
        # Парсинг батча
        tokenized_text = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        lengths = batch["lengths"]
        labels = batch["labels"].to(self.device)

        # Teacher-эмбеддинги текстов батча — считаются один раз (тексты фиксированы).
        teacher_emb = None
        if self.regularizers and self.teacher_embedder is not None:
            teacher_emb = self.teacher_embedder.encode(batch["metainfo"]["texts"]).to(self.device)

        # Создание обучаемых векторов e и m
        B = tokenized_text.shape[0]
        vectors = torch.nn.Parameter(
            torch.randn(
                B, 2, self.model.config.hidden_size,
                device=self.device,
                dtype=self.model.dtype,
            )
        )
        optimizer = torch.optim.AdamW(
            [vectors],
            lr=self.hyperparams["lr"],
            betas=self.hyperparams["betas"],
            weight_decay=self.hyperparams["weight_decay"],
        )

        # Запускаем maxiter итераций оптимизации
        max_accuracy = [0.0] * B
        best_vectors = [None] * B
        last_iter = [None] * B
        for iteration in range(maxiter):
            optimizer.zero_grad()

            current_input = generate_input(
                vectors, lengths, tokenized_text.size(1), self.pad_emb, self.device
            )
            logits = self.model(inputs_embeds=current_input, attention_mask=attention_mask).logits
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=self.tokenizer.pad_token_id,
            )

            # Регуляризация пространства прото-токенов (по вектору e).
            if teacher_emb is not None:
                e_proto = vectors[:, 0, :]
                reg_total = sum(reg.weight * reg(e_proto, teacher_emb) for reg in self.regularizers)
                loss = loss + reg_total.to(loss.dtype)

            pred = logits.argmax(dim=-1)

            # Считаем метрики и сохраняем лучшие вектора
            for i in range(B):
                current_len = lengths[i]
                current_pred = pred[i, :current_len]
                current_labels = labels[i, :current_len]

                accuracy = Metrics.calculate_accuracy(current_labels, current_pred)
                if accuracy > max_accuracy[i]:
                    max_accuracy[i] = accuracy
                    best_vectors[i] = vectors[i].detach().clone()

                if last_iter[i] is None and accuracy >= threshold:
                    last_iter[i] = iteration

            # Пропускаем остаток итераций если батч обучен
            if all(a >= threshold for a in max_accuracy):
                break

            loss.backward()
            optimizer.step()

        for i in range(B):
            if last_iter[i] is None:
                last_iter[i] = maxiter

        return max_accuracy, best_vectors, last_iter, B


@EXPERIMENTS.register("proto_fit")
class ProtoOptimizer(Runnable):

    def __init__(self, model, tokenizer, dataloader, engine, config, logger):
        self.model = model
        self.tokenizer = tokenizer
        self.dataloader = dataloader
        self.engine = engine
        self.config = config
        self.logger = logger

    @classmethod
    def from_config(cls, config: dict) -> "ProtoOptimizer":
        logger = setup_logger(config["logging"]["save_log_path"])
        tokenizer = load_tokenizer(config["model"]["path"])
        dataloader = build_dataloader(config, tokenizer, task_type="nar")
        model = load_frozen_causal_lm(config["model"]["path"], config["model"]["dtype"])

        opt = dict(DEFAULT_OPT)
        opt.update(config["training"].get("optimizer", {}))
        if isinstance(opt["betas"], list):
            opt["betas"] = tuple(opt["betas"])

        # Регуляризация пространства прото-токенов (опционально).
        regularizers = []
        teacher_embedder = None
        if config.get("regularizers"):
            from .regularizers import build_regularizers
            from .teacher import TeacherEmbedder

            regularizers = build_regularizers(config["regularizers"])
            teacher_embedder = TeacherEmbedder.from_config(config["teacher"])
            logger.info(
                "regularizers: %s", [type(r).__name__ for r in regularizers]
            )

        engine = NARfit(model, tokenizer, model.device, opt, regularizers, teacher_embedder)
        return cls(model, tokenizer, dataloader, engine, config, logger)

    def _save(self, result):
        with open(self.config["logging"]["save_path"], "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=4)

    def run(self):
        maxiter = self.config["training"]["maxiter"]
        threshold = self.config["training"]["threshold"]
        save_every = self.config["logging"]["save_every"]

        result = []
        total = 0
        success = 0
        self.logger.info("start training")
        for idx, batch in enumerate(self.dataloader):
            accuracy, best_vectors, last_iter, B = self.engine.train_batch(batch, maxiter, threshold)

            total += B
            success += sum(a >= threshold for a in accuracy)
            avg_acc = sum(accuracy) / B
            avg_iter = sum(last_iter) / B
            self.logger.info(
                f"Processed [{idx + 1}/{len(self.dataloader)}] "
                f"Success rate {success}/{total} "
                f"Average accuracy {avg_acc} "
                f"Average last iteration {avg_iter}"
            )

            for i in range(B):
                item = {k: v[i] for k, v in batch["metainfo"].items()}
                item["accuracy"] = accuracy[i]
                item["last_iter"] = last_iter[i]
                item["best_vectors"] = best_vectors[i].float().cpu().numpy().tolist()
                result.append(item)

            if (idx + 1) % save_every == 0:
                self.logger.info("saving checkpoint...")
                self._save(result)

        self.logger.info("final save...")
        self._save(result)
        return result
