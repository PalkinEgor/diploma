import json
import math
import random

import numpy as np
import torch

from ..base import Runnable
from ..metrics import Metrics
from ..registry import EXPERIMENTS
from ..runtime import build_dataloader, load_frozen_causal_lm, load_tokenizer, setup_logger
from ..utils import generate_input

DEFAULT_ALPHAS = [0.00, 0.05, 0.10, 0.20, 0.50, 1.00]
DEFAULT_NOISE_TYPES = ["gaussian", "uniform", "sinusoidal", "exponential"]


def get_noise(size, alpha, noise_type, ref_vector):
    """Вектор шума нужного типа, нормированный до alpha * ||ref_vector||."""
    if noise_type == "gaussian":
        noise = np.random.normal(size=size)
    elif noise_type == "uniform":
        noise = np.random.uniform(-1.0, 1.0, size)
    elif noise_type == "exponential":
        noise = np.random.exponential(size=size)
        noise *= np.random.choice([-1, 1], size=size)
    elif noise_type == "sinusoidal":
        k = random.choice(range(4, 33))
        freq = 2 * math.pi * k / size
        phase = np.random.uniform(0, 2 * math.pi, 1)[0]
        noise = np.sin(freq * np.arange(size) + phase)
    else:
        noise = np.random.normal(size=size)

    return (noise / np.linalg.norm(noise)) * np.linalg.norm(ref_vector) * alpha


@EXPERIMENTS.register("noise")
class NoiseExperiment(Runnable):
    
    def __init__(self, model, tokenizer, dataloader, alphas, noise_types, save_path, logger):
        self.model = model
        self.tokenizer = tokenizer
        self.dataloader = dataloader
        self.alphas = alphas
        self.noise_types = noise_types
        self.save_path = save_path
        self.logger = logger
        self.pad_emb = model.get_input_embeddings().weight[tokenizer.pad_token_id]

    @classmethod
    def from_config(cls, config: dict) -> "NoiseExperiment":
        logger = setup_logger(config["logging"].get("save_log_path"))
        np.random.seed(config["training"].get("seed", 42))
        tokenizer = load_tokenizer(config["model"]["path"])
        dataloader = build_dataloader(config, tokenizer, task_type="nar")  # NoiseDataset + collate_noise
        model = load_frozen_causal_lm(config["model"]["path"], config["model"]["dtype"])
        return cls(
            model,
            tokenizer,
            dataloader,
            config["training"].get("alphas", DEFAULT_ALPHAS),
            config["training"].get("noise_types", DEFAULT_NOISE_TYPES),
            config["logging"]["save_path"],
            logger,
        )

    def run(self):
        device = self.model.device
        n = len(self.dataloader.dataset)
        result = []
        total = len(self.alphas) * len(self.noise_types)
        done = 0

        for alpha in self.alphas:
            for noise_type in self.noise_types:
                acc_sum = seq_sum = prefix_sum = 0.0
                for batch in self.dataloader:
                    tokenized_text = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    lengths = batch["lengths"]
                    e_vectors = torch.tensor(batch["e_vectors"], device=device, dtype=self.model.dtype)
                    m_vectors = torch.tensor(batch["m_vectors"], device=device, dtype=self.model.dtype)
                    B = tokenized_text.size(0)

                    e_noisy = e_vectors.clone()
                    for i in range(B):
                        noise = get_noise(
                            e_vectors.shape[-1], alpha, noise_type,
                            e_vectors[i].detach().cpu().float().numpy(),
                        )
                        e_noisy[i] += torch.tensor(noise, device=device, dtype=self.model.dtype)

                    vectors = torch.stack([e_noisy, m_vectors], dim=1)
                    current_input = generate_input(
                        vectors, lengths, tokenized_text.size(1), self.pad_emb, device
                    )
                    with torch.no_grad():
                        logits = self.model(
                            inputs_embeds=current_input, attention_mask=attention_mask
                        ).logits
                        pred = logits.argmax(dim=-1)

                    for i in range(B):
                        L = lengths[i]
                        p, t = pred[i, :L], tokenized_text[i, :L]
                        acc_sum += Metrics.calculate_accuracy(t, p)
                        seq_sum += Metrics.rel_correct_prefix_len(t, p)
                        prefix_sum += Metrics.correct_prefix_len(t, p)

                result.append({
                    "alpha": alpha,
                    "noise": noise_type,
                    "accuracy": acc_sum / n,
                    "seq_accuracy": seq_sum / n,
                    "correct_prefix_length": prefix_sum / n,
                })
                done += 1
                self.logger.info("Processed %d/%d (alpha=%s, %s)", done, total, alpha, noise_type)

        with open(self.save_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
        return result
