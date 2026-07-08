import json

import torch

from ..base import Runnable
from ..registry import EXPERIMENTS
from ..runtime import build_dataloader, load_frozen_causal_lm, load_tokenizer, setup_logger
from ..utils import generate_input


@EXPERIMENTS.register("attention")
class AttentionVisualization(Runnable):
    """Сбор карт внимания к прото-токену e."""

    def __init__(self, model, tokenizer, dataloader, mean_attention, save_path, save_every, logger):
        self.model = model
        self.tokenizer = tokenizer
        self.dataloader = dataloader
        self.mean_attention = mean_attention
        self.save_path = save_path
        self.save_every = save_every
        self.logger = logger
        self.pad_emb = model.get_input_embeddings().weight[tokenizer.pad_token_id]

    @classmethod
    def from_config(cls, config: dict) -> "AttentionVisualization":
        logger = setup_logger(config["logging"].get("save_log_path"))
        tokenizer = load_tokenizer(config["model"]["path"])
        dataloader = build_dataloader(config, tokenizer, task_type="nar")
        model = load_frozen_causal_lm(config["model"]["path"], config["model"]["dtype"])
        # eager-внимание нужно для output_attentions
        if hasattr(model, "set_attn_implementation"):
            model.set_attn_implementation("eager")
        return cls(
            model,
            tokenizer,
            dataloader,
            config["training"].get("mean_attention", False),
            config["logging"]["save_path"],
            config["logging"].get("save_every", 10),
            logger,
        )

    def _save(self, result):
        with open(self.save_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=4)

    def run(self):
        device = self.model.device
        result = []
        for idx, batch in enumerate(self.dataloader):
            tokenized_text = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            lengths = batch["lengths"]
            texts = batch["texts"]
            e_vectors = torch.tensor(batch["e_vectors"], device=device, dtype=self.model.dtype)
            m_vectors = torch.tensor(batch["m_vectors"], device=device, dtype=self.model.dtype)
            B = tokenized_text.size(0)

            vectors = torch.stack([e_vectors, m_vectors], dim=1)
            current_input = generate_input(
                vectors, lengths, tokenized_text.size(1), self.pad_emb, device
            )
            with torch.no_grad():
                outputs = self.model(
                    inputs_embeds=current_input,
                    attention_mask=attention_mask,
                    output_attentions=True,
                )
            attention_weights = outputs.attentions  # кортеж по слоям: (B, heads, T, T)

            for i in range(B):
                attn_e_all_layers = []
                for layer_attn in attention_weights:
                    if self.mean_attention:
                        attn_e = layer_attn[i].mean(dim=0)[:, 0]      # (T,) усреднение по головам
                    else:
                        attn_e = layer_attn[i][:, :, 0]               # (heads, T)
                    attn_e_all_layers.append(attn_e.float().cpu().numpy().tolist())

                token_list = self.tokenizer.convert_ids_to_tokens(
                    tokenized_text[i][: lengths[i]].tolist()
                )
                result.append({
                    "text": texts[i],
                    "tokens": token_list,
                    "lengths": lengths[i],
                    "best_vectors": vectors[i].float().cpu().numpy().tolist(),
                    "attention_e": attn_e_all_layers,
                })

            if (idx + 1) % self.save_every == 0:
                self.logger.info("processed %d/%d batches", idx + 1, len(self.dataloader))
                self._save(result)

        self._save(result)
        return result
