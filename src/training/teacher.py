import json
import torch
from sentence_transformers import SentenceTransformer
from ..base import Runnable
from ..data import get_dataset
from ..registry import EXPERIMENTS
from ..runtime import resolve_dtype


class TeacherEmbedder:

    def __init__(self, path, dtype="float32", normalize=True):
        self.model = SentenceTransformer(path, torch_dtype=resolve_dtype(dtype))
        self.normalize = normalize

    @torch.no_grad()
    def encode(self, texts):
        return self.model.encode(
            texts,
            convert_to_tensor=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )

    @classmethod
    def from_config(cls, cfg: dict) -> "TeacherEmbedder":
        return cls(cfg["path"], cfg.get("dtype", "float32"), cfg.get("normalize", True))


@EXPERIMENTS.register("teacher_embeddings")
class TeacherEmbeddings(Runnable):
    """Предпосчёт teacher-эмбеддингов и сохранение в JSON."""

    def __init__(self, embedder, items, key, save_path, batch_size):
        self.embedder = embedder
        self.items = items
        self.key = key
        self.save_path = save_path
        self.batch_size = batch_size

    @classmethod
    def from_config(cls, config: dict) -> "TeacherEmbeddings":
        mode = config["training"].get("mode", "instruction")
        task_type = config["dataset"].get("task_type", "end2end")
        dataset = get_dataset(config["dataset"]["type"], task_type, config["dataset"]["path"])
        items = dataset.texts if mode == "text" else dataset.instructions

        embedder = TeacherEmbedder(
            config["model"]["path"],
            config["model"].get("dtype", "float32"),
            config["training"].get("normalize", True),
        )
        return cls(
            embedder,
            items,
            key=mode,
            save_path=config["logging"]["save_path"],
            batch_size=config["training"]["batch_size"],
        )

    def run(self):
        result = []
        for i in range(0, len(self.items), self.batch_size):
            batch = self.items[i:i + self.batch_size]
            emb = self.embedder.encode(batch)
            for text, vec in zip(batch, emb):
                result.append({self.key: text, "teacher_embedding": vec.float().cpu().tolist()})

        with open(self.save_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
        return result
