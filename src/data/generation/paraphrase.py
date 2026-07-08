import json
import os

from ...base import Runnable
from ...registry import EXPERIMENTS

AUGMENTATIONS = ["shift", "orfo", "typo", "delete", "insert", "multiply", "swap"]

_PARAPHRASE_PROMPT = """
You are a helpful assistant that generates high-quality paraphrases.
Paraphrases must preserve the original meaning and factual content.
Do not add new information.
Use different wording and sentence structures.

Original:
The model was trained on a large dataset.

Paraphrases:
1. The model was trained using a large amount of data.
2. A large dataset was used to train the model.
3. The model learned from a very large collection of data.

Original:
{text}

Return exactly {n} paraphrases as a JSON array of strings.
Only return valid JSON, do not include numbers or any additional text.
"""


@EXPERIMENTS.register("paraphrase")
class ParaphraseGenerator(Runnable):
    """Лексические + семантические аугментации ответов (порт legacy)."""

    def __init__(self, dataset_path, model_path, samples, seed, aug_number,
                 paraphrase_number, gen_kwargs, save_dir, checkpoint_file, checkpoint_step):
        self.dataset_path = dataset_path
        self.model_path = model_path
        self.samples = samples
        self.seed = seed
        self.aug_number = aug_number
        self.paraphrase_number = paraphrase_number
        self.gen_kwargs = gen_kwargs
        self.save_dir = save_dir
        self.checkpoint_file = checkpoint_file
        self.checkpoint_step = checkpoint_step
        self._model = None
        self._tokenizer = None

    @classmethod
    def from_config(cls, config: dict) -> "ParaphraseGenerator":
        tr = config.get("training", {})
        log = config["logging"]
        return cls(
            config["dataset"]["path"],
            config["model"]["path"],
            tr.get("samples", 70),
            tr.get("seed", 42),
            tr.get("aug_number", 6),
            tr.get("paraphrase_number", 9),
            tr.get("generation", {"max_new_tokens": 4096, "temperature": 0.7, "top_p": 0.9}),
            log["save_path"],
            log.get("checkpoint_file", os.path.join(log["save_path"], "paraphrase_ckpt.jsonl")),
            log.get("checkpoint_step", 10),
        )

    def _load_model(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path, torch_dtype="auto", device_map="auto"
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)

    def _paraphrases(self, text):
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": _PARAPHRASE_PROMPT.format(text=text, n=self.paraphrase_number)},
        ]
        prompt = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._tokenizer([prompt], return_tensors="pt").to(self._model.device)
        generated = self._model.generate(**inputs, do_sample=True, **self.gen_kwargs)
        out_ids = generated[0][len(inputs.input_ids[0]):].tolist()
        return self._tokenizer.decode(out_ids, skip_special_tokens=True)

    @staticmethod
    def _normalize_semantic(x):
        """Привести ответ LLM к плоскому списку строк."""
        if not isinstance(x, list):
            return []
        out = []
        for i in x:
            if isinstance(i, str):
                out.append(i)
            elif isinstance(i, list):
                out.extend(s for s in i if isinstance(s, str))
        return out

    def run(self):
        import random

        from augmentex import CharAug
        from datasets import Dataset, load_dataset

        char_aug = CharAug(
            unit_prob=0.3, min_aug=1, max_aug=5, mult_num=3,
            lang="eng", platform="pc", random_seed=self.seed,
        )

        def add_lexical(example):
            example["lexical"] = [
                char_aug.augment(text=example["response"], action=random.choice(AUGMENTATIONS))
                for _ in range(self.aug_number)
            ]
            return example

        data = load_dataset("arrow", data_files=self.dataset_path)["train"]
        data = data.select(range(self.samples))
        data = data.map(add_lexical, desc="Generate lexical paraphrases")
        data_list = data.to_list()

        self._load_model()

        # Чекпоинты семантических парафраз (восстановление по idx).
        done = {}
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, "r", encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    done[row["idx"]] = row
            print(f"Loaded checkpoint: {len(done)} samples")
        else:
            open(self.checkpoint_file, "w").close()

        with open(self.checkpoint_file, "a", encoding="utf-8") as f:
            for i, row in enumerate(data_list):
                if i in done:
                    data_list[i]["semantic"] = done[i]["semantic"]
                    continue
                try:
                    answer = json.loads(self._paraphrases(row["response"]))
                    semantic = self._normalize_semantic(answer)
                except Exception as e:  # noqa: BLE001 — генерация LLM ненадёжна
                    print(f"Something went wrong: {e}")
                    semantic = []

                data_list[i]["semantic"] = semantic
                f.write(json.dumps({"idx": i, "semantic": semantic}, ensure_ascii=False) + "\n")
                if (i + 1) % self.checkpoint_step == 0:
                    f.flush()
                    print(f"Processed {i + 1}/{len(data_list)}")

        Dataset.from_list(data_list).save_to_disk(self.save_dir)
        print(f"Dataset saved to {self.save_dir}")
        return data_list
