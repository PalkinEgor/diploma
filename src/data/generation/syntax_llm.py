import json

from ...base import Runnable
from ...registry import EXPERIMENTS
from .syntax_cfg import CATEGORIES

# Описания типов предложений для промптов (структура важнее смысла).
_TYPE_SPECS = {
    "simple": ("Simple declarative sentence\n- One subject and one predicate\n"
               "- No subordinate clauses", '"The child sleeps."', "3-6 words"),
    "complex": ("Complex declarative sentence\n- One main clause\n"
                "- One modifier phrase separated by commas\n- No conjunctions between clauses",
                '"The child, tired after the day, sleeps"', "6-12 words, use commas exactly as shown"),
    "question_simple": ("Simple interrogative sentence\n- Yes/no question\n- No modifiers",
                        '"Do birds sing?"', "3-6 words, use auxiliary verb (do/does), end with '?'"),
    "question_complex": ("Complex interrogative sentence\n- Yes/no question\n"
                         "- Include one modifier phrase\n- No subordinate clauses",
                         '"Does the tired child sleep at night?"', "6-12 words, end with '?'"),
    "incentive_simple": ("Simple imperative sentence\n- No subject\n- No modifiers",
                         '"Run" / "Close the door"', "1-4 words, use base verb form"),
    "incentive_complex": ("Complex imperative sentence\n- No subject\n"
                          "- Include one modifier phrase\n- No conjunctions",
                          '"Open the door in silence"', "4-10 words, use base verb form"),
    "one_part": ("One-part sentence\n- Only a noun phrase OR only a verb phrase\n"
                 "- No subject-predicate structure", '"Night." / "Running." / "Silence."', "1-3 words"),
}


def _prompt(category, sample_size):
    spec, example, reqs = _TYPE_SPECS[category]
    return (
        f"Generate {sample_size} English sentences.\n\n"
        f"Sentence type:\n- {spec}\n\n"
        f"Structure example:\n{example}\n\n"
        f"Requirements:\n- Focus on syntax, not meaning\n- Use simple common words\n- {reqs}\n"
        f"- Return exactly {sample_size} sentences as a JSON array of strings.\n"
        "Only return valid JSON, do not include numbers or any additional text.\n"
        "Return JSON only. Start with '[' and end with ']'"
    )


def safe_json_load(text):
    """Извлечь JSON-массив из ответа LLM (между первой ``[`` и последней ``]``)."""
    text = text.strip()
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1:
        raise ValueError("No JSON array found")
    return json.loads(text[start:end + 1])


@EXPERIMENTS.register("syntax_llm")
class SyntaxLLMGenerator(Runnable):
    """Генератор синтаксических предложений через LLM (порт legacy)."""

    def __init__(self, model_path, sample_size, iterations, gen_kwargs, save_path):
        self.model_path = model_path
        self.sample_size = sample_size
        self.iterations = iterations
        self.gen_kwargs = gen_kwargs
        self.save_path = save_path
        self._model = None
        self._tokenizer = None

    @classmethod
    def from_config(cls, config: dict) -> "SyntaxLLMGenerator":
        tr = config.get("training", {})
        return cls(
            config["model"]["path"],
            tr.get("sample_size", 30),
            tr.get("iterations", 1),
            tr.get("generation", {"max_new_tokens": 1024, "temperature": 0.7, "top_p": 0.9}),
            config["logging"]["save_path"],
        )

    def _load(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path, torch_dtype="auto", device_map="auto"
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)

    def _generate(self, prompt):
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        text = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._tokenizer([text], return_tensors="pt").to(self._model.device)
        generated = self._model.generate(**inputs, do_sample=True, **self.gen_kwargs)
        out_ids = generated[0][len(inputs.input_ids[0]):].tolist()
        return self._tokenizer.decode(out_ids, skip_special_tokens=True)

    def run(self):
        self._load()
        result = {cat: [] for cat in CATEGORIES}
        step = 0
        for cat in CATEGORIES:
            for _ in range(self.iterations):
                answer = None
                try:
                    answer = self._generate(_prompt(cat, self.sample_size))
                    result[cat].extend(safe_json_load(answer))
                except Exception as e:  # noqa: BLE001 — генерация LLM ненадёжна, продолжаем
                    print(f"Something went wrong: {e}\nProblem answer: {answer}")
                step += 1
                print(f"Progress: {step}/{len(CATEGORIES) * self.iterations}")

        with open(self.save_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False)
        return result
