import logging
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from .data import get_collator, get_dataset

DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

DEFAULT_OPT = {"lr": 0.01, "weight_decay": 0.01, "betas": (0.9, 0.9)}


def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def build_adamw(params, training_cfg):
    opt = dict(DEFAULT_OPT)
    opt.update(training_cfg.get("optimizer", {}))
    if isinstance(opt["betas"], list):
        opt["betas"] = tuple(opt["betas"])
    return torch.optim.AdamW(
        params, lr=opt["lr"], betas=opt["betas"], weight_decay=opt["weight_decay"]
    )


def resolve_dtype(name):
    if name not in DTYPE_MAP:
        raise ValueError(f"Unknown dtype: {name}. Доступно: {sorted(DTYPE_MAP)}")
    return DTYPE_MAP[name]


def load_tokenizer(path):
    tokenizer = AutoTokenizer.from_pretrained(path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_frozen_causal_lm(path, dtype):
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=resolve_dtype(dtype), device_map="auto"
    )
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    return model


def setup_logger(path=None, name="proto_tokens"):
    """Логгер в stdout (и в файл ``path``, если задан)."""
    handlers = [logging.StreamHandler()]
    if path:
        handlers.insert(0, logging.FileHandler(path))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
    )
    return logging.getLogger(name)


def build_dataloader(config, tokenizer, task_type):
    ds_cfg = config["dataset"]
    tr_cfg = config["training"]
    dataset = get_dataset(
        ds_cfg["type"],
        task_type,
        ds_cfg["path"],
        max_samples=ds_cfg.get("max_samples"),
        threshold=tr_cfg.get("threshold", 0.9),
    )
    collator = get_collator(ds_cfg["type"], task_type, tokenizer, tr_cfg["max_tokens"])
    return DataLoader(
        dataset,
        batch_size=tr_cfg["batch_size"],
        shuffle=tr_cfg.get("shuffle", False),
        collate_fn=collator,
    )
