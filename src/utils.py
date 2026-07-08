import random
import torch
import yaml
from huggingface_hub import snapshot_download


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def fix_seeds(seed):
    if seed is None:
        return
    torch.manual_seed(seed)
    random.seed(seed)


def load_model(path, dir_name, token=None):
    kwargs = dict(
        repo_id=path,
        local_dir=dir_name,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    if token:
        kwargs["token"] = token
    snapshot_download(**kwargs)


def generate_input(vectors, lengths, max_len, pad_embed, device):
    """Собрать входные эмбеддинги ``[e, m, m, ..., m, pad, ...]`` для декодера."""
    B, _, H = vectors.shape
    inputs = torch.zeros((B, max_len, H), device=device, dtype=vectors.dtype)

    for i, l in enumerate(lengths):
        inputs[i, 0] = vectors[i, 0]
        inputs[i, 1:l] = vectors[i, 1]
        inputs[i, l:] = pad_embed

    return inputs
