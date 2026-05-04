import torch
import yaml
from huggingface_hub import snapshot_download


# Загружаем конфиг эксперимента
def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

# Скачиваем модель локально    
def load_model(path, dir_name, token=None):
    if token:
        snapshot_download(
            repo_id=path,
            local_dir=dir_name,
            local_dir_use_symlinks=False,
            resume_download=True,
            token=token
        )
    else:
        snapshot_download(
            repo_id=path,
            local_dir=dir_name,
            local_dir_use_symlinks=False,
            resume_download=True
        )

# Создание схемы с одним e вектором и text_length - 1 m векторов
def generate_input(vectors, lengths, max_len, pad_embed, device):
    B, _, H = vectors.shape
    inputs = torch.zeros((B, max_len, H), device=device, dtype=vectors.dtype)
    
    for i, l in enumerate(lengths):
        inputs[i, 0] = vectors[i, 0]
        inputs[i, 1:l] = vectors[i, 1]
        inputs[i, l:] = pad_embed

    return inputs