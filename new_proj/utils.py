import torch
import yaml


# Загружаем конфиг эксперимента
def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

# Создание схемы с одним e вектором и text_length - 1 m векторов
def generate_input(vectors, lengths, max_len, pad_embed, device):
    B, _, H = vectors.shape
    inputs = torch.zeros((B, max_len, H), device=device, dtype=vectors.dtype)
    
    for i, l in enumerate(lengths):
        inputs[i, 0] = vectors[i, 0]
        inputs[i, 1:l] = vectors[i, 1]
        inputs[i, l:] = pad_embed

    return inputs