import torch
import random
from dataset import get_dataset
from datasets import load_from_disk
from collator import collate_dolly_end2end
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM\

HYPERPARAMS = {
    'lr': 0.01,
    'weight_decay': 0.01,
    'betas': (0.9, 0.9)
}

DTYPE_MAP = {
    'float32': torch.float32,
    'float16': torch.float16,
    'bfloat16': torch.bfloat16
}

def fix_seeds(seed):
    torch.manual_seed(seed)
    random.seed(seed)

def load_model(model_name, dtype):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=DTYPE_MAP[dtype], 
        device_map='auto')
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    return model

if __name__ == '__main__':
    model_name = '/userspace/pes/diploma_materials/Llama-3.2-1B'
    dataset_name = '/userspace/pes/diploma_materials/dolly_data'
    dataset_type = 'dolly'
    dtype = 'bfloat16'
    batch_size = 4
    max_tokens = 256
    seed = 42

    fix_seeds(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    raw_dataset = load_from_disk(dataset_name)
    dataset = get_dataset(dataset_type, raw_dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: collate_dolly_end2end(x, tokenizer, max_tokens))

    model = load_model(model_name, dtype)
    device = model.device

    
    epochs = 50
