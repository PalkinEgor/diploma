import argparse
import torch
from datasets import load_from_disk
from dataset import get_dataset
from collator import get_collator
from utils import load_config
from trainer import NARfit
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    fix_seeds(config['training']['seed'])

    # Загружаем данные
    raw_dataset = load_from_disk(config['dataset']['path'])
    dataset = get_dataset(config['dataset']['type'], raw_dataset)

    # Загружаем токенизатор
    tokenizer = AutoTokenizer.from_pretrained(config['model']['path'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Складываем данные в DataLoader
    collator = get_collator(config['dataset']['type'], tokenizer, config['trainig']['max_tokens'])
    dataloader = DataLoader(
        dataset, 
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=collator)
    
    # Загружаем модель
    model = AutoModelForCausalLM.from_pretrained(
        config['model']['path'], 
        torch_dtype=DTYPE_MAP[config['model']['dtype']], 
        device_map='auto')
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    device = model.device

    # Прогоняем выборку
    result = []
    nar_trainer = NARfit(model, tokenizer, device, HYPERPARAMS)
    for idx, batch in enumerate(dataloader):
        accuracy, best_vectors, last_iter, B = nar_trainer.train_batch(
            batch, 
            config['training']['maxiter'], 
            config['training']['threshold'])    
        for i in range(B):
            item = {key: value for key, value in batch['metainfo'].items()}
            item['accuracy'] = accuracy[i]
            item['best_vectors'] = best_vectors[i].float().cpu().numpy().tolist()
            result.append(item)
