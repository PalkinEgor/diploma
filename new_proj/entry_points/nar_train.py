import argparse
import torch
import random
import json
import logging
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
    random.seed(seed)

def setup_logger(path):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler(path),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    return logger

def load_model(config):
    model = AutoModelForCausalLM.from_pretrained(
        config['model']['path'], 
        torch_dtype=DTYPE_MAP[config['model']['dtype']], 
        device_map='auto')
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    return model

def build_dataloader(config, tokenizer):
    dataset = get_dataset(config['dataset']['type'], config['dataset']['path'], config['dataset']['max_samples'])
    collator = get_collator(config['dataset']['type'], config['task_type'], tokenizer, config['training']['max_tokens'])
    dataloader = DataLoader(
        dataset, 
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=collator)
    
    return dataloader


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    logger = setup_logger(config['logging']['save_log_path'])
    fix_seeds(config['training']['seed'])

    tokenizer = AutoTokenizer.from_pretrained(config['model']['path'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataloader = build_dataloader(config, tokenizer)
    
    model = load_model(config)
    device = model.device

    # Прогоняем выборку
    result = []
    total = 0
    success = 0
    nar_trainer = NARfit(model, tokenizer, device, HYPERPARAMS)
    logger.info('start training')
    for idx, batch in enumerate(dataloader):
        accuracy, best_vectors, last_iter, B = nar_trainer.train_batch(
            batch, 
            config['training']['maxiter'], 
            config['training']['threshold']) 
        
        total += B
        success += sum(a >= config['training']['threshold'] for a in accuracy)
        avg_acc = sum(accuracy) / B
        avg_iter = sum(last_iter) / B
        logger.info(f'Processed [{idx + 1}/{len(dataloader)}] '
                    f'Success rate {success}/{total} '
                    f'Average accuracy {avg_acc} '
                    f'Average last iteration {avg_iter}')
        
        for i in range(B):
            item = {k: v[i] for k, v in batch['metainfo'].items()}
            item['accuracy'] = accuracy[i]
            item['last_iter'] = last_iter[i]
            item['best_vectors'] = best_vectors[i].float().cpu().numpy().tolist()
            result.append(item)

        if (idx + 1) % config['logging']['save_every'] == 0:
            logger.info('saving checkpoint...')
            with open(config['logging']['save_path'], 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=4)
    
    # Сохранение результатов
    logger.info('final save...')
    with open(config['logging']['save_path'], 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
