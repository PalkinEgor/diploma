import argparse
import torch
import random
import logging
from tqdm import tqdm
from init_codebooks import CodeBooksInit
from dataset import get_dataset
from collator import get_collator
from utils import load_config
from trainer import CodeBookFit
from models import EncoderCodeBooksModel, FullCodeBooksModel
from transformers import AutoTokenizer
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

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

def init_codebooks(config):
    dataset = get_dataset(config['dataset']['type'], config['task_type'], config['dataset']['path'])
    code_book_init = CodeBooksInit(
        dataset, 
        config['training']['code_books']['V'], 
        config['training']['seed'],
        DTYPE_MAP[config['model']['dtype']],
        False      
    )
    code_book_init.init_codebooks()
    return code_book_init

def build_dataloader(config, tokenizer):
    dataset = get_dataset(config['dataset']['type'], config['task_type'], config['dataset']['path'])
    collator = get_collator(config['dataset']['type'], config['task_type'], tokenizer, config['training']['max_tokens'])
    dataloader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=collator
    )

    return dataloader

def load_model(config, code_book_init, tokenizer):
    encoder_model = EncoderCodeBooksModel(
        config['model']['path'],
        config['training']['code_books']['G'],
        config['training']['code_books']['V'],
        code_book_init.e_code_books,
        code_book_init.m_code_books,
        config['training']['code_books']['tau'],
        DTYPE_MAP[config['model']['dtype']],
        config['training']['code_books']['m_vector'],
        config['training']['mean_pooling']
    )
    encoder_model = encoder_model.to(DEVICE)
    full_model = FullCodeBooksModel(
        encoder_model, 
        config['model']['path'],
        tokenizer,
        DTYPE_MAP[config['model']['dtype']],
        config['training']['code_books']['m_vector']
    )
    full_model = full_model.to(DEVICE)
    return full_model

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

    code_book_init = init_codebooks(config)

    dataloader = build_dataloader(config, tokenizer)

    model = load_model(config, code_book_init, tokenizer)
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=HYPERPARAMS['lr'], 
        betas=HYPERPARAMS['betas'], 
        weight_decay=HYPERPARAMS['weight_decay']
    )
    device = next(model.parameters()).device

    codebooks_trainer = CodeBookFit(model, optimizer, tokenizer, device, config['training']['code_books']['m_vector'], config['training']['diversity_loss_weight'])
    n_epochs = config['training']['n_epochs']

    logger.info('start training')
    for epoch in range(n_epochs):
        epoch_accuracy = []
        epoch_loss = 0
        epoch_ce_loss = 0
        epoch_diversity_loss = 0
        logger.info(f'Start {epoch + 1} epoch')
        for idx, batch in tqdm(enumerate(dataloader)):
            loss, ce_loss, diversity_loss, accuracy = codebooks_trainer.train_batch(batch)
            epoch_accuracy.extend(accuracy)
            epoch_loss += loss
            epoch_ce_loss += ce_loss
            epoch_diversity_loss += diversity_loss

        logger.info(
            "Epoch: %d; Loss: %.4f; CE_Loss: %.4f; Diversity_Loss: %.4f; Accuracy: %.4f",
            epoch + 1,
            epoch_loss / len(dataloader),
            epoch_ce_loss / len(dataloader),
            epoch_diversity_loss / len(dataloader),
            sum(epoch_accuracy) / len(epoch_accuracy),
        )
    logger.info('finish training')        