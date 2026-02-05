import argparse
import torch
import json
import os
import random
import math
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader
from dotenv import load_dotenv
from torch.nn.utils.rnn import pad_sequence


# Dataset для текстов
class TextDataset(Dataset):
    def __init__(self, data):
        self.texts = [item['text'] for item in data]
        self.e_vectors = [item['best_vectors'][0] for item in data]
        self.m_vectors = [item['best_vectors'][1] for item in data]
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx], self.e_vectors[idx], self.m_vectors[idx]
    
# Подготовка батча перед подачей в модель
def collate_fn(batch, tokenizer):
    texts = [item[0] for item in batch]
    e_vectors = [item[1] for item in batch]
    m_vectors = [item[2] for item in batch]

    input_ids = [tokenizer.encode(text, return_tensors='pt').reshape(-1) for text in texts]
    lengths = [text.shape[0] for text in input_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = (input_ids != tokenizer.pad_token_id).long()

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'lengths': lengths,
        'texts': texts,
        'e_vectors': e_vectors,
        'm_vectors': m_vectors
    }

# Создание схемы с одним e вектором и text_length - 1 m векторов
def generate_input(vectors, lengths, max_len, device):
    B, _, H = vectors.shape
    inputs = torch.zeros((B, max_len, H), device=device, dtype=vectors.dtype)
    
    for i, l in enumerate(lengths):
        inputs[i, 0] = vectors[i, 0]
        inputs[i, 1:l] = vectors[i, 1]

    return inputs

# Функция для расчета метрик. 
# accuracy - точность на уровне токенов; 
# correct_prefix_length - длина правильного префикса;
# seq_accuracy - относительная длина правильного префикса;
def calculate_metrics(target, pred):
    accuracy = (pred == target).float().mean().item()
    first_wrong = torch.nonzero((pred != target).float())
    if first_wrong.shape[0] == 0:
        seq_accuracy = 1.0
        correct_prefix_length = len(pred)
    else:
        seq_accuracy = first_wrong[0].item() / len(pred)
        correct_prefix_length = first_wrong[0].item()
    return accuracy, seq_accuracy, correct_prefix_length

def get_texts(data):
    # Отбираем только качественные примеры
    good_data = []
    for item in data:
        if item['accuracy'] >= 0.9:
            good_data.append(item)

    result = []
    for item in good_data:
        if item['response'] == item['text']:
            result.append({'text': item['response'], 'best_vectors': item['best_vectors']})
    return result  

# Получаем вектор шума
def get_noise(size, alpha, noise_type, e_vector):
    if noise_type == 'gaussian':
        noise = np.random.normal(size=size)

    elif noise_type == 'uniform':
        noise = np.random.uniform(-1.0, 1.0, size)

    elif noise_type == 'exponential':
        noise = np.random.exponential(size=size)
        noise *= np.random.choice([-1, 1], size=size)

    elif noise_type == 'sinusoidal':
        k = random.choice(range(4, 33))
        freq = 2 * math.pi * k / size
        phase = np.random.uniform(0, 2 * math.pi, 1)[0]
        noise = np.sin(freq * np.arange(size) + phase)

    else:
        noise = np.random.normal(size=size)

    return (noise / np.linalg.norm(noise)) * np.linalg.norm(e_vector) * alpha


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-3.2-1B')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=1)

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    load_dotenv()
    hf_token = os.environ.get('HF_TOKEN')

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    HYPERPARAMS = {
        'lr': 0.01,
        'weight_decay': 0.01,
        'betas': (0.9, 0.9)
    }

    # Подготовка данных
    DATA_PATH = 'training_paraphrase.json'
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    data = get_texts(data)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_auth_token=hf_token, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    text_dataset = TextDataset(data)
    text_dataloader = DataLoader(
        text_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=lambda x: collate_fn(x, tokenizer))
    
    # Заморозка модели
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, 
        use_auth_token=hf_token, 
        torch_dtype=torch.bfloat16, 
        device_map='auto')
    for param in model.parameters():
        param.requires_grad = False
    model.set_attn_implementation('eager')
    model.eval()

    alpha = [0.05, 0.10, 0.20, 0.50, 1.00]
    noise_types = ['gaussian', 'uniform', 'sinusoidal', 'exponential']
    result = []
    SAVE_PATH = 'noise_effect.json'

    count = 0
    for a in alpha:
        for noise_type in noise_types:
            mean_accuracy = 0.0
            mean_seq_accuracy = 0.0
            mean_correct_prefix_length = 0.0
            for batch in text_dataloader:
                tokenized_text = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                lengths = batch['lengths']
                texts = batch['texts']
                e_vectors = torch.tensor(batch['e_vectors'], device=DEVICE, dtype=model.dtype)
                e_vectors_noisy = e_vectors.clone()
                m_vectors = torch.tensor(batch['m_vectors'], device=DEVICE, dtype=model.dtype)
                B = tokenized_text.size(0)

                for i in range(B):
                    noise = torch.tensor(get_noise(e_vectors.shape[-1], a, noise_type, e_vectors[i].detach().cpu().float().numpy()), device=DEVICE, dtype=model.dtype)
                    e_vectors_noisy[i] += noise
                
                vectors = torch.stack([e_vectors_noisy, m_vectors], dim=1)
                current_input = generate_input(vectors, lengths, tokenized_text.size(1), DEVICE)
                with torch.no_grad():
                    logits = model(inputs_embeds=current_input, attention_mask=attention_mask).logits
                    pred = logits.argmax(dim=-1)

                for i in range(B):
                    current_len = lengths[i]
                    current_pred = pred[i, :current_len]
                    current_labels = tokenized_text[i, :current_len]

                    accuracy, seq_accuracy, correct_prefix_length = calculate_metrics(current_labels, current_pred)
                    mean_accuracy += accuracy
                    mean_seq_accuracy += seq_accuracy
                    mean_correct_prefix_length += correct_prefix_length

            mean_accuracy /= len(data)
            mean_seq_accuracy /= len(data)
            mean_correct_prefix_length /= len(data)

            result.append({
                'alpha': a,
                'noise': noise_type,
                'accuracy': mean_accuracy,
                'seq_accuracy': mean_seq_accuracy,
                'correct_prefix_length': mean_correct_prefix_length
            })

            count += 1
            print(f'Processed: {count}/{len(alpha) * len(noise_types)}')

    # Сохранение результатов
    with open(SAVE_PATH, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)