import os
from dotenv import load_dotenv
import argparse
import torch
import json
import torch.nn.functional as F
from datasets import load_from_disk, load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

MAX_TOKENS = 512 # эмпирически выявлено
SAMPLES = 2048

# Dataset для текстов
class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx][1], self.texts[idx][0]
    
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
# correct_prefix_length - длина правильного суффикса;
# seq_accuracy - относительная длина правильного суффикса;
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

# Подготовка батча перед подачей в модель
def collate_fn(batch, tokenizer):
    texts = [item[0] for item in batch]
    indices = [item[1] for item in batch]
    input_ids = [tokenizer.encode(text, return_tensors='pt', max_length=MAX_TOKENS, truncation=True).reshape(-1) for text in texts]
    lengths = [text.shape[0] for text in input_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = (input_ids != tokenizer.pad_token_id).long()
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'lengths': lengths,
        'indices': indices,
        'texts': texts
    }

def get_texts(data):
    result = []
    for idx, item in enumerate(data['response']):
        result.append((idx, item))
    for idx, item in enumerate(data['lexical']):
        for j in item:
            result.append((idx, j))
    for idx, item in enumerate(data['semantic']):
        for j in item:
            result.append((idx, j))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_name', type=str, default='/userspace/pes/diploma_materials/Llama-3.2-1B')
    parser.add_argument('--maxiter', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    # parser.add_argument('--min_words', type=int, default=5)
    # parser.add_argument('--max_words', type=int, default=75)
    # parser.add_argument('--sample_size', type=int, default=float('inf'))
    parser.add_argument('--batch_size', type=int, default=2)
    args = parser.parse_args()
    load_dotenv()
    hf_token = os.environ.get('HF_TOKEN')

    torch.manual_seed(args.seed)

    DATASET_NAME = '/userspace/pes/diploma/data/data-00000-of-00001.arrow'
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    HYPERPARAMS = {
        'lr': 0.01,
        'weight_decay': 0.01,
        'betas': (0.9, 0.9)
    }
    
    dataset = load_dataset('arrow', data_files=DATASET_NAME)
    dataset = dataset['train']
    dataset = dataset.select(range(SAMPLES))
    df = dataset.to_pandas()
    all_texts = get_texts(dataset)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_auth_token=hf_token, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    text_dataset = TextDataset(all_texts)
    text_dataloader = DataLoader(
        text_dataset,
        num_workers=2, 
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
    model.eval()

    result = []
    SAVE_EVERY = 10
    THRESHOLD = 0.9
    SAVE_PATH = '/userspace/pes/diploma/data/training_paraphrase.json'

    for idx, batch in enumerate(tqdm(text_dataloader, desc='Processing dataset')):
        tokenized_text = batch['input_ids'].to(DEVICE)
        lengths = batch['lengths']
        attention_mask = batch['attention_mask'].to(DEVICE)
        indices = batch['indices']
        texts = batch['texts']
        labels = tokenized_text.clone()

        B = tokenized_text.size(0)

        # Создание обучаемых векторов e и m
        vectors = torch.nn.Parameter(torch.randn(B, 2, model.config.hidden_size, device=DEVICE, dtype=model.dtype))
        optimizer = torch.optim.AdamW([vectors], lr=HYPERPARAMS['lr'], betas=HYPERPARAMS['betas'], weight_decay=HYPERPARAMS['weight_decay'])

        # Хранение лучших метрик
        max_accuracy = [0.0] * B
        max_seq_accuracy = [0.0] * B
        best_vectors = [None] * B
        best_metrics = [(0.0, 0.0, 0)] * B
        last_iter = 0

        for iter in range(args.maxiter):
            last_iter = iter
            optimizer.zero_grad()

            current_input = generate_input(vectors, lengths, tokenized_text.size(1), DEVICE)
            logits = model(inputs_embeds=current_input, attention_mask=attention_mask).logits
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                labels.view(-1), 
                ignore_index=tokenizer.pad_token_id)
            pred = logits.argmax(dim=-1)

            for i in range(B):
                current_len = lengths[i]
                current_pred = pred[i, :current_len]
                current_labels = labels[i, :current_len]

                accuracy, seq_accuracy, correct_prefix_length = calculate_metrics(current_labels, current_pred)
                if (accuracy > max_accuracy[i]) or (accuracy == max_accuracy[i] and seq_accuracy > max_seq_accuracy[i]):
                    max_accuracy[i] = accuracy
                    max_seq_accuracy[i] = seq_accuracy
                    best_metrics[i] = (accuracy, seq_accuracy, correct_prefix_length)
                    best_vectors[i] = vectors[i].detach().clone()

            # При достижении идеальной точности пропускаем пример
            good_ex = 0
            for i in range(B):
                if max_accuracy[i] >= THRESHOLD:
                    good_ex += 1
            if good_ex == B:
                break

            loss.backward()
            optimizer.step()

        # Обновление результатов
        for i in range(B):
            result.append({
                'instruction': df.iloc[indices[i]]['instruction'],
                'context': df.iloc[indices[i]]['context'],
                'category': df.iloc[indices[i]]['category'],
                'response': df.iloc[indices[i]]['response'],
                'text': texts[i],
                'accuracy': best_metrics[i][0],
                'seq_accuracy': best_metrics[i][1],
                'correct_prefix_len': best_metrics[i][2],
                'best_vectors': best_vectors[i].float().cpu().numpy().tolist()
            })

        print(f'Processed {idx + 1}/{len(text_dataloader)}')
        print(f'Iterations {last_iter}')

        # Сохранение результатов
        if (idx + 1) % SAVE_EVERY == 0:
            with open(SAVE_PATH, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=4)
    # Сохранение результатов
    with open(SAVE_PATH, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)