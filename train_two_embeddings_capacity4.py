import argparse
import torch
import json
import torch.nn.functional as F
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

# Dataset для текстов
class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx], idx

# Создание схемы с одним e вектором и text_length - 1 m векторов
def generate_input(vectors, lengths, max_len, device):
    B, _, H = vectors.shape
    inputs = torch.zeros((B, max_len, H), device=device, dtype=vectors.dtype)
    
    for i, l in enumerate(lengths):
        inputs[i, 0] = vectors[i, 0]
        inputs[i, 1:l] = vectors[i, 1]

    return inputs

# Функция для расчета метрик
# accuracy - точность на уровне токенов
def calculate_accuracy(target, pred):
    accuracy = (pred == target).float().mean().item()
    return accuracy

# Подготовка батча перед подачей в модель
def collate_fn(batch, tokenizer, max_tokens):
    texts = [item[0] for item in batch]
    indices = [item[1] for item in batch]
    input_ids = [tokenizer.encode(text, return_tensors='pt', max_length=max_tokens, truncation=True).reshape(-1) for text in texts]
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

if __name__ == '__main__':
    # Входные параметры
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_name', type=str, default='/userspace/pes/diploma_materials/Llama-3.2-1B')
    parser.add_argument('--maxiter', type=int, default=2000)
    parser.add_argument('--max_tokens', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=1)
    args = parser.parse_args()

    # Фиксируем сиды
    torch.manual_seed(args.seed)

    DATASET_NAME = '/userspace/pes/diploma_materials/dolly_dataset'
    HYPERPARAMS = {
        'lr': 0.01,
        'weight_decay': 0.01,
        'betas': (0.9, 0.9)
    }

    # Достаем тексты
    dataset = load_from_disk(DATASET_NAME)
    df = dataset['train'].to_pandas()
    texts = df['response'].to_list()

    # Записываем в dataset и dataloader
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    text_dataset = TextDataset(texts)
    text_dataloader = DataLoader(
        text_dataset,
        num_workers=2, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=lambda x: collate_fn(x, tokenizer, args.max_tokens))
    
    # Заморозка модели
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, device_map='auto')
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    DEVICE = model.device
    
    result = []
    SAVE_EVERY = 10
    THRESHOLD = 0.95
    t_count = 0
    p_count = 0
    SAVE_PATH = '/userspace/pes/diploma/data/training_results.json'
    for idx, batch in enumerate(text_dataloader):
        # Разбираем батч
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
        
        max_accuracy = [0.0] * B
        best_vectors = [None] * B
        last_iter = 0
        for iter in range(args.maxiter):
            last_iter = iter
            optimizer.zero_grad()

            # Считаем лосс и делаем предсказания
            current_input = generate_input(vectors, lengths, tokenized_text.size(1), DEVICE)
            logits = model(inputs_embeds=current_input, attention_mask=attention_mask).logits
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                labels.view(-1), 
                ignore_index=tokenizer.pad_token_id)
            pred = logits.argmax(dim=-1)

            # Считаем метрики и сохраняем лучшие вектора
            for i in range(B):
                current_len = lengths[i]
                current_pred = pred[i, :current_len]
                current_labels = labels[i, :current_len]

                accuracy = calculate_accuracy(current_labels, current_pred)
                if accuracy > max_accuracy[i]:
                    max_accuracy[i] = accuracy
                    best_vectors[i] = vectors[i].detach().clone()
            
            # Пропускаем итерацию если батч обучен
            good_count = 0
            for i in range(B):
                if max_accuracy[i] >= THRESHOLD:
                    good_count += 1
            if good_count == B:
                break

            loss.backward()
            optimizer.step()

        # Считаем success rate
        p_count += B
        for i in range(B):
            if max_accuracy[i] >= THRESHOLD:
                t_count += 1

        # Логируем прогресс
        print(f'Processed {idx + 1}/{len(text_dataloader)}')
        print(f'Success rate {t_count}/{p_count}')
        print(f'Last iteration {last_iter}')
        print()

        # Обновление результатов
        for i in range(B):
            result.append({
                'instruction': df.iloc[indices[i]]['instruction'],
                'context': df.iloc[indices[i]]['context'],
                'category': df.iloc[indices[i]]['category'],
                'text': texts[i],
                'accuracy': max_accuracy[i],
                'best_vectors': best_vectors[i].float().cpu().numpy().tolist()
            })
        
        # Сохранение результатов
        if (idx + 1) % SAVE_EVERY == 0:
            with open(SAVE_PATH, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=4)

    # Сохранение результатов
    with open(SAVE_PATH, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)