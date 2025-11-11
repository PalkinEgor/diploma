import argparse
import torch
import json
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
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

# Оценка классического подхода
def evaluate_plain_model(model, input_ids):
    model.eval()
    with torch.no_grad():
        logits = model(input_ids).logits
        vocab_size = logits.shape[-1]
        
        logits_for_loss = logits[:, :-1, :].reshape((-1, vocab_size))
        targets_for_loss = input_ids[:, 1:].reshape(-1)

        loss = F.cross_entropy(logits_for_loss, targets_for_loss).item()
        pred = logits[:, :-1, :].argmax(dim=-1).reshape(-1)

        accuracy, seq_accuracy, correct_prefix_length = calculate_metrics(targets_for_loss, pred)
        return loss, accuracy, seq_accuracy, correct_prefix_length

# Создание схемы с одним e вектором и text_length - 1 m векторов
def generate_input_one(vectors, text_length):
    return torch.cat([vectors[:1, None, :], vectors[1:2, None, :].expand(-1, text_length - 1, -1)], dim=1)

# Создание схемы для целого батча
def generate_input(batch_vectors, lengths):
    embeds = []
    for vectors, length in zip(batch_vectors, lengths):
        embeds.append(generate_input_one(vectors, length).squeeze(0))
    return pad_sequence(embeds, batch_first=True, padding_value=0.0)    

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
def collate_fn(batch, tokenizer, device):
    texts = [item[0] for item in batch]
    indices = [item[1] for item in batch]
    input_ids = [tokenizer.encode(text, return_tensors='pt').reshape(-1) for text in texts]
    lengths = [text.shape[0] for text in input_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id).to(device)
    attention_mask = (input_ids != tokenizer.pad_token_id).int()
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'lengths': lengths,
        'indices': indices
    }

# Безопасное семплирование из категорий
def safe_sample(group, n, seed):
    if len(group) < n:
        return group
    return group.sample(n, random_state=seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-0.5B')
    parser.add_argument('--maxiter', type=int, default=3000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--min_words', type=int, default=5)
    parser.add_argument('--max_words', type=int, default=200)
    parser.add_argument('--sample_size', type=int, default=float('inf'))
    parser.add_argument('--batch_size', type=int, default=1)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    DATASET_NAME = 'databricks/databricks-dolly-15k'
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    HYPERPARAMS = {
        'lr': 0.01,
        'weight_decay': 0.01,
        'betas': (0.9, 0.9)
    }
    
    dataset = load_dataset(DATASET_NAME)
    df = dataset['train'].to_pandas()

    # Выбираем ответы, которые длиннее min_words слов
    df = df[df['response'].apply(lambda x: len(x.split(' ')) > args.min_words)].reset_index(drop=True)

    # Выбор из каждой категории по sample_size примеров
    df = df.groupby(by='category', group_keys=False).apply(lambda x: safe_sample(x, args.sample_size, args.seed)).reset_index(drop=True)
    texts = list(df['response'])

    # Обрезка слишком длинных ответов до max_words слов
    texts = [' '.join(text.split(' ')[:args.max_words]) if len(text.split(' ')) > args.max_words else text for text in texts]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    text_dataset = TextDataset(texts)
    text_dataloader = DataLoader(text_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=lambda x: collate_fn(x, tokenizer, DEVICE))

    # Заморозка модели
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(DEVICE)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    result = []
    SAVE_EVERY = 25
    SAVE_PATH = 'data/training_results.json'
    for idx, batch in enumerate(tqdm(text_dataloader, desc='Processing dataset')):
        tokenized_text = batch['input_ids']
        lengths = batch['lengths']
        attention_mask = batch['attention_mask']
        indices = batch['indices']
        labels = tokenized_text.clone()
        
        # Создание обучаемых векторов e и m
        vectors = torch.nn.Parameter(torch.randn(args.batch_size, 2, model.config.hidden_size, device=DEVICE))
        optimizer = torch.optim.AdamW([vectors], lr=HYPERPARAMS['lr'], betas=HYPERPARAMS['betas'], weight_decay=HYPERPARAMS['weight_decay'])
        
        # Хранение лучших метрик
        max_accuracy = torch.zeros(args.batch_size, device=DEVICE)
        max_seq_accuracy = torch.zeros(args.batch_size, device=DEVICE)
        best_vectors = [None] * args.batch_size
        best_metrics = [(0.0, 0.0, 0)] * args.batch_size
        
        for _ in range(args.maxiter):
            optimizer.zero_grad()
            
            current_input = generate_input(vectors, lengths)
            logits = model(inputs_embeds=current_input, attention_mask=attention_mask).logits
            logits = logits.reshape(-1, logits.shape[-1])
            loss = torch.nn.functional.cross_entropy(logits, labels.reshape(-1), ignore_index=tokenizer.pad_token_id)
            pred = logits.argmax(dim=-1).view(tokenized_text.shape)

            # Обновление метрик
            correct_counter = 0
            for i in range(args.batch_size):

                # При достижении идеальной точности пропускаем пример
                if max_accuracy[i] == 1.0:
                    correct_counter += 1
                    continue

                # Берем предсказания без паддинга и считаем метрики
                current_pred = pred[i, :lengths[i]]
                current_labels = labels[i, :lengths[i]]
                accuracy, seq_accuracy, correct_prefix_length = calculate_metrics(current_labels, current_pred)

                # Сохранение лучших метрик
                if (accuracy > max_accuracy[i]) or (accuracy == max_accuracy[i] and seq_accuracy > max_seq_accuracy[i]):
                    max_accuracy[i] = accuracy
                    max_seq_accuracy[i] = seq_accuracy
                    best_metrics[i] = (accuracy, seq_accuracy, correct_prefix_length)
                    best_vectors[i] = vectors[i].detach().clone()
            if correct_counter == args.batch_size:
                break

            loss.backward()
            optimizer.step()
        
        # Обновление результатов
        for i in range(args.batch_size):
            result.append({
                'instruction': df.iloc[indices[i]]['instruction'],
                'context': df.iloc[indices[i]]['context'],
                'category': df.iloc[indices[i]]['category'],
                'text': df.iloc[indices[i]]['response'],
                'accuracy': best_metrics[i][0],
                'seq_accuracy': best_metrics[i][1],
                'correct_prefix_len': best_metrics[i][2],
                'best_vectors': best_vectors[i].cpu().numpy().tolist()
        })
            
        # Сохранение результатов
        if (idx + 1) % SAVE_EVERY == 0:
            with open(SAVE_PATH, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=4)
    
    # Сохранение результатов
    with open(SAVE_PATH, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)