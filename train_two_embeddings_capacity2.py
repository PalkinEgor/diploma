import os
from dotenv import load_dotenv
import argparse
import torch
import json
import torch.nn.functional as F
from datasets import load_from_disk
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
        return self.texts[idx][1], self.texts[idx][0]
    
# Создание схемы с одним e вектором и text_length - 1 m векторов
def generate_input(vectors, text_length):
    return torch.cat([vectors[0:1, :].unsqueeze(0), vectors[1:2, :].unsqueeze(0).expand(1, text_length - 1, -1)], dim=1)

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
    parser.add_argument('--model_name', type=str, default='EleutherAI/pythia-410m')
    parser.add_argument('--maxiter', type=int, default=3000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--min_words', type=int, default=5)
    parser.add_argument('--max_words', type=int, default=75)
    parser.add_argument('--sample_size', type=int, default=float('inf'))
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--init', type=str, default='random')
    args = parser.parse_args()
    load_dotenv()
    hf_token = os.environ.get('HF_TOKEN')

    torch.manual_seed(args.seed)

    DATASET_NAME = 'C:\\Users\\79237\\Desktop\\diploma\\data'
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    HYPERPARAMS = {
        'lr': 0.01,
        'weight_decay': 0.01,
        'betas': (0.9, 0.9)
    }
    
    dataset = load_from_disk(DATASET_NAME)
    df = dataset.to_pandas()
    texts = get_texts(dataset)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_auth_token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    text_dataset = TextDataset(texts)
    text_dataloader = DataLoader(text_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=lambda x: collate_fn(x, tokenizer, DEVICE))

    # Заморозка модели
    model = AutoModelForCausalLM.from_pretrained(args.model_name, use_auth_token=hf_token).to(DEVICE)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    result = []
    SAVE_EVERY = 25
    THRESHOLD = 0.9
    SAVE_PATH = 'data/training_results.json'

    for idx, batch in enumerate(tqdm(text_dataloader, desc='Processing dataset')):
        tokenized_text = batch['input_ids']
        lengths = batch['lengths']
        attention_mask = batch['attention_mask']
        indices = batch['indices']
        texts = batch['texts']
        labels = tokenized_text.clone()

        # Создание обучаемых векторов e и m
        vectors = torch.nn.Parameter(torch.randn(2, model.config.hidden_size, device=DEVICE))
        if args.init == 'mean':
            with torch.no_grad():
                emb = model.get_input_embeddings().weight
                mean_vec = emb.mean(dim=0)
                vectors[:] = mean_vec
        optimizer = torch.optim.AdamW([vectors], lr=HYPERPARAMS['lr'], betas=HYPERPARAMS['betas'], weight_decay=HYPERPARAMS['weight_decay'])

        # Хранение лучших метрик
        max_accuracy = 0.0
        max_seq_accuracy = 0.0
        best_vectors = None
        best_metrics = (0.0, 0.0, 0)
        
        # Ранняя остановка
        patience = 0

        for iter in range(args.maxiter):
            optimizer.zero_grad()

            current_input = generate_input(vectors, lengths[0])
            logits = model(inputs_embeds=current_input, attention_mask=attention_mask).logits
            loss = torch.nn.functional.cross_entropy(logits[0, :, :], labels[0], ignore_index=tokenizer.pad_token_id)
            pred = logits.argmax(dim=-1).view(tokenized_text.shape)

            # При достижении идеальной точности пропускаем пример
            if max_accuracy >= THRESHOLD:
                break

            # Берем предсказания без паддинга и считаем метрики
            current_pred = pred[0, :lengths[0]]
            current_labels = labels[0, :lengths[0]]
            accuracy, seq_accuracy, correct_prefix_length = calculate_metrics(current_labels, current_pred)
            
            # Сохранение лучших метрик
            if (accuracy > max_accuracy) or (accuracy == max_accuracy and seq_accuracy > max_seq_accuracy):
                max_accuracy = accuracy
                max_seq_accuracy = seq_accuracy
                best_metrics = (accuracy, seq_accuracy, correct_prefix_length)
                best_vectors = vectors.detach().clone()

            loss.backward()
            optimizer.step()

        # Обновление результатов
        result.append({
            'instruction': df.iloc[indices[0]]['instruction'],
            'context': df.iloc[indices[0]]['context'],
            'category': df.iloc[indices[0]]['category'],
            'text': df.iloc[indices[0]]['response'],
            'accuracy': best_metrics[0],
            'seq_accuracy': best_metrics[1],
            'correct_prefix_len': best_metrics[2],
            'best_vectors': best_vectors.cpu().numpy().tolist()
        })

        # Сохранение результатов
        if (idx + 1) % SAVE_EVERY == 0:
            print(f'Processed {idx + 1}/{len(texts)}')
            with open(SAVE_PATH, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=4)

    # Сохранение результатов
    with open(SAVE_PATH, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)