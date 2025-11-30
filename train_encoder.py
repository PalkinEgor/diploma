import json
import os
import argparse
import torch
import pandas as pd
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoTokenizer, AutoModel
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


# Класс кастомной модели
class Model(torch.nn.Module):
    def __init__(self, model_name, output_dim, freeze_bert=True):
        super().__init__()

        # Bert енкодер
        self.bert = AutoModel.from_pretrained(model_name)

        # Проекционная голова для e вектора
        self.e_proj = torch.nn.Linear(self.bert.config.hidden_size, output_dim)

        # Проекционная голова для m вектора
        self.m_proj = torch.nn.Linear(self.bert.config.hidden_size, output_dim)

        # Проекционная голова для среднего значения распределения длин
        self.mu = torch.nn.Linear(self.bert.config.hidden_size, 1)

        # Проекционная голова для стандартного отклонения распределения длин
        self.std = torch.nn.Linear(self.bert.config.hidden_size, 1)

        # Заморозка модели
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask=None):
        out = self.bert(input_ids, attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        e = self.e_proj(cls)
        m = self.m_proj(cls)
        mu = self.mu(cls)
        std = self.std(cls)
        return e, m, mu, std        

# Dataset для текстов и соответствующих векторов
class TextDataset(Dataset):
    def __init__(self, texts, vectors):
        self.texts = texts
        self.vectors = vectors
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx], self.vectors[idx]

# Подготовка батча перед подачей в модель
def collate_fn(batch, tokenizer, decoder_tokenizer, device):
    max_len = tokenizer.model_max_length
    texts = [item[0] for item in batch]
    vectors = [torch.tensor(item[1]) for item in batch]
    input_ids = [tokenizer(text, add_special_tokens=True, return_tensors='pt', truncation=True, max_length=max_len)['input_ids'].reshape(-1) for text in texts]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id).to(device)
    attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)
    vectors = torch.stack(vectors).to(device)
    lengths = torch.tensor([len(decoder_tokenizer(text, return_tensors='pt')['input_ids'].reshape(-1)) for text in texts]).to(device)
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'vectors': vectors,
        'lengths': lengths
    }

# Реализация InfoNCE loss
def info_nce_loss(e_pred, m_pred, e_target, m_target, temperature=0.07):
    # Нормализация e и m векторов
    e_pred = F.normalize(e_pred, dim=-1)
    m_pred = F.normalize(m_pred, dim=-1)
    e_target = F.normalize(e_target, dim=-1)
    m_target = F.normalize(m_target, dim=-1)

    # Составление матриц похожести
    e_sim = torch.matmul(e_pred, e_target.T) / temperature
    m_sim = torch.matmul(m_pred, m_target.T) / temperature

    # Выбор позитивных пар
    e_positive = torch.arange(e_pred.size(0), device=e_pred.device)
    m_positive = torch.arange(m_pred.size(0), device=m_pred.device)

    # Расчет функции потерь
    e_loss = F.cross_entropy(e_sim, e_positive)
    m_loss = F.cross_entropy(m_sim, m_positive)
    loss = 0.5 * (e_loss + m_loss)
    return loss

# Реализация функции потерь для длин ответов
def gaussian_loss(target, mu, std):
    return (0.5 * (std + ((target - mu) ** 2) / torch.exp(std))).mean()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_name', type=str, default='FacebookAI/xlm-roberta-base')
    parser.add_argument('--decoder_model_name', type=str, default='EleutherAI/pythia-410m')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lora', type=bool, default=False)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    DATASET_NAME = 'data/training_results.json'
    SAVE_DIR = 'checkpoints'
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    HYPERPARAMS = {
        'lr': 0.0005,
        'weight_decay': 0.01,
        'betas': (0.9, 0.9)
    }

    with open(DATASET_NAME, 'r', encoding='utf-8') as f:
        data = pd.DataFrame(json.load(f))
    data['text'] = data['text'].apply(lambda x: ' '.join(x.split()[:75]))
    data = data[data['accuracy'] >= 0.85]
    OUTPUT_DIM = len(data['best_vectors'].to_list()[0][0])
    
    # Загрузка модели и токенизаторов
    model = Model(args.model_name, OUTPUT_DIM, freeze_bert=False).to(DEVICE)
    if args.lora:
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=8,
            lora_alpha=32,
            target_modules=['key', 'query', 'value'],
            lora_dropout=0.1,
            bias='none'
        )
        model.bert = get_peft_model(model.bert, lora_config)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    decoder_tokenizer = AutoTokenizer.from_pretrained(args.decoder_model_name)
    if decoder_tokenizer.pad_token is None:
        decoder_tokenizer.pad_token = decoder_tokenizer.eos_token

    # Формирование Датасетов и Даталоадеров
    X = data['instruction'].to_list()
    y = data['best_vectors'].to_list()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=args.seed)
    train_dataset = TextDataset(X_train, y_train)
    test_dataset = TextDataset(X_test, y_test)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                                collate_fn=lambda x: collate_fn(x, tokenizer, decoder_tokenizer, DEVICE))
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, 
                                collate_fn=lambda x: collate_fn(x, tokenizer, decoder_tokenizer, DEVICE))
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=HYPERPARAMS['lr'], betas=HYPERPARAMS['betas'], weight_decay=HYPERPARAMS['weight_decay'])
    best_val_loss = float('inf')
    for i in range(args.num_epochs):

        # Прогон эпохи обучения модели
        model.train()
        train_loss = 0.0
        train_infoNCE_loss = 0.0
        train_gaussian_loss = 0.0 
        for batch in train_dataloader:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            lengths = batch['lengths']
            e_target = batch['vectors'][:, 0, :]
            m_target = batch['vectors'][:, 1, :]

            e_pred, m_pred, mu_pred, std_pred = model(input_ids, attention_mask)
            info_loss = info_nce_loss(e_pred, m_pred, e_target, m_target)
            gauss_loss = gaussian_loss(lengths, mu_pred, std_pred)
            loss = info_loss + gauss_loss

            train_infoNCE_loss += info_loss.item()
            train_gaussian_loss += gauss_loss.item()
            train_loss += loss.item()            
        
            loss.backward()
            optimizer.step()

        train_loss /= len(train_dataloader)
        train_infoNCE_loss /= len(train_dataloader)
        train_gaussian_loss /= len(train_dataloader)

        # Прогон эпохи оценки модели
        model.eval()
        val_loss = 0.0
        val_infoNCE_loss = 0.0
        val_gaussian_loss = 0.0
        with torch.no_grad():
            for batch in test_dataloader:
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                lengths = batch['lengths']
                e_target = batch['vectors'][:, 0, :]
                m_target = batch['vectors'][:, 1, :]

                e_pred, m_pred, mu_pred, std_pred = model(input_ids, attention_mask)
                info_loss = info_nce_loss(e_pred, m_pred, e_target, m_target)
                gauss_loss = gaussian_loss(lengths, mu_pred, std_pred)
                loss = info_loss + gauss_loss

                val_infoNCE_loss += info_loss.item()
                val_gaussian_loss += gauss_loss.item()
                val_loss += loss.item()

        val_loss /= len(test_dataloader)
        val_infoNCE_loss /= len(test_dataloader)
        val_gaussian_loss /= len(test_dataloader)

        # Сохранение лучшей модели
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(SAVE_DIR, 'best_model.pt')  
            #torch.save(model.state_dict(), save_path)          

        # Логирование лосса    
        print(f'Epoch: {i + 1}/{args.num_epochs}')
        print(f'Train Loss: {train_loss}; Eval Loss: {val_loss}')
        print(f'Train InfoNCE Loss: {train_infoNCE_loss}; Eval InfoNCE Loss: {val_infoNCE_loss}')
        print(f'Train Gaussian Loss: {train_gaussian_loss}; Eval Gaussian Loss: {val_gaussian_loss}')
        print()