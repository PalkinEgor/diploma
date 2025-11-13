import torch
import os
import argparse
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from datasets import load_dataset
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu


# Dataset для текстов
class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx]
    
# Подготовка батча перед подачей в модель    
def collate_fn(batch, tokenizer, device):
    input_ids = [tokenizer(text, add_special_tokens=True, return_tensors='pt', truncation=True, max_length=tokenizer.model_max_length)['input_ids'].reshape(-1) for text in batch]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id).to(device)
    attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)
    return {
        'texts': batch,
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }

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

# Формирование длин овтетов    
def get_lengths(mu, std, strategy):
    std = torch.exp(0.5 * std)
    if strategy == 'sample':
        lengths = torch.normal(mean=mu, std=std)
        lengths = torch.clamp(lengths, min=mu - 2 * std, max=mu + 2 * std)
    if strategy == 'mean':
        lengths = mu
    lengths = torch.clamp(lengths.round(), min=1).long()
    return lengths

# Создание схемы с одним e вектором и text_length - 1 m векторов
def generate_input_one(vectors, text_length):
    return torch.cat([vectors[:1, None, :], vectors[1:2, None, :].expand(-1, text_length - 1, -1)], dim=1)

# Создание схемы для целого батча
def generate_input(batch_vectors, lengths):
    embeds = []
    for vectors, length in zip(batch_vectors, lengths):
        embeds.append(generate_input_one(vectors, length).squeeze(0))
    return pad_sequence(embeds, batch_first=True, padding_value=0.0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--decoder_name', type=str, default='Qwen/Qwen2.5-0.5B')
    parser.add_argument('--encoder_name', type=str, default='FacebookAI/xlm-roberta-base')
    parser.add_argument('--checkpoint_name', type=str, default='best_model.pt')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--min_words', type=int, default=5)
    parser.add_argument('--max_words', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    DATASET_NAME = 'databricks/databricks-dolly-15k'
    CHECKPOINT_PATH = os.path.join('checkpoints', args.checkpoint_name)

    dataset = load_dataset(DATASET_NAME)
    df = dataset['train'].to_pandas()

    # Выбираем ответы, которые длиннее min_words слов
    df = df[df['response'].apply(lambda x: len(x.split(' ')) > args.min_words)].reset_index(drop=True)
    texts = list(df['response'])

    # Обрезка слишком длинных ответов до max_words слов
    texts = [' '.join(text.split(' ')[:args.max_words]) if len(text.split(' ')) > args.max_words else text for text in texts]

    # Загрузка модели и токенайзера декодера
    decoder_model = AutoModelForCausalLM.from_pretrained(args.decoder_name).to(DEVICE)
    for param in decoder_model.parameters():
        param.requires_grad = False
    decoder_model.eval()
    decoder_tokenizer = AutoTokenizer.from_pretrained(args.decoder_name)

    # Загрузка модели и токенайзера енкодера
    encoder_model = Model(model_name=args.encoder_name, output_dim=decoder_model.config.hidden_size).to(DEVICE)
    encoder_model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    for param in encoder_model.parameters():
        param.requires_grad = False
    encoder_model.eval()
    encoder_tokenizer = AutoTokenizer.from_pretrained(args.encoder_name, truncation=True)

    test_dataset = TextDataset(texts)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=lambda x: collate_fn(x, encoder_tokenizer, DEVICE))

    corpus_preds = []
    for batch in tqdm(test_dataloader, desc='Processing dataset'):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']

        e, m, mu, std = encoder_model(input_ids, attention_mask)
        lengths = get_lengths(mu, std, 'mean')
        max_len = torch.max(lengths)
        vectors = torch.stack([e, m], dim=1)

        current_input = generate_input(vectors, lengths)    
        attention_mask = torch.arange(max_len, device=lengths.device).expand(lengths.size(0), max_len) < lengths

        logits = decoder_model(inputs_embeds=current_input, attention_mask=attention_mask).logits
        pred_ids = torch.argmax(logits, dim=-1)
        preds = decoder_tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        preds = [pred.split(' ') for pred in preds]
        corpus_preds.extend(preds)
    
    # Подсчет метрик
    texts = [[text.split(' ')] for text in texts]
    bleu_score_corpus = corpus_bleu(texts, corpus_preds)
    print(f'Corpus BLEU Score: {bleu_score_corpus}')
