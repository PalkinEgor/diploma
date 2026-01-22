import json
import os
import torch
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from dotenv import load_dotenv


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
    
# Создание схемы с одним e вектором и text_length - 1 m векторов
def generate_input(vectors, lengths, max_len, device):
    B, _, H = vectors.shape
    inputs = torch.zeros((B, max_len, H), device=device, dtype=vectors.dtype)
    
    for i, l in enumerate(lengths):
        inputs[i, 0] = vectors[i, 0]
        inputs[i, 1:l] = vectors[i, 1]

    return inputs
    
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
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-3.2-1B')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--mean_attention', type=bool, default=False)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
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

    result = []
    SAVE_EVERY = 10
    if args.mean_attention:
        SAVE_PATH = 'mean_attention.json'
    else:
        SAVE_PATH = 'attention.json'
    for idx, batch in enumerate(tqdm(text_dataloader, desc='Processing dataset')):
        tokenized_text = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        lengths = batch['lengths']
        texts = batch['texts']
        e_vectors = torch.tensor(batch['e_vectors'], device=DEVICE, dtype=model.dtype)
        m_vectors = torch.tensor(batch['m_vectors'], device=DEVICE, dtype=model.dtype)
        B = tokenized_text.size(0)

        vectors = torch.stack([e_vectors, m_vectors], dim=1)
        current_input = generate_input(vectors, lengths, tokenized_text.size(1), DEVICE)

        with torch.no_grad():
            outputs = model(inputs_embeds=current_input, attention_mask=attention_mask, output_attentions=True)
        attention_weights = outputs.attentions

        for i in range(B):
            attn_e_all_layers = []
            for layer_attn in attention_weights:
                if args.mean_attention:
                    attn_e = layer_attn[i].mean(dim=0)[:, 0]
                else:
                    attn_e = layer_attn[i][:, :, 0]
                attn_e = attn_e.float().cpu().numpy().tolist()
                attn_e_all_layers.append(attn_e)

            token_list = tokenizer.convert_ids_to_tokens(tokenized_text[i][:lengths[i]].tolist())

            result.append({
                'text': texts[i],
                'tokens': token_list,
                'lengths': lengths[i],
                'best_vectors': vectors[i].float().cpu().numpy().tolist(),
                'attention_e': attn_e_all_layers
            })

        # Сохранение результатов
        if (idx + 1) % SAVE_EVERY == 0:
            with open(SAVE_PATH, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=4)
    
    # Сохранение результатов
    with open(SAVE_PATH, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)