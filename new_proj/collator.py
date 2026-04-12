from torch.nn.utils.rnn import pad_sequence


# Для датасета yahma/alpaca-cleaned, для задачи векторизации
def collate_alpaca(batch, tokenizer, max_tokens):
    texts = [item[0] for item in batch]
    instructions = [item[1] for item in batch]

    input_ids = [tokenizer.encode(text, return_tensors='pt', max_length=max_tokens, truncation=True).reshape(-1) for text in texts]
    lengths = [text.shape[0] for text in input_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = (input_ids != tokenizer.pad_token_id).long()

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'lengths': lengths,
        'labels': input_ids.clone(),
        'metainfo': {
            'texts': texts,
            'instructions': instructions
        }
    }

# Для датасета databricks/databricks-dolly-15k, для задачи векторизации
def collate_dolly(batch, tokenizer, max_tokens):
    texts = [item[0] for item in batch]
    instructions = [item[1] for item in batch]
    categories = [item[2] for item in batch]

    input_ids = [tokenizer.encode(text, return_tensors='pt', max_length=max_tokens, truncation=True).reshape(-1) for text in texts]
    lengths = [text.shape[0] for text in input_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = (input_ids != tokenizer.pad_token_id).long()

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'lengths': lengths,
        'labels': input_ids.clone(),
        'metainfo': {
            'texts': texts,
            'instructions': instructions,
            'categories': categories
        }
    }

# # Для датасета yahma/alpaca-cleaned, для задачи end2end
# def collate_alpaca_end2end(batch, tokenizer, max_tokens):
#     texts = [item[0] for item in batch]
#     instructions = [item[1] for item in batch]
    
#     input_ids = [tokenizer.encode(instruction, return_tensors='pt', max_length=max_tokens, truncation=True).reshape(-1) for instruction in instructions]
#     lengths = [text.shape[0] for text in input_ids]
#     input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
#     attention_mask = (input_ids != tokenizer.pad_token_id).long()

#     return {
#         'input_ids': input_ids,
#         'attention_mask': attention_mask,
#         'lengths': lengths,
#         'labels': input_ids.clone(),
#         'metainfo': {
#             'texts': texts,
#             'instructions': instructions
#         }
#     }

# Для датасета databricks/databricks-dolly-15k, для задачи end2end
def collate_dolly_end2end(batch, tokenizer, max_tokens):
    texts = [item[0] for item in batch]
    instructions = [item[1] for item in batch]
    categories = [item[2] for item in batch]

    # Готовим интсрукции
    ins_input_ids = [tokenizer.encode(instruction, return_tensors='pt', max_length=max_tokens, truncation=True).reshape(-1) for instruction in instructions]
    ins_lengths = [text.shape[0] for text in ins_input_ids]
    ins_input_ids = pad_sequence(ins_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    ins_attention_mask = (ins_input_ids != tokenizer.pad_token_id).long()

    # Готовим ответ
    ans_input_ids = [tokenizer.encode(text, return_tensors='pt', max_length=max_tokens, truncation=True).reshape(-1) for text in texts]
    ans_lengths = [text.shape[0] for text in ans_input_ids]
    ans_input_ids = pad_sequence(ans_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    ans_attention_mask = (ans_input_ids != tokenizer.pad_token_id).long()

    return {
        'instruction': {
            'input_ids': ins_input_ids,
            'attention_mask': ins_attention_mask,
            'lengths': ins_lengths
        },
        'answer': {
            'input_ids': ans_input_ids,
            'attention_mask': ans_attention_mask,
            'lengths': ans_lengths,
            'labels': ans_input_ids.clone()
        },
        'metainfo': {
            'texts': texts,
            'instructions': instructions,
            'categories': categories
        }
    }

# Для зашумленных векторов
def collate_noise(batch, tokenizer, max_tokens):
    texts = [item[0] for item in batch]
    e_vectors = [item[1] for item in batch]
    m_vectors = [item[2] for item in batch]

    input_ids = [tokenizer.encode(text, return_tensors='pt', max_length=max_tokens, truncation=True).reshape(-1) for text in texts]
    lengths = [text.shape[0] for text in input_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = (input_ids != tokenizer.pad_token_id).long()

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'lengths': lengths,
        'e_vectors': e_vectors,
        'm_vectors': m_vectors
    }

# Фабрика для выбора коллатора
def get_collator(dataset_type, tokenizer, max_tokens):
    if dataset_type == 'alpaca':
        return lambda batch: collate_alpaca(batch, tokenizer, max_tokens)
    elif dataset_type == 'dolly':
        return lambda batch: collate_dolly(batch, tokenizer, max_tokens)
    elif dataset_type == 'noise':
        return lambda batch: collate_noise(batch, tokenizer, max_tokens)
    else:
        raise ValueError(f'Unknown dataset: {dataset_type}')