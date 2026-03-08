from torch.nn.utils.rnn import pad_sequence


# Для датасета yahma/alpaca-cleaned
def collate_alpaca(batch, tokenizer, max_tokens):
    texts = [item[0] for item in batch]
    instructions = [item[1] for item in batch]

    input_ids = [tokenizer.encode(text, return_tensors='pt', max_length=max_tokens, truncation=True).reshape(-1) for text in texts]
    lengths = [text.shape[0] for text in input_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = (input_ids != tokenizer.pad_token_id).long()

    return {
        'texts': texts,
        'instructions': instructions,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'lengths': lengths
    }

# Для датасета databricks/databricks-dolly-15k
def collate_dolly(batch, tokenizer, max_tokens):
    texts = [item[0] for item in batch]
    instructions = [item[1] for item in batch]
    categories = [item[2] for item in batch]

    input_ids = [tokenizer.encode(text, return_tensors='pt', max_length=max_tokens, truncation=True).reshape(-1) for text in texts]
    lengths = [text.shape[0] for text in input_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = (input_ids != tokenizer.pad_token_id).long()

    return {
        'texts': texts,
        'instructions': instructions,
        'categories': categories,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'lengths': lengths
    }