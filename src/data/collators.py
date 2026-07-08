import torch
from torch.nn.utils.rnn import pad_sequence


def _targets(e_vectors, m_vectors):
    return {
        "e": torch.tensor(e_vectors, dtype=torch.float32),
        "m": torch.tensor(m_vectors, dtype=torch.float32),
    }


def _encode(texts, tokenizer, max_tokens):
    """Токенизировать список строк в батч."""
    input_ids = [
        tokenizer.encode(t, return_tensors="pt", max_length=max_tokens, truncation=True).reshape(-1)
        for t in texts
    ]
    lengths = [ids.shape[0] for ids in input_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = (input_ids != tokenizer.pad_token_id).long()
    return input_ids, attention_mask, lengths


def collate_alpaca(batch, tokenizer, max_tokens):
    texts = [item[0] for item in batch]
    instructions = [item[1] for item in batch]

    input_ids, attention_mask, lengths = _encode(texts, tokenizer, max_tokens)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "lengths": lengths,
        "labels": input_ids.clone(),
        "metainfo": {"texts": texts, "instructions": instructions},
    }


def collate_dolly(batch, tokenizer, max_tokens):
    texts = [item[0] for item in batch]
    instructions = [item[1] for item in batch]
    categories = [item[2] for item in batch]

    input_ids, attention_mask, lengths = _encode(texts, tokenizer, max_tokens)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "lengths": lengths,
        "labels": input_ids.clone(),
        "metainfo": {"texts": texts, "instructions": instructions, "categories": categories},
    }


def collate_alpaca_end2end(batch, tokenizer, max_tokens):
    texts = [item[0] for item in batch]
    instructions = [item[1] for item in batch]
    e_vectors = [item[2] for item in batch]
    m_vectors = [item[3] for item in batch]

    ins_input_ids, ins_attention_mask, ins_lengths = _encode(instructions, tokenizer, max_tokens)
    ans_input_ids, ans_attention_mask, ans_lengths = _encode(texts, tokenizer, max_tokens)
    return {
        "instruction": {
            "input_ids": ins_input_ids,
            "attention_mask": ins_attention_mask,
            "lengths": ins_lengths,
        },
        "answer": {
            "input_ids": ans_input_ids,
            "attention_mask": ans_attention_mask,
            "lengths": ans_lengths,
            "labels": ans_input_ids.clone(),
        },
        "targets": _targets(e_vectors, m_vectors),
        "metainfo": {"texts": texts, "instructions": instructions},
    }


def collate_dolly_end2end(batch, tokenizer, max_tokens):
    texts = [item[0] for item in batch]
    instructions = [item[1] for item in batch]
    categories = [item[2] for item in batch]
    e_vectors = [item[3] for item in batch]
    m_vectors = [item[4] for item in batch]

    ins_input_ids, ins_attention_mask, ins_lengths = _encode(instructions, tokenizer, max_tokens)
    ans_input_ids, ans_attention_mask, ans_lengths = _encode(texts, tokenizer, max_tokens)
    return {
        "instruction": {
            "input_ids": ins_input_ids,
            "attention_mask": ins_attention_mask,
            "lengths": ins_lengths,
        },
        "answer": {
            "input_ids": ans_input_ids,
            "attention_mask": ans_attention_mask,
            "lengths": ans_lengths,
            "labels": ans_input_ids.clone(),
        },
        "targets": _targets(e_vectors, m_vectors),
        "metainfo": {"texts": texts, "instructions": instructions, "categories": categories},
    }


def collate_noise(batch, tokenizer, max_tokens):
    texts = [item[0] for item in batch]
    e_vectors = [item[1] for item in batch]
    m_vectors = [item[2] for item in batch]

    input_ids, attention_mask, lengths = _encode(texts, tokenizer, max_tokens)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "lengths": lengths,
        "e_vectors": e_vectors,
        "m_vectors": m_vectors,
    }


def get_collator(dataset_type, task_type, tokenizer, max_tokens):
    """Фабрика коллатора по типу датасета и типу задачи."""
    if task_type == "end2end":
        if dataset_type == "alpaca":
            return lambda batch: collate_alpaca_end2end(batch, tokenizer, max_tokens)
        elif dataset_type == "dolly":
            return lambda batch: collate_dolly_end2end(batch, tokenizer, max_tokens)
        else:
            raise ValueError(f"Unknown dataset: {dataset_type}")
    elif task_type == "nar":
        if dataset_type == "alpaca":
            return lambda batch: collate_alpaca(batch, tokenizer, max_tokens)
        elif dataset_type == "dolly":
            return lambda batch: collate_dolly(batch, tokenizer, max_tokens)
        elif dataset_type == "noise":
            return lambda batch: collate_noise(batch, tokenizer, max_tokens)
        else:
            raise ValueError(f"Unknown dataset: {dataset_type}")
    else:
        raise ValueError(f"Unknown task type: {task_type}")
