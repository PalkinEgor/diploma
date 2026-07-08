import json
from torch.utils.data import Dataset
from datasets import load_from_disk


class AlpacaDataset(Dataset):
    """yahma/alpaca-cleaned для задачи векторизации (этап 1)."""

    def __init__(self, path, max_samples):
        raw_dataset = load_from_disk(path)
        dataset = raw_dataset.filter(lambda x: x["input"] == "" or x["input"] == "None")
        dataset = dataset["train"].to_pandas()
        self.texts = dataset["output"].to_list()[:max_samples]
        self.instructions = dataset["instruction"].to_list()[:max_samples]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.instructions[idx]


class DollyDataset(Dataset):
    """databricks-dolly-15k для задачи векторизации (этап 1)."""

    def __init__(self, path, max_samples):
        raw_dataset = load_from_disk(path)
        dataset = raw_dataset["train"].to_pandas()
        self.texts = dataset["response"].to_list()[:max_samples]
        self.instructions = dataset["instruction"].to_list()[:max_samples]
        self.categories = dataset["category"].to_list()[:max_samples]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.instructions[idx], self.categories[idx]


class DollyDatasetEnd2End(Dataset):
    """databricks-dolly-15k для задачи end2end (этап 2)."""

    def __init__(self, path, threshold=0.9):
        with open(path, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        data = [
            {
                "text": item["texts"],
                "instruction": item["instructions"],
                "category": item["categories"],
                "best_vectors": item["best_vectors"],
            }
            for item in dataset
            if item["accuracy"] >= threshold
        ]
        self.texts = [item["text"] for item in data]
        self.instructions = [item["instruction"] for item in data]
        self.categories = [item["category"] for item in data]
        self.e_vectors = [item["best_vectors"][0] for item in data]
        self.m_vectors = [item["best_vectors"][1] for item in data]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return (
            self.texts[idx],
            self.instructions[idx],
            self.categories[idx],
            self.e_vectors[idx],
            self.m_vectors[idx],
        )


class AlpacaDatasetEnd2End(Dataset):
    """yahma/alpaca-cleaned для задачи end2end (этап 2)."""

    def __init__(self, path, threshold=0.9):
        with open(path, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        data = [
            {
                "text": item["texts"],
                "instruction": item["instructions"],
                "best_vectors": item["best_vectors"],
            }
            for item in dataset
            if item["accuracy"] >= threshold
        ]
        self.texts = [item["text"] for item in data]
        self.instructions = [item["instruction"] for item in data]
        self.e_vectors = [item["best_vectors"][0] for item in data]
        self.m_vectors = [item["best_vectors"][1] for item in data]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return (
            self.texts[idx],
            self.instructions[idx],
            self.e_vectors[idx],
            self.m_vectors[idx],
        )


class NoiseDataset(Dataset):
    def __init__(self, path, threshold=0.9):
        with open(path, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        data = [
            {"text": item["texts"], "best_vectors": item["best_vectors"]}
            for item in dataset
            if item["accuracy"] >= threshold
        ]
        self.texts = [item["text"] for item in data]
        self.e_vectors = [item["best_vectors"][0] for item in data]
        self.v_vectors = [item["best_vectors"][1] for item in data]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.e_vectors[idx], self.v_vectors[idx]


def get_dataset(dataset_type, task_type, path, max_samples=None, threshold=0.9):
    """Фабрика датасета по типу датасета и типу задачи."""
    if task_type == "nar":
        if dataset_type == "alpaca":
            return AlpacaDataset(path, max_samples)
        elif dataset_type == "dolly":
            return DollyDataset(path, max_samples)
        elif dataset_type == "noise":
            return NoiseDataset(path, threshold)
        else:
            raise ValueError(f"Unknown dataset: {dataset_type}")
    elif task_type == "end2end":
        if dataset_type == "alpaca":
            return AlpacaDatasetEnd2End(path, threshold)
        elif dataset_type == "dolly":
            return DollyDatasetEnd2End(path, threshold)
        else:
            raise ValueError(f"Unknown dataset: {dataset_type}")
    else:
        raise ValueError(f"Unknown task type: {task_type}")
