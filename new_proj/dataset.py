import json
from torch.utils.data import Dataset


# Для датасета yahma/alpaca-cleaned
class AlpacaDataset(Dataset):
    def __init__(self, dataset):
        dataset = dataset.filter(lambda x: x['input'] == '' or x['input'] == 'None')
        dataset = dataset['train'].to_pandas()
        self.texts = dataset['output'].to_list()
        self.instructions = dataset['instruction'].to_list()
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx], self.instructions[idx]

# Для датасета databricks/databricks-dolly-15k
class DollyDataset(Dataset):
    def __init__(self, dataset):
        dataset = dataset['train'].to_pandas()
        self.texts = dataset['response'].to_list()
        self.instructions = dataset['instruction'].to_list()
        self.categories = dataset['category'].to_list()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.instructions[idx], self.categories[idx]

# Для зашумленных векторов
class NoiseDataset(Dataset):
    def __init__(self, path, threshold=0.9):
        with open(path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)

        data = [{'text': item['texts'], 'best_vectors': item['best_vectors']} for item in dataset if item['accuracy'] >= threshold]
        self.texts = [item['text'] for item in data]
        self.e_vectors = [item['best_vectors'][0] for item in data]
        self.v_vectors = [item['best_vectors'][1] for item in data]

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx], self.e_vectors[idx], self.v_vectors[idx]
    
# Фабрика для выбора датасета
def get_dataset(dataset_type, raw_dataset=None, path=None):
    if dataset_type == 'alpaca':
        return AlpacaDataset(raw_dataset)
    elif dataset_type == 'dolly':
        return DollyDataset(raw_dataset)
    elif dataset_type == 'noise':
        return NoiseDataset(path)
    else:
        raise ValueError(f'Unknown dataset: {dataset_type}')