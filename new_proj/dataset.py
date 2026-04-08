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
    
# Фабрика для выбора датасета
def get_dataset(dataset_type, raw_dataset):
    if dataset_type == 'alpaca':
        return AlpacaDataset(raw_dataset)
    elif dataset_type == 'dolly':
        return DollyDataset(raw_dataset)
    else:
        raise ValueError(f'Unknown dataset: {dataset_type}')