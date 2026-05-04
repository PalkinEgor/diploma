import json
from dataset import get_dataset
from sentence_transformers import SentenceTransformer

class TeacherEmbeddings:
    def __init__(self, config, dtype):
        self.dataset = get_dataset(config['dataset']['type'], config['task_type'], config['dataset']['path'])
        self.model = SentenceTransformer(config['model']['path'], torch_dtype=dtype)
        self.batch_size = config['training']['batch_size']
        self.normalize = config['training']['normalize']
        self.save_path = config['logging']['save_path']
        self.instructions = self.dataset.instructions

    def get_embeddings(self):
        all_embeddings = []
        for i in range(0, len(self.instructions), self.batch_size):
            batch = self.instructions[i:i + self.batch_size]
            emb = self.model.encode(
                batch,
                convert_to_tensor=False,
                normalize_embeddings=self.normalize,
                show_progress_bar=False
            )
            all_embeddings.extend(emb.tolist())
        result = []
        for i in range(len(all_embeddings)):
            result.append({
                'instruction': self.instructions[i],
                'teacher_embedding': all_embeddings[i]
            })
        with open(self.save_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=4)      
