import torch
import random
import math
from utils import generate_input
from metrics import Metrics


# Класс для обучения прото-токенов
class NARfit:
    def __init__(self, model, tokenizer, device, hyperparams):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.hyperparams = hyperparams
        self.pad_emb = model.get_input_embeddings().weight[tokenizer.pad_token_id].to(self.device)

    def train_batch(self, batch, maxiter, threshold):
        # Парсинг батча
        tokenized_text = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        lengths = batch['lengths']
        labels = batch['labels'].to(self.device)

        # Создание обучаемых векторов e и m
        B = tokenized_text.shape[0]
        vectors = torch.nn.Parameter(
            torch.randn(
                B, 2, self.model.config.hidden_size, 
                device=self.device, 
                dtype=self.model.dtype))
        optimizer = torch.optim.AdamW(
            [vectors], 
            lr=self.hyperparams['lr'], 
            betas=self.hyperparams['betas'], 
            weight_decay=self.hyperparams['weight_decay'])
        
        # Запускаем maxiter итераций обучения
        max_accuracy = [0.0] * B
        best_vectors = [None] * B
        last_iter = [None] * B
        for iter in range(maxiter):
            optimizer.zero_grad()

            # Считаем лосс и делаем предсказания
            current_input = generate_input(
                vectors, 
                lengths, 
                tokenized_text.size(1), 
                self.pad_emb, 
                self.device)
            logits = self.model(inputs_embeds=current_input, attention_mask=attention_mask).logits
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                labels.view(-1), 
                ignore_index=self.tokenizer.pad_token_id)
            pred = logits.argmax(dim=-1)

            # Считаем метрики и сохраняем лучшие вектора
            for i in range(B):
                current_len = lengths[i]
                current_pred = pred[i, :current_len]
                current_labels = labels[i, :current_len]

                accuracy = Metrics.calculate_accuracy(current_labels, current_pred)
                if accuracy > max_accuracy[i]:
                    max_accuracy[i] = accuracy
                    best_vectors[i] = vectors[i].detach().clone()

                if last_iter[i] is None and accuracy >= threshold:
                    last_iter[i] = iter
            
            # Пропускаем итерацию если батч обучен
            if all(a >= threshold for a in max_accuracy):
                break

            loss.backward()
            optimizer.step()

        for i in range(B):
            if last_iter[i] is None:
                last_iter[i] = maxiter
                
        return max_accuracy, best_vectors, last_iter, B

# Для моедели на основе кодовых книг    
class CodeBookFit:
    def __init__(self, full_model, optimizer, tokenizer, device, hyperparams):
        self.full_model = full_model
        self.optimizer = optimizer
        self.tokenizer = tokenizer
        self.device = device
        self.hyperparams = hyperparams
    
    def train_batch(self, batch):
        self.full_model.train()
        self.optimizer.zero_grad()
        accuracies = []     

        # Парсим батч
        tokenized_instruction = batch['instruction']['input_ids'].to(self.device)
        tokenized_answers = batch['answer']['input_ids'].to(self.device)
        instruction_attention_mask = batch['instruction']['attention_mask'].to(self.device)
        answer_attention_mask = batch['answer']['attention_mask'].to(self.device)
        answer_lengths = batch['answer']['lengths']
        labels = batch['answer']['labels'].to(self.device)
        B = tokenized_instruction.shape[0]

        # Получаем логиты
        logits = self.full_model(
            tokenized_instruction,
            tokenized_answers,
            instruction_attention_mask,
            answer_attention_mask,
            answer_lengths
        )

        # Считаем лосс
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), 
            labels.view(-1), 
            ignore_index=self.tokenizer.pad_token_id
        )
        pred = logits.argmax(dim=-1)

        # Считаем метрики
        for i in range(B):
            current_len = answer_lengths[i]
            current_pred = pred[i, :current_len]
            current_labels = labels[i, :current_len]
            accuracy = Metrics.calculate_accuracy(current_labels, current_pred)
            accuracies.append(accuracy)

        loss.backward()
        self.optimizer.step()

        return loss.item(), accuracies
    
    def run_batch(self, batch):
        self.full_model.eval()
        accuracies = []

        # Парсим батч
        tokenized_instruction = batch['instruction']['input_ids'].to(self.device)
        tokenized_answers = batch['answer']['input_ids'].to(self.device)
        instruction_attention_mask = batch['instruction']['attention_mask'].to(self.device)
        answer_attention_mask = batch['answer']['attention_mask'].to(self.device)
        answer_lengths = batch['answer']['lengths']
        labels = batch['answer']['labels'].to(self.device)
        B = tokenized_instruction.shape[0]

        with torch.no_grad():
            logits = self.full_model(
                tokenized_instruction,
                tokenized_answers,
                instruction_attention_mask,
                answer_attention_mask,
                answer_lengths
            )

            # Считаем лосс
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=self.tokenizer.pad_token_id
            )
            pred = logits.argmax(dim=-1)

        # Считаем метрики
        for i in range(B):
            current_len = answer_lengths[i]
            current_pred = pred[i, :current_len]
            current_labels = labels[i, :current_len]
            accuracy = Metrics.calculate_accuracy(current_labels, current_pred)
            accuracies.append(accuracy)

        return loss.item(), accuracies

# Класс для зашумленных векторов    
class NoiseExp:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.pad_emb = model.get_input_embeddings().weight[tokenizer.pad_token_id]
        self.alpha = [0.00, 0.05, 0.10, 0.20, 0.50, 1.00]
        self.noise_types = ['gaussian', 'uniform', 'sinusoidal', 'exponential']

    # Получаем вектор шума
    def get_noise(self, shape, alpha, noise_type, ref_vector):
        if noise_type == 'gaussian':
            noise = torch.randn(shape, device=self.device, dtype=self.model.dtype)

        elif noise_type == 'uniform':
            noise = torch.rand(shape, device=self.device, dtype=self.model.dtype) * 2 - 1 # равномерное распределние в таких границах [-1, 1]

        elif noise_type == 'exponential':
            noise = torch.distributions.Exponential(1.0).sample(shape).to(device=self.device, dtype=self.model.dtype)
            noise *= torch.randint(0, 2, shape, device=self.device, dtype=self.model.dtype) * 2 - 1

        elif noise_type == 'sinusoidal':
            size = shape[-1]
            k = random.choice(range(4, 33))
            freq = 2 * math.pi * k / size            
            phase = random.uniform(0, math.pi * 2)

            x = torch.arange(size, device=self.device, dtype=self.model.dtype)
            noise = torch.sin(freq * x + phase).to(device=self.device, dtype=self.model.dtype)

        else:
            noise = torch.randn(shape, device=self.device, dtype=self.model.dtype)

        # Нормализация
        norm = torch.norm(noise, dtype=self.model.dtype)
        if norm > 0:
            noise = noise / norm
        noise = noise * torch.norm(ref_vector, dtype=self.model.dtype) * alpha

        return noise
    
    def run_batch(self, batch):
        # Парсинг батча
        tokenized_text = batch['input_ids'].to(device=self.device)
        attention_mask = batch['attention_mask'].to(device=self.device)        
        lengths = batch['lengths']
        e_vectors_orig = torch.tensor(batch['e_vectors'], device=self.device, dtype=self.model.dtype)
        m_vectors = torch.tensor(batch['m_vectors'], device=self.device, dtype=self.model.dtype)
        B = tokenized_text.shape[0]

        result = []
        for a in self.alpha:
            for noise_type in self.noise_types:
                # Добавляем шум
                e_vectors = e_vectors_orig.clone()
                for i in range(B):
                    noise = self.get_noise(e_vectors[i].shape, a, noise_type, e_vectors[i])
                    e_vectors[i] += noise

                # Делаем forward pass
                vectors = torch.stack([e_vectors, m_vectors], dim=1)
                current_input = generate_input(vectors, lengths, tokenized_text.size(1), self.pad_emb, self.device)
                with torch.no_grad():
                    logits = self.model(inputs_embeds=current_input, attention_mask=attention_mask).logits
                    pred = logits.argmax(dim=-1)
                
                # Cчитаем метрики
                for i in range(B):
                    current_len = lengths[i]
                    current_pred = pred[i, :current_len]
                    current_labels = tokenized_text[i, :current_len]
                    accuracy = Metrics.calculate_accuracy(current_labels, current_pred)
                    
                    item = {'alpha': a, 'noise_type': noise_type, 'accuracy': accuracy}
                    result.append(item)

        return result