import torch
from utils import generate_input
from metrics import calculate_accuracy


class NARfit:
    def __init__(self, model, tokenizer, device, hyperparams):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.hyperparams = hyperparams
        self.pad_emb = model.get_input_embeddings().weight[tokenizer.pad_token_id]

    def train_batch(self, batch, maxiter, threshold):
        # Парсинг батча
        tokenized_text = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        lengths = batch['lengths']
        labels = tokenized_text.clone()

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
        last_iter = 0
        for iter in range(maxiter):
            last_iter = iter
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

                accuracy = calculate_accuracy(current_labels, current_pred)
                if accuracy > max_accuracy[i]:
                    max_accuracy[i] = accuracy
                    best_vectors[i] = vectors[i].detach().clone()
            
            # Пропускаем итерацию если батч обучен
            if all(a >= threshold for a in max_accuracy):
                break

            loss.backward()
            optimizer.step()

        return max_accuracy, best_vectors, last_iter