import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import generate_input
from transformers import AutoModel, AutoModelForCausalLM


# Модуль квантизации
class GumbelVectorQuantizer(nn.Module):
    def __init__(self, dim, G, V, code_books_init, dtype, tau):
        super().__init__()

        assert dim % G == 0, f'dim ({dim}) must be divisible by G ({G})'

        self.G = G
        self.V = V
        self.code_dim = dim // G
        self.tau = tau
        self.proj = nn.Linear(dim, G * V, dtype=dtype)

        centroids = code_books_init
        centroids = code_books_init.reshape(V, G, dim // G) # (V, G, d/G)
        centroids = centroids.permute(1, 0, 2) # (G, V, d/G)
        self.code_books = nn.Parameter(centroids.to(dtype=dtype))

        self.out_proj = nn.Linear(G * self.code_dim, dim, dtype=dtype)

    def forward(self, x):
        B, _ = x.shape

        logits = self.proj(x).view(B, self.G, self.V) # (B, G, V)
        probs = F.gumbel_softmax(logits, tau=self.tau, hard=True, dim=-1) # (B, G, V)
        codes = torch.einsum('bgv,gvd->bgd', probs, self.code_books) # (B, G, V) @ (G, V, code_dim) -> (B, G, code_dim)
        codes = codes.reshape(B, self.G * self.code_dim)
        out = self.out_proj(codes)

        # diversity loss
        soft_probs = F.softmax(logits, dim=-1)
        avg_probs = soft_probs.mean(dim=0)
        diversity_loss = (avg_probs * torch.log(avg_probs + 1e-7)).sum()
        diversity_loss = diversity_loss / (self.G * self.V)

        return out, diversity_loss


# Энкодер модели с использованием кодовых книг
class EncoderCodeBooksModel(nn.Module):
    def __init__(self, encoder_name, G, V, e_code_books_init, m_code_books_init, tau, dtype, m_vector, mean_pooling):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(encoder_name, torch_dtype=dtype)
        dim = self.encoder.config.hidden_size
        self.e_quantizer = GumbelVectorQuantizer(dim, G, V, e_code_books_init, dtype, tau)
        self.m_vector = m_vector
        self.mean_pooling = mean_pooling
        if m_vector:
            self.m_quantizer = GumbelVectorQuantizer(dim, G, V, m_code_books_init, dtype, tau)
        else:
            self.m_proj = nn.Linear(dim, dim, dtype=dtype)

    def forward(self, input_ids, attention_mask=None):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        if self.mean_pooling:
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            x = (hidden * mask).sum(dim=1)
            x = x / mask.sum(dim=1).clamp(min=1e-6)
        else:
            lengths = attention_mask.sum(dim=1) - 1
            x = hidden[torch.arange(hidden.size(0), device=hidden.device), lengths]

        e_code_book, e_diversity_loss = self.e_quantizer(x)
        if self.m_vector:
            m_code_book, m_diversity_loss = self.m_quantizer(x)
            return e_code_book, m_code_book, e_diversity_loss, m_diversity_loss
        else:
            m_code_book = self.m_proj(x)
            return e_code_book, m_code_book, e_diversity_loss


# Полная модель с использованием кодовых книг
class FullCodeBooksModel(nn.Module):
    def __init__(self, encoder_model, decoder_name, tokenizer, dtype, m_vector):
        super().__init__()

        self.encoder_model = encoder_model
        self.decoder = AutoModelForCausalLM.from_pretrained(decoder_name, torch_dtype=dtype)        

        self.pad_token_id = tokenizer.pad_token_id
        self.decoder_pad_emb = self.decoder.get_input_embeddings().weight[self.pad_token_id]

        self.dtype = dtype
        self.m_vector = m_vector

    def train(self, mode=True):
        super().train(mode)
        for param in self.decoder.parameters():
            param.requires_grad = False
        self.decoder.eval()
        return self

    def forward(self, tokenized_instruction, tokenized_answers, instruction_attention_mask, answer_attention_mask, answer_lengths):
        if self.m_vector:
            e_vector, m_vector, e_diversity_loss, m_diversity_loss = self.encoder_model(tokenized_instruction, instruction_attention_mask)
        else:
            e_vector, m_vector, e_diversity_loss = self.encoder_model(tokenized_instruction, instruction_attention_mask)
        
        vectors = torch.stack([e_vector, m_vector], dim=1)
        vectors = vectors.to(dtype=self.decoder.dtype)
        current_input = generate_input(
            vectors,
            answer_lengths,
            tokenized_answers.size(1),
            self.decoder_pad_emb,
            vectors.device
        )
        logits = self.decoder(inputs_embeds=current_input, attention_mask=answer_attention_mask).logits

        if self.m_vector:
            return logits, e_diversity_loss, m_diversity_loss
        else:
            return logits, e_diversity_loss