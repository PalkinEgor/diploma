import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForCausalLM


# Модель с использованием кодовых книг
class CodeBooksModel(nn.Module):
    def __init__(self, encoder_name, decoder_name, e_code_books, m_code_books, dtype):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(encoder_name, torch_dtype=dtype, device_map='auto')
        self.decoder = AutoModelForCausalLM.from_pretrained(decoder_name, torch_dtype=dtype, device_map='auto')
        for param in self.decoder.parameters():
            param.requires_grad = False
        self.decoder.eval()
        self.e_classifier = nn.Linear(self.encoder.config.hidden_size, e_code_books.shape[0])
        self.m_classifier = nn.Linear(self.encoder.config.hidden_size, m_code_books.shape[0])
        self.e_code_books = torch.nn.Parameter(e_code_books.to(dtype=dtype))
        self.m_code_books = torch.nn.Parameter(m_code_books.to(dtype=dtype))
        self.dtype = dtype
    
    def forward(self, input_ids, attention_mask=None):
        x = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, -1, :]

        # получение вектора e
        e_logits = self.e_classifier(x)
        e_probs = F.gumbel_softmax(e_logits, tau=1.0, hard=True, dim=-1)
        e_code_book = e_probs @ self.e_code_books

        # получение вектора m
        m_logits = self.m_classifier(x)
        m_probs = F.gumbel_softmax(m_logits, tau=1.0, hard=True, dim=-1)
        m_code_book = m_probs @ self.m_code_books
        
        return e_code_book, m_code_book

# Полная модель с использованием кодовых книг (work in progress)
class FullCodeBooksModel(nn.Module):
    def __init__(self, encoder_codebook, decoder_name):
        super.__init__()

        self.encoder_codebook = encoder_codebook
        self.decoder = AutoModelForCausalLM.from_pretrained(decoder_name, torch_dtype=encoder_codebook.dtype, device_map='auto')
        for param in self.decoder.parameters():
            param.requires_grad = False
        self.decoder.eval()
