import torch
import numpy as np
from sklearn.cluster import KMeans

class CodeBooksInit:
    def __init__(self, dataset, code_books, seed, dtype, norm):
        self.dataset = dataset
        self.code_books = code_books
        self.seed = seed
        self.norm = norm
        self.dtype = dtype

        self.e_code_books = None
        self.m_code_books = None
    
    def init_codebooks(self):
        # Нормализация
        e_vectors = self.dataset.e_vectors
        m_vectors = self.dataset.m_vectors
        if self.norm:
            e_vectors = e_vectors / (np.linalg.norm(e_vectors, axis=1, keepdims=True) + 1e-8)
            m_vectors = m_vectors / (np.linalg.norm(m_vectors, axis=1, keepdims=True) + 1e-8)

        # Инициализируем e кодовые книги
        kmeans_e = KMeans(n_clusters=self.code_books, random_state=self.seed)
        kmeans_e.fit(e_vectors)
        self.e_code_books = torch.tensor(kmeans_e.cluster_centers_)

        # Инициализируем m кодовые книги
        k_means_m = KMeans(n_clusters=self.code_books, random_state=self.seed)
        k_means_m.fit(m_vectors)
        self.m_code_books = torch.tensor(k_means_m.cluster_centers_)

        return self.e_code_books, self.m_code_books