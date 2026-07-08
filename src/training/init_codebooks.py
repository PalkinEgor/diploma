import numpy as np
import torch
from sklearn.cluster import KMeans


class CodeBooksInit:
    """Инициализирует центроиды e/m кодовых книг через KMeans."""

    def __init__(self, dataset, code_books, seed, dtype, norm=False):
        self.dataset = dataset
        self.code_books = code_books      # число центроидов V
        self.seed = seed
        self.norm = norm
        self.dtype = dtype

        self.e_code_books = None
        self.m_code_books = None

    def init_codebooks(self):
        e_vectors = np.asarray(self.dataset.e_vectors)
        m_vectors = np.asarray(self.dataset.m_vectors)
        if self.norm:
            e_vectors = e_vectors / (np.linalg.norm(e_vectors, axis=1, keepdims=True) + 1e-8)
            m_vectors = m_vectors / (np.linalg.norm(m_vectors, axis=1, keepdims=True) + 1e-8)

        kmeans_e = KMeans(n_clusters=self.code_books, random_state=self.seed)
        kmeans_e.fit(e_vectors)
        self.e_code_books = torch.tensor(kmeans_e.cluster_centers_, dtype=self.dtype)

        kmeans_m = KMeans(n_clusters=self.code_books, random_state=self.seed)
        kmeans_m.fit(m_vectors)
        self.m_code_books = torch.tensor(kmeans_m.cluster_centers_, dtype=self.dtype)

        return self.e_code_books, self.m_code_books
