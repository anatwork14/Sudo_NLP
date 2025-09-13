import re
from underthesea import word_tokenize
from config import * 
from gensim.models import Word2Vec
import torch
import numpy as np

class Utils: 
    def read_file(path, encoding):
            content = []
            with open(path, "r", encoding=encoding, errors="ignore") as f:
                for i, line in enumerate(f, start=1):
                    if i <= 2:   # skip first 2 lines
                        continue
                    low = line.lower()
                    low = re.sub(r"[^0-9a-zA-ZÀ-Ỹà-ỹ\s]", "", low)
                    if "mục lục" in low or "dịch giả" in low:
                        break
                    content.append(low.strip())
            return " ".join(content)
    
    def tokenize(text): 
        return word_tokenize(text)
    
    def embedding_data(self, corpus):
        self.w2v_model = Word2Vec(
            sentences=corpus,   # tokenized text
            vector_size=100,    # embedding dimension
            window=5,           # context window
            min_count=2,        # ignore rare words
            workers=4           # parallel training
        )
        self.vocab = list(self.w2v_model.wv.index_to_key)
        self.stoi = {w: i for i, w in enumerate(self.vocab)}
        self.itos = {i: w for w, i in self.stoi.items()}

        self.embedding_dim = self.w2v_model.vector_size
        self.vocab_size = len(self.vocab)
    
        return self.w2v_model
    
    
    def get_embedding_matrix(self):
        embedding_matrix = np.zeros((self.vocab_size, self.embedding_dim))
        for word, idx in self.stoi.items():
            embedding_matrix[idx] = self.w2v_model.wv[word]

        self.embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float32)
        return self.embedding_matrix
        