import re
import os
from underthesea import word_tokenize
from gensim.models import Word2Vec
from config import *
import numpy as np

class Utils:
    def __init__(self):
        # nothing special to init for now
        pass
    def load_data_from_source(self, source):
        titles = []
        abstracts = []
        articles = []
        count = 0
        for filename in os.listdir(source):
            if (count == count_max):
                break
            file_path = os.path.join(source, filename)   # <-- full path
            if os.path.isfile(file_path):
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    article = []
                    for i, line in enumerate(f, start=1):
                        text = re.sub(r'[()]', '', line.strip().lower())      # remove all ( )
                        text = re.sub(r'[.?!:;,…]+$', '', text)
                        if i == 1: 
                            titles.append(text)
                        elif (i == 3):
                            abstracts.append(text)
                        article.append(text)
                    articles.append(' '.join(article))
            count+=1
        return titles, abstracts, articles
    
    def split_sentences(self, text):
        # Very simple splitter: split on ., ! or ? and drop empties
        sents = re.sub(r"[\[\]\(\)\'\",.!?:]", " ", text)
        return sents.split()
    
    def load_corpus(self):
        titles, _, articles = self.load_data_from_source(data_path)
        
        corpus = []

        # Articles contain many sentences, so break them up first
        for article in articles:
            corpus.append(self.split_sentences(article))
        return corpus, titles, articles
    
    def word2vec_model(self, corpus):

        try:
            # Try to load the existing model
            model = Word2Vec.load(word2vec_model_path)
            print("Loaded existing model:", word2vec_model_path)
        except (FileNotFoundError, ValueError):
            # File doesn't exist or is corrupted – train a new one
            print("Model not found or invalid. Training a new model...")
            model = Word2Vec(
                corpus,
                vector_size=embedding_size,
                window=5,
                min_count=1,
                sg=1,          # skip-gram (better for small data)
                workers=4
            )
            model.save(word2vec_model_path)
            print("New model saved to:", word2vec_model_path)

        return model
    
    def process_embeddings(self, word2vec_model):
        word_vectors = word2vec_model.wv
        stoi = {w: i+4 for i, w in enumerate(word_vectors.key_to_index)}
        specials = ['<pad>', '<sos>', '<eos>', '<unk>']
        for i,s in enumerate(specials): stoi[s] = i
        itos = {i:s for s,i in stoi.items()}

        embedding_dim = embedding_size
        embedding_matrix = np.zeros((len(stoi), embedding_dim))
        for w, idx in stoi.items():
            if w in word_vectors:
                embedding_matrix[idx] = word_vectors[w]
            else:
                embedding_matrix[idx] = np.random.normal(scale=0.6, size=(embedding_dim,))
        
        return stoi, itos, embedding_matrix
    
    def encode_sentence(tokens, stoi, max_len):
        ids = [stoi.get("<sos>")]
        ids += [stoi.get(t, stoi["<unk>"]) for t in tokens]
        ids.append(stoi.get("<eos>"))
        ids = ids[:max_len] + [stoi["<pad>"]] * max(0, max_len - len(ids))
        return ids
    
    