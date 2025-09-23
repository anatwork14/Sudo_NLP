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
        english, vietnamese = [], []
        
        en_re = re.compile(r"<s id='en\d+'>(.*?)</s>")
        vn_re = re.compile(r"<s id='vn\d+'>(.*?)</s>")
        
        for filename in os.listdir(source):
            file_path = os.path.join(source, filename)   # <-- full path
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    block_en, block_vn = [], []
                    for line in f:
                        # collect all matches in the current line
                        ens = en_re.findall(line)
                        vns = vn_re.findall(line)
                        if ens: english.append(ens)
                        if vns: vietnamese.append(vns)
        return english, vietnamese

    def preprocess(self, data):
        return [
        [word_tokenize(sentence[0].lower(), format="text")]  # keep the same nested structure
        for sentence in data]
        
    def tokenize(self, data):
        tokenized_data = []
        for row in data:
            sentence = row[0]
            # remove ", - characters
            cleaned = re.sub(r'["\-]', '', sentence)
            # split by space
            tokens = cleaned.split()
            tokenized_data.append(tokens)
        return tokenized_data
    
    def load_corpus(self):
        english, vietnamese = self.load_data_from_source(source)
        vietnamese = self.preprocess(vietnamese)
        tokenized_english = self.tokenize(english)
        tokenized_vietnamese = self.tokenize(vietnamese)
        
        corpus = tokenized_english + tokenized_vietnamese
        return corpus, tokenized_english, tokenized_vietnamese
    
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
    
    