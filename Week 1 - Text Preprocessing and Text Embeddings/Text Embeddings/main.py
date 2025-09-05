from data_loader import *
from tokenizer import * 
from config import *
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import random

data = loader(PATH)

word2vec_model = tokenize(data)

embedding_matrix = word2vec_model.wv.vectors
print("Embedding matrix shape:", embedding_matrix.shape)

word_to_index = {word: idx for idx, word in enumerate(word2vec_model.wv.index_to_key)}
print("First 30 tokens:",word2vec_model.wv.index_to_key[:30])  # first 30 tokens

for i, word in enumerate(list(word_to_index.keys())[:5]):
    print(f"Embedding for '{word}':")
    print(embedding_matrix[word_to_index[word]])

word2vec_model.wv.most_similar('một')
print("Try to compute 'nam'-'nữ'+'bé':",word2vec_model.wv.most_similar(positive=['nam', 'nữ'], negative=['bé']))

word_freq = {word: word2vec_model.wv.get_vecattr(word, "count") for word in word2vec_model.wv.index_to_key}

# Top 10 most frequent
top10 = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
print("Top 10 words frequency:",top10)

words = ["một", "hai", "ba"]
embeddings = [word2vec_model.wv[w] for w in words if w in word2vec_model.wv]

df = pd.DataFrame(embeddings, index=words)
print('Display 3 words embeddings:', words)
print(df.head())


words = list(word2vec_model.wv.index_to_key)

# Select 10 random words
random_words = random.sample(words, 10)
random_words


embeddings = [word2vec_model.wv[w] for w in random_words if w in word2vec_model.wv]

df = pd.DataFrame(embeddings, index=random_words)
print(df)

# Plot heatmap
plt.figure(figsize=(16, 8))
sns.heatmap(df, cmap="coolwarm", annot=False, cbar=True)
plt.title("Word Embeddings Heatmap")
plt.xlabel("Embedding dimensions")
plt.ylabel("Words")
plt.show()