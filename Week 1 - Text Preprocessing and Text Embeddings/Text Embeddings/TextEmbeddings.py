get_ipython().system('pip install -U gensim')

with open('/content/drive/MyDrive/SudoCode/Week 1 - Text Embeddings/viwik19.txt', encoding='utf-16') as f:
    data = f.read()

from nltk import word_tokenize
import pandas as pd
import nltk
import random
nltk.download('punkt_tab')
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from gensim.models import Word2Vec


tokens = word_tokenize(data)

sentences = [tokens]

model = Word2Vec(
    sentences,
    vector_size=100,
    window=5,
    min_count=5,
    sg=1,
    workers=4,
    epochs=10,
    compute_loss=True 
)

model.build_vocab(sentences)
model.train(
        sentences,
        total_examples=model.corpus_count,
        epochs=10
)

embedding_matrix = model.wv.vectors
print("Embedding matrix shape:", embedding_matrix.shape)

word_to_index = {word: idx for idx, word in enumerate(model.wv.index_to_key)}
print(model.wv.index_to_key[:30])  # first 30 tokens

for i, word in enumerate(list(word_to_index.keys())[:5]):
    print(f"Embedding for '{word}':")
    print(embedding_matrix[word_to_index[word]])

model.wv.most_similar('một')
model.wv.most_similar(positive=['nam', 'nữ'], negative=['bé'])

word_freq = {word: model.wv.get_vecattr(word, "count") for word in model.wv.index_to_key}

# Top 10 most frequent
top10 = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
print(top10)

words = ["một", "hai", "ba"]
embeddings = [model.wv[w] for w in words if w in model.wv]

df = pd.DataFrame(embeddings, index=words)
print(df.head())


words = list(model.wv.index_to_key)

# Select 10 random words
random_words = random.sample(words, 10)
random_words


embeddings = [model.wv[w] for w in random_words if w in model.wv]

df = pd.DataFrame(embeddings, index=random_words)
print(df)

# Plot heatmap
plt.figure(figsize=(16, 8))
sns.heatmap(df, cmap="coolwarm", annot=False, cbar=True)
plt.title("Word Embeddings Heatmap")
plt.xlabel("Embedding dimensions")
plt.ylabel("Words")
plt.show()



