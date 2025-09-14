# Data Source
folder = 'data'
num_doc = 10
# LSTM config
hidden_dim = 128
num_layers = 2
learning_rate = 0.05

# word2vec config
vector_size=100,    # embedding dimension
window=5,           # context window
min_count=2,        # ignore rare words
workers=4           # parallel training

# Predict Config
next_words = 40
