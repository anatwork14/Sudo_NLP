import torch.nn as nn

class TextGenerateModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, embedding_matrix):
        super(TextGenerateModel, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight = nn.Parameter(embedding_matrix)  # load word2vec embedding_matrix
        self.embedding.weight.requires_grad = False  # Dont need to update

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x) # Convert Input Embeddings Matrix to nnEmbeddings for fast lookup
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out[:, -1, :])  # predict in last time step
        return out, hidden
