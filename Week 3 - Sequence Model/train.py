import torch 
from model import TextGenerateModel
from utils import Utils
from config import * 

class Trainer():
    def train():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = TextGenerateModel(Utils.vocab_size, Utils.embedding_dim, hidden_dim=hidden_dim, num_layers=num_layers, embedding_matrix=Utils.embedding_matrix).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

        for epoch in range(5):
            model.train()
            total_loss = 0
            for X, y in loader: # X = Sequence, y = Target (Next Word to be predict)
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()
                out, _ = model(X)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")
