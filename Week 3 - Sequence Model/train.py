# import torch 
# from model import TextGenerateModel
# from utils import Utils
# from config import * 
# from loader import * 
# from torch.nn import nn
# from dataset import TextDataset
# from torch.utils.data import DataLoader

# class Trainer():
#     def train():
#         data = Loader().corpus
#         corpus = [Utils.tokenize(x) for x in data]
#         Utils.embedding_data(corpus)
        
#         dataset = TextDataset(corpus, Utils.stoi, seq_len=50)
#         loader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=TextDataset.collate_fn)
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#         model = TextGenerateModel(Utils.vocab_size, Utils.embedding_dim, hidden_dim=hidden_dim, num_layers=num_layers, embedding_matrix=Utils.embedding_matrix).to(device)
#         criterion = nn.CrossEntropyLoss()
#         optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

#         for epoch in range(5):
#             model.train()
#             total_loss = 0
#             for X, y in loader: # X = Sequence, y = Target (Next Word to be predict)
#                 X, y = X.to(device), y.to(device)
#                 optimizer.zero_grad()
#                 out, _ = model(X)
#                 loss = criterion(out, y)
#                 loss.backward()
#                 optimizer.step()
#                 total_loss += loss.item()
#             print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")

#         return model


import os
import torch 
from model import TextGenerateModel
from utils import Utils
from config import * 
from loader import * 
import torch.nn as nn
from dataset import TextDataset
from torch.utils.data import DataLoader

class Trainer:
    @staticmethod
    def train(model_path="textgen_model.pth"):
        data = Loader().corpus
        corpus = [Utils.tokenize(x) for x in data]
        utils = Utils()
        utils.embedding_data(corpus = corpus)
        
        dataset = TextDataset(corpus, utils.stoi, seq_len=50)
        loader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=TextDataset.collate_fn)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- Create model ---
        model = TextGenerateModel(utils.vocab_size, utils.embedding_dim, hidden_dim=hidden_dim, num_layers=num_layers, embedding_matrix=utils.embedding_matrix).to(device)

        # --- If saved model exists, just load it ---
        if os.path.exists(model_path):
            print(f"🔹 Found existing model at {model_path}, loading instead of training...")
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            return model, utils.itos, utils.stoi

        # --- Otherwise, train ---
        print("⚡ Training new model...")
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in range(5):
            model.train()
            total_loss = 0
            for X, y in loader:  # X = Sequence, y = Target (Next Word to be predict)
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()
                out, _ = model(X)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")

        # --- Save model after training ---
        torch.save(model.state_dict(), model_path)
        print(f"✅ Model saved to {model_path}")

        return model, utils.itos, utils.stoi
