import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import *     # make sure this is your model class
from dataset import *
from utils import Utils
from config import *

class SummarizeTrainer:
    def __init__(self):
        self.utils = Utils()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # --- data prep ---
        self.corpus, self.titles, self.articles = self.utils.load_corpus()
        self.stoi, self.itos, self.embedding_matrix = self.utils.process_embeddings(
            self.utils.word2vec_model(self.corpus)
        )
    def train_one_epoch(self, model, dataloader, optimizer, criterion, device, teacher_forcing_ratio=0.5):
        model.train()
        total_loss = 0

        for src, tgt in dataloader:           # src: [B,S], tgt: [B,T]
            src, tgt = src.to(device), tgt.to(device)

            optimizer.zero_grad()

            # model returns predictions for each target step except the first <sos>
            output = model(src, tgt, teacher_forcing_ratio)
            # output shape: [B, T-1, vocab]

            # Align target: skip first token (<sos>)
            target = tgt[:, 1:]                # [B, T-1]

            # Flatten for CrossEntropy: [(B*(T-1)), vocab]
            loss = criterion(
                output.reshape(-1, output.size(-1)),
                target.reshape(-1)
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # optional: gradient clipping
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)
    def train_or_load(self):
        """
        Load the trained model if it exists, else train and save it.
        Returns the TranslationTransformer instance.
        """
        # --- check if we already have a saved model ---
        if os.path.isfile(summarization_model_path):
            print(f"Found saved model at {summarization_model_path}. Loading...")
            encoder = ModelEncoder(vocab_size=len(self.stoi), embedding_size=120, hidden_size=256, embedding_weights=self.embedding_matrix, stoi=self.stoi)
            decoder = ModelDecoder(len(self.stoi), emb_size=120, hidden_size=256, stoi=self.stoi)
            model = SummarizationModel(encoder, decoder).to(self.device)
            model.load_state_dict(torch.load(summarization_model_path, map_location=self.device))
            model.eval()
            return model

        print("No saved model found. Starting training...")


        train_ds = SummDataset(self.articles, self.titles, self.stoi)
        train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)

        # --- model / loss / optimizer ---
        encoder = ModelEncoder(vocab_size=len(self.stoi), embedding_size=120, hidden_size=256, embedding_weights=self.embedding_matrix)
        decoder = ModelDecoder(len(self.stoi), emb_size=120, hidden_size=256)
        sum_model   = SummarizationModel(encoder, decoder).to(self.device)
        
        criterion = nn.CrossEntropyLoss(ignore_index=self.stoi["<pad>"])
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        # --- training loop ---
        NUM_EPOCHS = 15

        for epoch in range(1, NUM_EPOCHS + 1):
            train_loss = self.train_one_epoch(sum_model, train_loader,
                                        optimizer, criterion, self.device)
            # val_loss = evaluate(model, val_loader, criterion, device)  # if you have val set
            if (train_loss <= 0.2):
                break
            print(f"Epoch {epoch:02d}: train loss = {train_loss:.4f}")


        # --- save model ---
        os.makedirs(os.path.dirname(summarization_model_path), exist_ok=True)
        torch.save(model.state_dict(), summarization_model_path)
        print(f"Model saved to {summarization_model_path}")

        return model
