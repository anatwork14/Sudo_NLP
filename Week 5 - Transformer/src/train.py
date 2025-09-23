import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import TranslationTransformer     # make sure this is your model class
from dataset import TranslationDataset
from utils import Utils
from config import (                         # expects these variables in config.py
    embedding_size,
    max_len_src,
    max_len_tgt,
    translation_model_path        # e.g. "model/translation_model.pt"
)

class TranslatorTrainer:
    def __init__(self):
        self.utils = Utils()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # --- data prep ---
        self.corpus, self.tokenized_english, self.tokenized_vietnamese = self.utils.load_corpus()
        self.stoi, itos, self.embedding_matrix = self.utils.process_embeddings(
            self.utils.word2vec_model(self.corpus)
        )

    def train_or_load(self):
        """
        Load the trained model if it exists, else train and save it.
        Returns the TranslationTransformer instance.
        """
        # --- check if we already have a saved model ---
        if os.path.isfile(translation_model_path):
            print(f"Found saved model at {translation_model_path}. Loading...")
            model = TranslationTransformer(len(self.stoi), embedding_size, self.embedding_matrix, self.stoi).to(self.device)
            model.load_state_dict(torch.load(translation_model_path, map_location=self.device))
            model.eval()
            return model

        print("No saved model found. Starting training...")

        src_ids = [self.utils.encode_sentence(s, self.stoi, max_len_src)
                   for s in self.tokenized_english]
        tgt_ids = [self.utils.encode_sentence(s, self.stoi, max_len_tgt)
                   for s in self.tokenized_vietnamese]

        train_ds = TranslationDataset(src_ids, tgt_ids)
        train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)

        # --- model / loss / optimizer ---
        model = TranslationTransformer(len(self.stoi), embedding_size).to(self.device)
        criterion = nn.CrossEntropyLoss(ignore_index=self.stoi["<pad>"])
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        # --- training loop ---
        for epoch in range(20):
            model.train()
            running_loss = 0.0
            for src, tgt in train_dl:
                src, tgt = src.to(self.device), tgt.to(self.device)
                optimizer.zero_grad()

                output = model(src, tgt[:, :-1])
                loss = criterion(
                    output.reshape(-1, len(self.stoi)),
                    tgt[:, 1:].reshape(-1)
                )
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            avg_loss = running_loss / len(train_dl)
            print(f"Epoch {epoch+1}, loss={avg_loss:.4f}")

        # --- save model ---
        os.makedirs(os.path.dirname(translation_model_path), exist_ok=True)
        torch.save(model.state_dict(), translation_model_path)
        print(f"Model saved to {translation_model_path}")

        return model
