# train_word_rnn.py
import os
import unicodedata
from collections import Counter, defaultdict
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import random

# -------------------------
# 1. Gather files per author
# -------------------------
def gather_files(root_dir: str):
    author_files = defaultdict(list)
    for root, dirs, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith('.txt'):
                path = os.path.join(root, f)
                parts = f.rsplit('-', 1)
                if len(parts) == 2:
                    author = parts[1].rsplit('.',1)[0].strip()
                else:
                    author = 'unknown'
                author_files[author].append(path)
    return dict(author_files)

# -------------------------
# 2. Normalize & tokenize
# -------------------------
def normalize_text(s: str) -> str:
    s = unicodedata.normalize('NFC', s)
    return s

def tokenize(text: str):
    # Simple baseline (replace with underthesea for VN if you want)
    return text.strip().split()

# -------------------------
# 3. Build vocab
# -------------------------
def build_word_vocab(texts, min_freq=2, max_vocab=30000):
    counter = Counter()
    for t in texts:
        counter.update(tokenize(t))
    # keep frequent words
    tokens = [w for w, c in counter.most_common(max_vocab) if c >= min_freq]
    specials = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
    vocab = specials + tokens
    stoi = {w: i for i, w in enumerate(vocab)}
    itos = {i: w for i, w in enumerate(vocab)}
    return stoi, itos

# -------------------------
# 4. Dataset
# -------------------------
class WordDataset(Dataset):
    def __init__(self, author_files, stoi, seq_len=30):
        self.examples = []
        self.author_to_idx = {a: i for i, a in enumerate(sorted(author_files))}
        self.stoi = stoi
        self.seq_len = seq_len

        for author, files in author_files.items():
            for p in files:
                with open(p, 'r', encoding='utf-8', errors='ignore') as f:
                    text = normalize_text(f.read())
                    words = tokenize(text)
                    # sliding window
                    for i in range(0, len(words) - seq_len):
                        chunk = words[i:i+seq_len+1]
                        self.examples.append((self.author_to_idx[author], chunk))

    def __len__(self): return len(self.examples)

    def __getitem__(self, idx):
        author_id, chunk = self.examples[idx]
        ids = [self.stoi.get('<BOS>')]
        ids += [self.stoi.get(w, self.stoi['<UNK>']) for w in chunk]
        # pad if needed
        if len(ids) < self.seq_len+1:
            ids += [self.stoi['<PAD>']] * (self.seq_len+1 - len(ids))
        input_ids = torch.tensor(ids[:-1], dtype=torch.long)
        target_ids = torch.tensor(ids[1:], dtype=torch.long)
        return author_id, input_ids, target_ids

# -------------------------
# 5. Model
# -------------------------
class WordLSTM(nn.Module):
    def __init__(self, vocab_size, n_authors, emb_size=128, author_emb=32, hidden=512, layers=2, dropout=0.2):
        super().__init__()
        self.word_emb = nn.Embedding(vocab_size, emb_size)
        self.author_emb = nn.Embedding(n_authors, author_emb)
        self.lstm = nn.LSTM(emb_size + author_emb, hidden, layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden, vocab_size)

    def forward(self, input_ids, author_ids, hidden=None):
        B, T = input_ids.size()
        wemb = self.word_emb(input_ids)
        aemb = self.author_emb(author_ids).unsqueeze(1).expand(-1, T, -1)
        x = torch.cat([wemb, aemb], dim=-1)
        out, hidden = self.lstm(x, hidden)
        logits = self.fc(out)
        return logits, hidden

# -------------------------
# 6. Sampling
# -------------------------
def sample(model, stoi, itos, author_id, seed_words=None, length=50, temperature=1.0, top_k=None, device='cpu'):
    model.eval()
    seed_words = seed_words or []
    ids = [stoi['<BOS>']] + [stoi.get(w, stoi['<UNK>']) for w in seed_words]
    input_tensor = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    author_tensor = torch.tensor([author_id], dtype=torch.long, device=device)
    hidden = None
    generated = seed_words[:]

    with torch.no_grad():
        logits, hidden = model(input_tensor, author_tensor, hidden)
        last_logits = logits[0, -1]
        for _ in range(length):
            logits = last_logits / temperature
            probs = F.softmax(logits, dim=0)
            if top_k:
                top_vals, top_idx = torch.topk(probs, top_k)
                probs = torch.zeros_like(probs).scatter(0, top_idx, top_vals)
                probs /= probs.sum()
            next_id = torch.multinomial(probs, 1).item()
            word = itos.get(next_id, '<UNK>')
            generated.append(word)
            input_tensor = torch.tensor([[next_id]], device=device)
            logits, hidden = model(input_tensor, author_tensor, hidden)
            last_logits = logits[0, -1]

    return ' '.join(generated)

# -------------------------
# 7. Training loop
# -------------------------
def train_loop(root_dir, epochs=5, batch_size=64, seq_len=30, device='cpu'):
    author_files = gather_files(root_dir)
    texts = []
    for files in author_files.values():
        for p in files:
            with open(p, 'r', encoding='utf-8', errors='ignore') as f:
                texts.append(normalize_text(f.read()))
    stoi, itos = build_word_vocab(texts)
    dataset = WordDataset(author_files, stoi, seq_len=seq_len)
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = WordLSTM(len(stoi), len(dataset.author_to_idx)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss(ignore_index=stoi['<PAD>'])

    for e in range(epochs):
        model.train()
        total = 0
        for author_ids, x, y in dl:
            x, y = x.to(device), y.to(device)
            author_ids = torch.tensor(author_ids, device=device)
            opt.zero_grad()
            logits, _ = model(x, author_ids)
            loss = loss_fn(logits.view(-1, len(stoi)), y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
        print(f"Epoch {e+1}/{epochs}, Loss={total/len(dl):.4f}")
    return model, stoi, itos, dataset.author_to_idx
