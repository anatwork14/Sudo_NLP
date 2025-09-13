from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class TextDataset(Dataset):
    def __init__(self, corpus, stoi, seq_len=50):
        self.data = []
        for doc in corpus:
            encoded = [stoi[w] for w in doc if w in stoi]
            for i in range(1, len(encoded)):
                seq = encoded[:i+1]
                if len(seq) > seq_len:
                    seq = seq[-seq_len:]
                self.data.append(seq)

    def __len__(self):
        return len(self.data)

    # When going with text generation --> return current sequence + target word (next word - have to predict)
    def __getitem__(self, idx):
        seq = torch.tensor(self.data[idx][:-1], dtype=torch.long)
        target = torch.tensor(self.data[idx][-1], dtype=torch.long)
        return seq, target

# Seperate [(sequence1, target1), (sequence2, target2), ...] --> [sequence1, sequence2, ...] [target1, target2, ...]
    def collate_fn(batch):
        sequences, targets = zip(*batch)
        sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
        targets = torch.stack(targets)
        return sequences, targets
