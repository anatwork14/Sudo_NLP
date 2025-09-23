import torch
from torch.utils.data import Dataset, DataLoader

class TranslationDataset(Dataset):
    def __init__(self, src, tgt):
        self.src, self.tgt = src, tgt
    def __len__(self): return len(self.src)
    def __getitem__(self, i):
        return torch.tensor(self.src[i]), torch.tensor(self.tgt[i])