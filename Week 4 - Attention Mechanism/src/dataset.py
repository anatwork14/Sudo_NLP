from torch.utils.data import Dataset, DataLoader
import torch

class SummDataset(Dataset):
    def __init__(self, src_texts, target_texts, vocab, max_src=900, max_tgt=40):
        self.src = src_texts
        self.tgt = target_texts 
        self.vocab = vocab
        self.max_src = max_src # Max word in an article
        self.max_tgt = max_tgt # Max target words

    def encode(self, text, max_len):
        ids = [self.vocab.get(tok, self.vocab['<unk>']) for tok in text.split()]
        ids = ids[:max_len]
        return ids + [self.vocab['<pad>']]*(max_len - len(ids))

    def __getitem__(self, i):
        src_ids = self.encode(self.src[i], self.max_src)
        tgt_ids = [self.vocab['<sos>']] + \
                  self.encode(self.tgt[i], self.max_tgt-2) + \
                  [self.vocab['<eos>']]
        tgt_ids = tgt_ids + [self.vocab['<pad>']]*(self.max_tgt - len(tgt_ids))
        return torch.LongTensor(src_ids), torch.LongTensor(tgt_ids)

    def __len__(self): return len(self.src)