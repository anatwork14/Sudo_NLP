from model import *
from underthesea import word_tokenize
from utils import * 
from train import *

class Predict:
    def __init__(self):
        # 1) Load or train the translation model
        trainer = SummarizeTrainer()
        self.summarization_model = trainer.train_or_load()
        self.summarization_model.eval()  # set to eval mode

        # 2) Device (CPU/GPU)
        self.device = next(self.summarization_model.parameters()).device

        # 3) Utils and corpus
        self.utils = Utils()
        corpus, _, _ = self.utils.load_corpus()

        # 4) Vocabulary
        self.stoi, self.itos, _ = self.utils.process_embeddings(
            self.utils.word2vec_model(corpus)
        )
    def encode_text(self, text, vocab, max_len=800):
        ids = [vocab.get(tok, vocab['<unk>']) for tok in text.split()]
        ids = ids[:max_len]
        return torch.LongTensor([ids + [vocab['<pad>']] * (max_len - len(ids))])
    def evaluate(model, dataloader, criterion, device):
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for src, tgt in dataloader:
                src, tgt = src.to(device), tgt.to(device)
                output = model(src, tgt)  # no teacher forcing
                target = tgt[:, 1:]
                loss = criterion(output.reshape(-1, output.size(-1)),
                                target.reshape(-1))
                total_loss += loss.item()
        return total_loss / len(dataloader)
    
    def generate_title_lstm(self, article_text, max_len=40):
        self.summarization_model.eval()
        with torch.no_grad():
            # encode_text already returns [1, max_len]
            src = self.encode_text(article_text, self.stoi).to(self.device)

            encoder_outputs, hidden = self.summarization_model.encoder(src)

            input_token = torch.tensor([self.stoi['<sos>']], device=self.device)
            generated = []

            for _ in range(max_len):
                output, hidden, _ = self.summarization_model.decoder(input_token, hidden, encoder_outputs)
                next_token = output.argmax(1)
                if next_token.item() == self.stoi['<eos>']:
                    break
                generated.append(next_token.item())
                input_token = next_token

        words = [self.itos[i] for i in generated]
        return " ".join(words)

