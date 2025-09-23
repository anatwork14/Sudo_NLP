from model import *
from underthesea import word_tokenize
from utils import * 
from train import *
class Predict:
    def __init__(self):
        # 1) Load or train the translation model
        trainer = TranslatorTrainer()
        self.translation_model = trainer.train_or_load()
        self.translation_model.eval()  # set to eval mode

        # 2) Device (CPU/GPU)
        self.device = next(self.translation_model.parameters()).device

        # 3) Utils and corpus
        self.utils = Utils()
        corpus, _, _ = self.utils.load_corpus()

        # 4) Vocabulary
        self.stoi, self.itos, _ = self.utils.process_embeddings(
            self.utils.word2vec_model(corpus)
        )
    def translate(self, sentence_text, lang = "en"):
        """
        Greedy decoding for a single sentence.
        """
        # 1) Tokenize
        if lang == "vi":
            tokens = word_tokenize(sentence_text.lower())
        else:
            tokens = sentence_text.lower().strip().split()

        # 2) Encode source
        src_ids = torch.tensor(
            [Utils.encode_sentence(tokens, self.stoi, max_len_src)],
            dtype=torch.long,
            device=self.device
        )

        # 3) Greedy decoding
        tgt_tokens = [self.stoi["<sos>"]]
        for _ in range(max_len_tgt):
            tgt_ids = torch.tensor([tgt_tokens], dtype=torch.long, device=self.device)
            with torch.no_grad():
                logits = self.translation_model(src_ids, tgt_ids)
            next_id = logits[0, -1].argmax(dim=-1).item()
            if next_id == self.stoi["<eos>"]:
                break
            tgt_tokens.append(next_id)

        return [self.itos[i] for i in tgt_tokens[1:]]  # drop <sos>