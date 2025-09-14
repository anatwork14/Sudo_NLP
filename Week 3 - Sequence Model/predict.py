import torch.nn.functional as F
import torch 

class Predictor:
    def generate_text(model, seed_text, stoi, itos, next_words=20, seq_len=50, temperature=1.0):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        words = seed_text.split()
        for _ in range(next_words):
            encoded = [stoi.get(w, 0) for w in words]
            encoded = torch.tensor(encoded[-seq_len:], dtype=torch.long).unsqueeze(0).to(device)

            with torch.no_grad():
                out, _ = model(encoded)
                probs = F.softmax(out / temperature, dim=-1).squeeze()
                word_id = torch.multinomial(probs, 1).item()

            words.append(itos[word_id])
        return " ".join(words)