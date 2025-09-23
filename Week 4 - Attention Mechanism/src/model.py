import torch.nn as nn
import torch 

class ModelEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, embedding_weights, stoi):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_size, padding_idx=stoi['<pad>'])
        self.embedding.weight.data.copy_(torch.tensor(embedding_weights))
        self.embedding.weight.requires_grad = False
        
        # LSTM
        self.rnn = nn.LSTM(input_size= embedding_size, hidden_size=hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, hidden_size) # A_foward + A_backward as input and then compress to one
        
    def forward(self, src):
        emb = self.embedding(src)
        outputs, (h_n, c_n) = self.rnn(emb)   # outputs: [B,S,2H], h_n/c_n: [2,B,H]

        # Ghép 2 hướng lại thành [B, 2H]
        h_cat = torch.cat((h_n[-2], h_n[-1]), dim=1)   # [B, 2H]
        c_cat = torch.cat((c_n[-2], c_n[-1]), dim=1)   # [B, 2H]

        # Chiếu xuống H và thêm chiều num_layers=1
        h_final = torch.tanh(self.fc(h_cat)).unsqueeze(0)  # [1,B,H]
        c_final = torch.tanh(self.fc(c_cat)).unsqueeze(0)  # [1,B,H]

        return self.fc(outputs), (h_final, c_final)
    
class ModelAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size*2, hidden_size) # Concat Encoder + Decoder
        self.v = nn.Linear(hidden_size, 1, bias=False)
        self.softmax = nn.Softmax()

    def forward(self, decoder_hidden, encoder_outputs):
        a = encoder_outputs                             # [B, S, H]
        h, _ = decoder_hidden               # take only the hidden state
        # h shape: [num_layers, B, H]  (here num_layers=1 for decoder)
        h = h.permute(1, 0, 2)              # -> [B, 1, H]
        B, S, H = encoder_outputs.size()
        s = h.repeat(1, S, 1)               # [B, S, H]

        energy = torch.relu(self.attn(torch.cat((s, a), dim=2)))  # [B, S, H]
        scores = self.v(energy).squeeze(2)              # [B, S]
        attn_weights = torch.softmax(scores, dim=1)     # [B, S]
        context = torch.bmm(attn_weights.unsqueeze(1), a)  # [B, 1, H]
        return context, attn_weights

class ModelDecoder(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, stoi):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=stoi['<pad>'])
        self.rnn = nn.LSTM(hidden_size + emb_size, hidden_size, batch_first=True)
        self.fc_out = nn.Linear(hidden_size*2 + emb_size, vocab_size)
        self.attention = ModelAttention(hidden_size)

    def forward(self, input_token, hidden, encoder_outputs):
        # input_token: [B] current word index
        emb = self.embedding(input_token).unsqueeze(1)     # [B,1,E]
        context, attn_weights = self.attention(hidden, encoder_outputs)
        rnn_input = torch.cat((emb, context), dim=2)       # [B,1,E+H]
        output, hidden = self.rnn(rnn_input, hidden)       # output: [B,1,H]
        concat = torch.cat((output, context, emb), dim=2)  # [B,1,H+H+E]
        prediction = self.fc_out(concat).squeeze(1)        # [B,vocab]
        return prediction, hidden, attn_weights

class SummarizationModel(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        encoder_outputs, hidden = self.encoder(src)
        input_token = tgt[:,0]        # usually <sos>
        outputs = []
        for t in range(1, tgt.size(1)):
            output, hidden, _ = self.decoder(input_token, hidden, encoder_outputs)
            outputs.append(output.unsqueeze(1))
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            input_token = tgt[:,t] if teacher_force else output.argmax(1)
        return torch.cat(outputs, dim=1)   # [B, T-1, vocab]
