import torch
import torch.nn as nn

class TranslationTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, embedding_matrix, stoi,nhead=6,
                 num_layers=4, ff_dim=512, dropout=0.1, max_len=500):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix, dtype=torch.float),
            freeze=False,
            padding_idx=stoi["<pad>"]
        )
        self.pos_enc = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        self.stoi = stoi
        self.transformer = nn.Transformer(
            d_model=embed_dim,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, src, tgt):
        """
        src: (batch, src_len)
        tgt: (batch, tgt_len)
        """
        device = src.device

        # --- Embeddings + positional encodings ---
        src_emb = self.embedding(src) + self.pos_enc[:, :src.size(1), :]
        tgt_emb = self.embedding(tgt) + self.pos_enc[:, :tgt.size(1), :]

        # --- Masks ---
        # 1) Look-ahead (causal) mask for the decoder
        tgt_len = tgt.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            tgt_len, device=device
        )

        # 2) Padding masks
        src_pad_mask = (src == self.stoi["<pad>"])
        tgt_pad_mask = (tgt == self.stoi["<pad>"])

        out = self.transformer(
            src_emb,
            tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_pad_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_pad_mask,
        )
        return self.fc_out(out)