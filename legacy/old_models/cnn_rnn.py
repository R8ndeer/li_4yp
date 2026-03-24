"""CNN backbone + RNN decoder model."""

import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
    def __init__(self, out_channels: int):
        super().__init__()
        self.features = nn.Sequential(  # In: (B, 3, 224, 224)
            nn.Conv2d(3, 16, kernel_size=3, padding=1),  # -> (B, 16, 224, 224)
            nn.BatchNorm2d(16),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> (B, 16, 112, 112)
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # -> (B, 32, 112, 112)
            nn.BatchNorm2d(32),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> (B, 32, 56, 56)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # -> (B, 64, 56, 56)
            nn.BatchNorm2d(64),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> (B, 64, 28, 28)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # -> (B, 128, 28, 28)
            nn.BatchNorm2d(128),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> (B, 128, 14, 14)
        )
        self.flatten = nn.Flatten()
        self.proj = nn.Sequential(
            # nn.Dropout(0.5),
            nn.Linear(128 * 14 * 14, 512),  # -> (B, 512)
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, out_channels),  # -> (B, out_channels)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.proj(x)
        return x  # (B, out_channels)


class RNNDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        output_dim: int,
        feat_dim: int,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.start_emb = nn.Parameter(torch.zeros(embed_dim))  # (embed_dim,)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        # Initialize hidden state with CNN features
        self.feat_to_hidden = nn.Linear(feat_dim, hidden_dim)

    def forward(self, tokens_in, x):
        emb = self.embedding(tokens_in)  # (B, seq_len, embed_dim)
        emb[:, 0, :] = self.start_emb  # inject start token embedding at t=0

        h0 = torch.tanh(self.feat_to_hidden(x)).unsqueeze(0)  # (1, B, hidden_dim)
        c0 = torch.zeros_like(h0)
        out, _ = self.lstm(emb, (h0, c0))
        logits = self.fc(out)  # (B, seq_len, output_dim)
        return logits


class CNNRNNModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        output_dim: int,
        feat_dim: int,
    ):
        super().__init__()
        self.encoder = CNNEncoder(out_channels=feat_dim)
        self.decoder = RNNDecoder(
            vocab_size, embed_dim, hidden_dim, output_dim, feat_dim
        )

    def forward(self, tokens_in, input_img):
        img_feat = self.encoder(input_img)  # (B, feat_dim)
        logits = self.decoder(
            tokens_in, img_feat
        )  # (B, seq_len, output_dim) == (B, 4, 10)
        return logits
