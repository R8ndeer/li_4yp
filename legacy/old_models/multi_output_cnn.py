"""Multi-Output CNN model."""

import torch
import torch.nn as nn


class MultiOutputCNN(nn.Module):
    def __init__(self, vocab_size: int):
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
        )
        self.output_base = nn.Linear(512, vocab_size)  # -> (B, vocab_size)
        self.output_prim = nn.Linear(512, vocab_size)  # -> (B, vocab_size)
        self.output_sec = nn.Linear(512, vocab_size)  # -> (B, vocab_size)
        self.output_tert = nn.Linear(512, vocab_size)  # -> (B, vocab_size)

    def forward(self, input_img: torch.Tensor):
        x = self.features(input_img)
        x = self.flatten(x)
        x = self.proj(x)
        out_base = self.output_base(x)
        out_prim = self.output_prim(x)
        out_sec = self.output_sec(x)
        out_tert = self.output_tert(x)
        logits = torch.stack(
            [out_base, out_prim, out_sec, out_tert], dim=1
        )  # (B, 4, vocab_size)
        return logits  # (B, 4, vocab_size)
