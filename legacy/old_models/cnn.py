import torch
import torch.nn as nn
import torch.nn.functional as F

class ShadeCNN(nn.Module):
    def __init__(self, num_classes: list | tuple):
        """CNN model for multi-task hair color classification.

        Args:
            num_classes (list | tuple): Number of classes for each task
        """
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),   # (B, 3, 224, 224) -> (B, 16, 224, 224)
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),                             

            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # (B, 16, 112, 112) -> (B, 32, 112, 112)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),                             

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # (B, 32, 56, 56) -> (B, 64, 56, 56)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),                              # -> (B, 64, 28, 28) 
            nn.AdaptiveAvgPool2d((4, 4))                  # -> (B, 64, 4, 4)
        )
        self.flatten = nn.Flatten()
        in_features = 64 * 4 * 4
        self.shared_fc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.base_head = nn.Linear(128, num_classes[0])
        self.primary_head = nn.Linear(128, num_classes[1])
        self.secondary_head = nn.Linear(128, num_classes[2])

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.shared_fc(x)

        return {
            "base": self.base_head(x),
            "primary": self.primary_head(x),
            "secondary": self.secondary_head(x)
        }


class DeeperShadeCNN(nn.Module):
    def __init__(self, num_classes: list | tuple):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),   # (B, 3, 224, 224) -> (B, 16, 224, 224)
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # (B, 16, 112, 112) -> (B, 32, 112, 112)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # (B, 32, 56, 56) -> (B, 64, 56, 56)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # (B, 64, 28, 28) -> (B, 128, 28, 28)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),                              # -> (B, 128, 14, 14)
        )
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # -> (B, 128, 1, 1)
        self.shared_fc = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.base_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes[0])
        )
        self.primary_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes[1])
        )
        self.secondary_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes[2])
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.shared_fc(x)

        return {
            "base": self.base_head(x),
            "primary": self.primary_head(x),
            "secondary": self.secondary_head(x)
        }