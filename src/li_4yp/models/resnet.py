import torch.nn as nn
from torchvision import models


class ShadeResNet(nn.Module):
    """
    Multi-Head ResNet for Hair Swatch Classification.
    Uses a standard ResNet backbone with three separate classification heads.
    """

    def __init__(
        self,
        num_classes=[12, 11, 11],
        dropout_rate=0.2,
        backbone="resnet18",
        pretrained=True,
    ):
        super().__init__()

        # 1. Load Backbone
        if backbone == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.resnet18(weights=weights)
            in_features = self.backbone.fc.in_features
        elif backbone == "resnet34":
            weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.resnet34(weights=weights)
            in_features = self.backbone.fc.in_features
        elif backbone == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.resnet50(weights=weights)
            in_features = self.backbone.fc.in_features
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Remove the original fully connected layer
        # We replace it with Identity so we can get the features
        self.backbone.fc = nn.Identity()

        # 2. Define Heads
        # Base Color Head
        self.base_head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes[0]),
        )

        # Primary Tone Head
        self.primary_head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes[1]),
        )

        # Secondary Tone Head
        self.secondary_head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes[2]),
        )

    def forward(self, x):
        # Extract features [Batch, in_features]
        features = self.backbone(x)

        return {
            "base": self.base_head(features),
            "primary": self.primary_head(features),
            "secondary": self.secondary_head(features),
        }
