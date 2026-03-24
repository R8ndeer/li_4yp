import torch.nn as nn
from torchvision import models


class MbNetV3Encoder(nn.Module):
    def __init__(self, feat_dim=256):
        super().__init__()
        mbnet = models.mobilenet_v3_large(
            weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2
        )
        for param in mbnet.parameters():
            param.requires_grad = False
        for param in mbnet.features[-2:].parameters():
            param.requires_grad = True

        self.features = mbnet.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        in_features = mbnet.classifier[0].in_features
        self.embed = nn.Linear(in_features, feat_dim)

    def forward(self, images):
        features = self.features(images)
        features = self.pool(features)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features


class MbNetShadeCNN(nn.Module):
    def __init__(self, feat_dim=256, num_classes=[12, 11, 11]):
        super().__init__()
        self.encoder = MbNetV3Encoder(feat_dim=feat_dim)

        self.base_head = nn.Linear(feat_dim, num_classes[0])
        self.primary_head = nn.Linear(feat_dim, num_classes[1])
        self.secondary_head = nn.Linear(feat_dim, num_classes[2])

    def forward(self, images):
        features = self.encoder(images)

        return {
            "base": self.base_head(features),
            "primary": self.primary_head(features),
            "secondary": self.secondary_head(features),
        }
