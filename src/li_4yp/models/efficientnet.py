import torch
import torch.nn as nn
from torchvision import models

class ShadeEfficientNet(nn.Module):
    def __init__(self, num_classes=[12, 11, 11]):
        super().__init__()
        self.dropout_rate = 0.3
        
        # 1. Load EfficientNet-B0 (Small, fast, effective)
        # We use 'DEFAULT' weights (ImageNet)
        weights = models.EfficientNet_B0_Weights.DEFAULT
        self.backbone = models.efficientnet_b0(weights=weights)
        
        # 2. Strategic Unfreezing
        # Freeze early layers (structural/edge detection)
        # Unfreeze the last 2-3 blocks (high-level feature combination)
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # EfficientNet features are in .features sequences. 
        # Unfreeze the last 30% of the network to allow "Color Adaptation"
        # The .features list has 9 blocks. We unfreeze from block 6 onwards.
        for param in self.backbone.features[6:].parameters():
            param.requires_grad = True
            
        # 3. Handle Feature Dimensions
        # EfficientNet-B0 outputs 1280 channels at the final layer
        in_features = 1280 
        
        # We replace the original classifier
        self.backbone.classifier = nn.Identity() 
        
        # 4. The Heads (No bottleneck projection!)
        # Direct connection: 1280 -> Head
        self.base_head = nn.Sequential(
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(in_features, num_classes[0])
        )
        
        self.primary_head = nn.Sequential(
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(in_features, num_classes[1])
        )
        
        self.secondary_head = nn.Sequential(
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(in_features, num_classes[2])
        )

    def forward(self, x):
        # Extract features (Batch, 1280, 7, 7) -> Pooling -> (Batch, 1280)
        # Note: EfficientNet's .forward() usually includes the classifier. 
        # We use .features() to get the maps, then pool manually.
        
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        
        return {
            'base': self.base_head(x),
            'primary': self.primary_head(x),
            'secondary': self.secondary_head(x)
        }
