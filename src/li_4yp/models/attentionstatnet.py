import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentivePixelStatNet(nn.Module):
    """
    Improved Deep Sets implementation for Hair Swatches.
    
    Hypothesis: Pattern doesn't matter, but 'Quality' of pixels does.
    Difference from v1: Uses Attention to ignore highlights/shadows before averaging.
    
    Structure:
    1. Pixel MLP (phi): Projects RGB -> Latent Color Space
    2. Attention (w): Scores each pixel's reliability
    3. Aggregation (rho): Weighted Mean + Weighted Std
    """
    def __init__(self, num_classes=[12, 11, 11], dropout_rate=0.2):
        super().__init__()
        
        # 1. The "Phi" Network (Pixel-wise Feature Extractor)
        # Deep narrow MLP to learn a better color space than RGB
        self.pixel_mlp = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=1),   # 1x1 Conv = Dense on pixels
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        # 2. The Attention Mechanism
        # Decides "How much should this pixel count?"
        # e.g., Highlights/Shadows gets low weights.
        self.attention_net = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1),
            nn.Tanh(),                     # Tanh is often stable for attention
            nn.Conv2d(64, 1, kernel_size=1) # Output: 1 score per pixel
        )
        
        # 3. The "Rho" Network (Classifier Heads)
        # Input: 128 (Mean) + 128 (Std) = 256 features
        self.base_head = self._make_head(256, num_classes[0], dropout_rate)
        self.primary_head = self._make_head(256, num_classes[1], dropout_rate)
        self.secondary_head = self._make_head(256, num_classes[2], dropout_rate)

    def _make_head(self, in_dim, out_dim, dropout):
        return nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, out_dim)
        )

    def forward(self, x):
        # x: [Batch, 3, H, W]
        
        # --- 1. Feature Extraction (Permutation Invariant) ---
        feats = self.pixel_mlp(x) # [B, 128, H, W]
        
        # --- 2. Attention Masking ---
        # Calculate attention scores
        attn_logits = self.attention_net(feats) # [B, 1, H, W]
        
        # Flatten spatial dimensions to treat as a set
        b, c, h, w = feats.shape
        feats_flat = feats.view(b, c, -1)       # [B, 128, N_pixels]
        attn_flat = attn_logits.view(b, 1, -1)  # [B, 1, N_pixels]
        
        # Softmax ensures weights sum to 1 over the set of pixels
        attn_weights = F.softmax(attn_flat, dim=2) 
        
        # --- 3. Weighted Statistical Aggregation ---
        # Global Mean = Sum(Weight_i * Feat_i)
        global_mean = torch.sum(feats_flat * attn_weights, dim=2) # [B, 128]
        
        # Global Variance = Sum(Weight_i * (Feat_i - Mean)^2)
        # We use the weighted mean we just calculated
        variance_term = (feats_flat - global_mean.unsqueeze(2)) ** 2
        global_var = torch.sum(variance_term * attn_weights, dim=2)
        global_std = torch.sqrt(global_var + 1e-6) # [B, 128]
        
        # Combine
        stats = torch.cat([global_mean, global_std], dim=1) # [B, 256]
        
        return {
            'base': self.base_head(stats),
            'primary': self.primary_head(stats),
            'secondary': self.secondary_head(stats)
        }


class AttentiveStatNetOneMoment(nn.Module):
    """
    Simplified Attentive PixelStatNet using only Weighted Mean.
    """
    def __init__(self, num_classes=[12, 11, 11], dropout_rate=0.2):
        super().__init__()
        
        # 1. The "Phi" Network (Pixel-wise Feature Extractor)
        # Deep narrow MLP to learn a better color space than RGB
        self.pixel_mlp = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=1),   # 1x1 Conv = Dense on pixels
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        # 2. The Attention Mechanism
        # Decides "How much should this pixel count?"
        # e.g., Highlights/Shadows gets low weights.
        self.attention_net = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1),
            nn.Tanh(),                     # Tanh is often stable for attention
            nn.Conv2d(64, 1, kernel_size=1) # Output: 1 score per pixel
        )
        
        # 3. The "Rho" Network (Classifier Heads)
        # Input: 128 (Mean)
        self.base_head = self._make_head(128, num_classes[0], dropout_rate)
        self.primary_head = self._make_head(128, num_classes[1], dropout_rate)
        self.secondary_head = self._make_head(128, num_classes[2], dropout_rate)

    def _make_head(self, in_dim, out_dim, dropout):
        return nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, out_dim)
        )

    def forward(self, x):
        # x: [Batch, 3, H, W]
        
        # --- 1. Feature Extraction (Permutation Invariant) ---
        feats = self.pixel_mlp(x) # [B, 128, H, W]
        
        # --- 2. Attention Masking ---
        # Calculate attention scores
        attn_logits = self.attention_net(feats) # [B, 1, H, W]
        
        # Flatten spatial dimensions to treat as a set
        b, c, h, w = feats.shape
        feats_flat = feats.view(b, c, -1)       # [B, 128, N_pixels]
        attn_flat = attn_logits.view(b, 1, -1)  # [B, 1, N_pixels]
        
        # Softmax ensures weights sum to 1 over the set of pixels
        attn_weights = F.softmax(attn_flat, dim=2) 
        
        # --- 3. Weighted Statistical Aggregation ---
        # Global Mean = Sum(Weight_i * Feat_i)
        global_mean = torch.sum(feats_flat * attn_weights, dim=2) # [B, 128]
        
        return {
            'base': self.base_head(global_mean),
            'primary': self.primary_head(global_mean),
            'secondary': self.secondary_head(global_mean)
        }