import torch
import torch.nn as nn

class PixelStatNet(nn.Module):
    """
    A 'Deep Random Forest' style network (Geometric Deep Learning).
    Uses 1x1 convolutions to learn a color space per-pixel (permutation invariant), 
    then uses statistical pooling (Mean + Std) to capture global texture/variation.
    """
    def __init__(self, num_classes=[12, 11, 11], dropout_rate=0.2):
        super().__init__()
        
        # 1. Pixel-wise Feature Extraction (The "Color Space" Learner)
        # 1x1 Convs acting as a Shared MLP on every pixel.
        self.pixel_mlp = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=1), 
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        # 2. The Heads
        # Input dim = 128 features * 2 stats (Mean + Std) = 256
        input_dim = 256
        
        self.base_head = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes[0])
        )
        
        self.primary_head = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes[1])
        )
        
        self.secondary_head = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes[2])
        )

    def forward(self, x):
        # x: [Batch, 3, 224, 224]
        
        # 1. Extract Color Features per pixel
        x = self.pixel_mlp(x) # -> [Batch, 128, 224, 224]
        
        # 2. Statistical Pooling 
        # Global Mean: Average color presence
        global_mean = torch.mean(x, dim=[2, 3]) # -> [Batch, 128]
        
        # Global Std: Color variance (captures shine/texture)
        global_std = torch.std(x, dim=[2, 3])   # -> [Batch, 128]
        
        # Concatenate: [Batch, 256]
        stats = torch.cat([global_mean, global_std], dim=1) 
        
        return {
            'base': self.base_head(stats),
            'primary': self.primary_head(stats),
            'secondary': self.secondary_head(stats)
        }


class PixelMoreStatNet(PixelStatNet):
    def __init__(self, num_classes=[12, 11, 11], dropout_rate=0.2):
        super().__init__(num_classes, dropout_rate)

        self.pool = nn.AdaptiveAvgPool2d((56, 56))  # Downsample for efficiency
                                                    # size must divide 224 for mps compatibility

        input_dim = 128 * 5  # Mean + Std + 3 Quantiles

        self.base_head = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes[0])
        )
        
        self.primary_head = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes[1])
        )
        
        self.secondary_head = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes[2])
        )

    def forward(self, x):
        # x: [Batch, 3, 224, 224]
        
        # 1. Extract Color Features per pixel
        x = self.pixel_mlp(x) # -> [Batch, 128, 224, 224]
        x = self.pool(x)     # -> [Batch, 128, 56, 56]
        
        # 2. Statistical Pooling 
        # Global Mean: Average color presence
        mean = torch.mean(x, dim=[2, 3]) # -> [Batch, 128]
        # Global Std: Color variance (captures shine/texture)
        std = torch.std(x, dim=[2, 3])   # -> [Batch, 128]
        # Global Quantiles
        quantiles = torch.quantile(
            x.view(x.size(0), x.size(1), -1),  # [Batch, 128, 224*224]
            q=torch.tensor([0.25, 0.5, 0.75], device=x.device), 
            dim=2
        )  # -> [3, Batch, 128]
        p25, p50, p75 = quantiles[0], quantiles[1], quantiles[2]
        
        # Concatenate: [Batch, 512]
        stats = torch.cat([mean, std, p25, p50, p75], dim=1) 
        
        return {
            'base': self.base_head(stats),
            'primary': self.primary_head(stats),
            'secondary': self.secondary_head(stats)
        }
    

class HybridPixelStatNet(nn.Module):
    """
    Hybrid Model: Combines PixelStatNet (Image) with Classical Features (CSV).
    
    Image Branch: 1x1 Conv -> Downsample -> Mean + Std (Texture/Color Variance)
    CSV Branch: Raw dense features (Mean LAB, Chroma, etc.)
    """
    def __init__(self, num_classes=[12, 11, 11], dropout_rate=0.2, csv_feature_dim=16):
        super().__init__()
        
        # 1. Image Branch (PixelStatNet - Reduced Capacity)
        # Reduced channels (32->64) because CSV features do the heavy lifting for color.
        self.pixel_mlp = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=1), 
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # self.downsampler = nn.AdaptiveAvgPool2d((28, 28))
        
        # 2. Input Dimension Calculation
        # Image Stats: 64 channels * 2 stats (Mean, Std) = 128 features
        # CSV Stats: e.g., 4 features (L, a, b, C)
        combined_dim = 128 + csv_feature_dim 
        
        # 3. Heads
        self.base_head = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes[0])
        )
        
        self.primary_head = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes[1])
        )
        
        self.secondary_head = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes[2])
        )

    def forward(self, x, csv_features):
        # x: [Batch, 3, 224, 224]
        # csv_features: [Batch, csv_feature_dim]
        
        # --- Image Branch ---
        x = self.pixel_mlp(x) 
        # x = self.downsampler(x) # Downsample to 28x28
        
        # Flatten spatial dims
        x_flat = x.flatten(2) 
        
        # Calculate Only the Winning Stats
        mean = torch.mean(x_flat, dim=2)
        std = torch.std(x_flat, dim=2)
        
        # [Batch, 128]
        image_stats = torch.cat([mean, std], dim=1)
        
        # --- Hybrid Fusion ---
        # Concatenate Image Stats with CSV Features
        # [Batch, 128 + csv_dim]
        combined = torch.cat([image_stats, csv_features], dim=1)
        
        return {
            'base': self.base_head(combined),
            'primary': self.primary_head(combined),
            'secondary': self.secondary_head(combined)
        }


class PatchStatNet(nn.Module):
    """
    Lightweight PatchStatNet.
    
    Architectural Parity with PixelStatNet:
    - Same depth (3 Convolutional Layers)
    - Same feature progression (32 -> 64 -> 128)
    
    Difference:
    - PixelStatNet: All Convs happen at 224x224, then Pool.
    - This Model: Conv -> Pool -> Conv -> Pool. 
      It forces the later layers to see "Patches" instead of "Pixels".
    """
    def __init__(self, num_classes=[12, 11, 11], dropout_rate=0.2):
        super().__init__()
        
        # Layer 1: Pixel Analysis (Input: 224x224)
        # Learns immediate color mappings (RGB -> Latent)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        # Downsample 1 (Mixes 2x2 pixels)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2) # -> 112x112
        
        # Layer 2: Patch Analysis (Input: 112x112)
        # Learns relationships between neighboring pixel colors
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        # Downsample 2
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2) # -> 56x56
        
        # Layer 3: Region Analysis (Input: 56x56)
        # Learns broader texture trends
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        # Final Downsample to fixed size for stats
        # We ensure we have enough spatial samples (14x14=196) for stable Std Dev
        self.final_pool = nn.AdaptiveAvgPool2d((14, 14))
        
        # Heads (Same as PixelStatNet)
        combined_dim = 256 # 128 channels * 2 stats (Mean + Std)
        
        self.base_head = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes[0])
        )
        
        self.primary_head = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes[1])
        )
        
        self.secondary_head = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes[2])
        )

    def forward(self, x):
        # x: [B, 3, 224, 224]
        
        # Step 1
        x = self.conv1(x) # [B, 32, 224, 224]
        x = self.pool1(x) # [B, 32, 112, 112]
        
        # Step 2
        x = self.conv2(x) # [B, 64, 112, 112]
        x = self.pool2(x) # [B, 64, 56, 56]
        
        # Step 3
        x = self.conv3(x) # [B, 128, 56, 56]
        x = self.final_pool(x) # [B, 128, 14, 14]
        
        # Global Statistics
        x_flat = x.flatten(2) # [B, 128, 196]
        mean = torch.mean(x_flat, dim=2)
        std = torch.std(x_flat, dim=2)
        
        stats = torch.cat([mean, std], dim=1) # [B, 256]
        
        return {
            'base': self.base_head(stats),
            'primary': self.primary_head(stats),
            'secondary': self.secondary_head(stats)
        }