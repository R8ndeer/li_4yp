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
            nn.ReLU(),
        )

        # self.pixel_mlp = nn.Sequential(
        #     nn.Conv2d(3, 64, kernel_size=1),   # 1x1 Conv = Dense on pixels
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 128, kernel_size=1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(),
        #     nn.Conv2d(128, 128, kernel_size=1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU()
        # )

        # 2. The Heads
        # Input dim = 128 features * 2 stats (Mean + Std) = 256
        input_dim = 256

        self.base_head = nn.Sequential(
            nn.Linear(input_dim, 128),
            # nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes[0]),
        )

        self.primary_head = nn.Sequential(
            nn.Linear(input_dim, 128),
            # nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes[1]),
        )

        self.secondary_head = nn.Sequential(
            nn.Linear(input_dim, 128),
            # nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes[2]),
        )

    def forward(self, x):
        # x: [Batch, 3, 224, 224]

        # 1. Extract Color Features per pixel
        x = self.pixel_mlp(x)  # -> [Batch, 128, 224, 224]

        # 2. Statistical Pooling
        # Global Mean: Average color presence
        global_mean = torch.mean(x, dim=[2, 3])  # -> [Batch, 128]

        # Global Std: Color variance (captures shine/texture)
        global_std = torch.std(x, dim=[2, 3])  # -> [Batch, 128]

        # Concatenate: [Batch, 256]
        stats = torch.cat([global_mean, global_std], dim=1)

        # global_mean = torch.mean(x, dim=[2, 3])                       # [B, 128]
        # global_var  = torch.var(x, dim=[2, 3], correction=0)          # [B, 128]
        # global_std  = torch.sqrt(global_var + 1e-6)                   # [B, 128]
        # stats = torch.cat([global_mean, global_std], dim=1)           # [B, 256]

        return {
            "base": self.base_head(stats),
            "primary": self.primary_head(stats),
            "secondary": self.secondary_head(stats),
        }


# from typing import Sequence, Dict


# class Residual1x1Lite(nn.Module):
#     """
#     Lightweight 1x1 Conv Residual Block.
#     Adds pixel-wise depth with minimal extra cost.
#     """
#     def __init__(self, channels: int):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Conv2d(channels, channels, kernel_size=1, bias=False),
#             nn.BatchNorm2d(channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channels, channels, kernel_size=1, bias=False),
#             nn.BatchNorm2d(channels),
#         )
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         residual = x
#         out = self.net(x)
#         return self.relu(out + residual)


# class PixelStatNet(nn.Module):
#     """
#     PixelStatNet v3: slightly deeper than v1 with a cheap residual block.

#     - Shared 1x1 conv "MLP" over pixels
#     - One lightweight residual block at 32 channels
#     - Global mean + std pooling -> 3 parallel heads (Base / Primary / Secondary)
#     """

#     def __init__(
#         self,
#         num_classes: Sequence[int] = (12, 11, 11),
#         dropout_rate: float = 0.2,
#     ):
#         super().__init__()

#         # 1. Pixel-wise Feature Extraction
#         self.pixel_mlp = nn.Sequential(
#             # 3 -> 32
#             nn.Conv2d(3, 32, kernel_size=1, bias=False),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),

#             # Residual block at 32 channels (cheap, adds depth)
#             Residual1x1Lite(32),

#             # 32 -> 64
#             nn.Conv2d(32, 64, kernel_size=1, bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),

#             # 64 -> 128
#             nn.Conv2d(64, 128, kernel_size=1, bias=False),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#         )

#         # 2. Heads
#         # 128 features * 2 stats (mean + std) = 256
#         feat_dim = 128 * 2

#         self.base_head = self._make_head(feat_dim, num_classes[0], dropout_rate)
#         self.primary_head = self._make_head(feat_dim, num_classes[1], dropout_rate)
#         self.secondary_head = self._make_head(feat_dim, num_classes[2], dropout_rate)

#     @staticmethod
#     def _make_head(in_dim: int, out_dim: int, dropout: float) -> nn.Module:
#         return nn.Sequential(
#             nn.Linear(in_dim, 128),
#             nn.ReLU(inplace=True),
#             nn.Dropout(dropout),
#             nn.Linear(128, out_dim),
#         )

#     def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
#         """
#         Args:
#             x: Tensor of shape [B, 3, H, W]

#         Returns:
#             dict with keys 'base', 'primary', 'secondary',
#             each of shape [B, num_classes_k]
#         """
#         # Pixel-wise feature extraction
#         x = self.pixel_mlp(x)             # [B, 128, H, W]

#         # Global statistical pooling
#         mean = x.mean(dim=(2, 3))         # [B, 128]
#         std = x.std(dim=(2, 3))           # [B, 128]

#         stats = torch.cat([mean, std], dim=1)  # [B, 256]

#         return {
#             "base": self.base_head(stats),
#             "primary": self.primary_head(stats),
#             "secondary": self.secondary_head(stats),
#         }


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
            nn.Linear(128, num_classes[0]),
        )

        self.primary_head = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes[1]),
        )

        self.secondary_head = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes[2]),
        )

    def forward(self, x):
        # x: [Batch, 3, 224, 224]

        # 1. Extract Color Features per pixel
        x = self.pixel_mlp(x)  # -> [Batch, 128, 224, 224]
        x = self.pool(x)  # -> [Batch, 128, 56, 56]

        # 2. Statistical Pooling
        # Global Mean: Average color presence
        mean = torch.mean(x, dim=[2, 3])  # -> [Batch, 128]
        # Global Std: Color variance (captures shine/texture)
        std = torch.std(x, dim=[2, 3])  # -> [Batch, 128]
        # Global Quantiles
        quantiles = torch.quantile(
            x.view(x.size(0), x.size(1), -1),  # [Batch, 128, 224*224]
            q=torch.tensor([0.25, 0.5, 0.75], device=x.device),
            dim=2,
        )  # -> [3, Batch, 128]
        p25, p50, p75 = quantiles[0], quantiles[1], quantiles[2]

        # Concatenate: [Batch, 512]
        stats = torch.cat([mean, std, p25, p50, p75], dim=1)

        return {
            "base": self.base_head(stats),
            "primary": self.primary_head(stats),
            "secondary": self.secondary_head(stats),
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
            nn.ReLU(),
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
            nn.Linear(128, num_classes[0]),
        )

        self.primary_head = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes[1]),
        )

        self.secondary_head = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes[2]),
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
            "base": self.base_head(combined),
            "primary": self.primary_head(combined),
            "secondary": self.secondary_head(combined),
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

        # Block 1: Raw Pixel Analysis (224x224)
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)  # -> 112x112

        # Block 2: Patch Analysis (112x112)
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)  # -> 56x56

        # Block 3: Region Analysis (56x56)
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        # Final adaptive pool to get manageable size for stats
        # 14x14 = 196 patches. Enough for a robust std deviation.
        self.final_pool = nn.AdaptiveAvgPool2d((14, 14))

        # Heads
        # Input: 128 channels * 2 stats (Mean + Std) = 256
        combined_dim = 256

        self.base_head = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes[0]),
        )

        self.primary_head = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes[1]),
        )

        self.secondary_head = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes[2]),
        )

    def forward(self, x):
        # x: [B, 3, 224, 224]

        # Hierarchy
        x = self.block1(x)
        x = self.pool1(x)

        x = self.block2(x)
        x = self.pool2(x)

        x = self.block3(x)
        x = self.final_pool(x)  # -> [B, 128, 14, 14]

        # Global Statistical Pooling
        x_flat = x.flatten(2)  # -> [B, 128, 196]
        mean = torch.mean(x_flat, dim=2)
        std = torch.std(x_flat, dim=2)

        stats = torch.cat([mean, std], dim=1)  # -> [B, 256]

        return {
            "base": self.base_head(stats),
            "primary": self.primary_head(stats),
            "secondary": self.secondary_head(stats),
        }
