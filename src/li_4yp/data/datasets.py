import pandas as pd
from typing import Tuple, Optional, Any
from PIL import Image
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class HairSwatchDataset(Dataset):
    def __init__(
        self, 
        data_dir: str | Path,
        csv_file: str | Path = "hair_swatch_labels.csv",
        transform: Optional[transforms.Compose] = None,
    ):
        """Initialize the dataset.
        
        Args:
            data_dir: Directory containing images and CSV file
            csv_file: Name of CSV file with labels (default: "hair_swatch_labels.csv")
            transform: Image transforms to apply
        """
        self.data_dir = Path(data_dir)
        
        # Load labels
        csv_path = self.data_dir / csv_file
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
            
        self.df = pd.read_csv(csv_path)
        
        # Validate required columns
        required_cols = ['filename', 'Base', 'Primary', 'Secondary', 'Tertiary']
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in CSV: {missing_cols}")
        
        # Store image paths and labels
        self.image_paths = [self.data_dir / fname for fname in self.df['filename']]
        
        # Convert labels to tensor
        labels_array = self.df[['Base', 'Primary', 'Secondary', 'Tertiary']].values
        
        # Remap Base values from 1... to 0... for PyTorch compatibility
        # Keep -1 as -1 (missing values)
        base_mask = labels_array[:, 0] != -1  # Find non-missing base values
        if base_mask.any():
            labels_array[base_mask, 0] = labels_array[base_mask, 0] - 1  # 1-12 -> 0-11
        
        self.labels = torch.tensor(labels_array, dtype=torch.long)
        
        # Set up transforms
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
            
        # Validate a few images exist
        missing_images = [path for path in self.image_paths[:5] if not path.exists()]
        if missing_images:
            print(f"Warning: Some images missing. First: {missing_images[0]}")
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[Any, torch.Tensor]:
        """Get image and label by index."""
        img_path = self.image_paths[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"Failed to load image {img_path}: {e}")
        
        # Apply transforms (assumed to include ToTensor)
        transformed_image = self.transform(image)
        
        # Get label
        label = self.labels[idx]
        
        return transformed_image, label
    
    @classmethod
    def from_config(cls, config_dict: dict, **kwargs):
        """Create dataset from config dictionary."""
        data_dir = config_dict.get('data_dir')
        csv_file = config_dict.get('csv_file')
        
        if not data_dir:
            raise ValueError("config_dict must contain 'data_dir'")
        if not csv_file:
            raise ValueError("config_dict must contain 'csv_file'")

        return cls(data_dir=data_dir, csv_file=csv_file, **kwargs)
    
    def get_info(self) -> dict:
        """Get dataset information as a dictionary."""
        return {
            'num_samples': len(self),
            'data_dir': str(self.data_dir),
            'csv_file': self.df.shape,
            'label_columns': ['Base', 'Primary', 'Secondary', 'Tertiary'],
            'sample_filename': self.df['filename'].iloc[0] if not self.df.empty else None
        }


class DigitalSwatchDataset(Dataset):
    def __init__(
        self, 
        data_dir: str | Path,
        csv_file: str | Path,
        transform: Optional[transforms.Compose] = None,
    ):
        """Initialize the dataset.
        
        Args:
            data_dir: Directory containing images and CSV file
            csv_file: Name of CSV file with labels (default: "hair_swatch_labels.csv")
            transform: Image transforms to apply
        """
        self.data_dir = Path(data_dir)
        
        # Load labels
        csv_path = self.data_dir / csv_file
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
            
        self.df = pd.read_csv(csv_path)
        
        # Validate required columns
        required_cols = ['filename', 'Base', 'Primary', 'Secondary']
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in CSV: {missing_cols}")
        
        # Store image paths and labels
        self.image_paths = [self.data_dir / fname for fname in self.df['filename']]
        
        # Convert labels to tensor
        labels_array = self.df[['Base', 'Primary', 'Secondary']].values

        # Remap Base values from 1... to 0... for PyTorch compatibility
        base_mask = labels_array[:, 0] == -1  # Find missing base values
        if base_mask.any():
            raise ValueError("Base labels cannot be -1 in DigitalSwatchDataset.")
        labels_array[:, 0] = labels_array[:, 0] - 1  # 1-12 -> 0-11

        # Replace -1 in Primary and Secondary with a valid class index (10)
        labels_array[labels_array == -1] = 10  # Assuming 10 is the index for 'no color'

        self.labels = torch.tensor(labels_array, dtype=torch.long)
        
        # Set up transforms
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
            
        # Validate a few images exist
        missing_images = [path for path in self.image_paths[:5] if not path.exists()]
        if missing_images:
            print(f"Warning: Some images missing. First: {missing_images[0]}")
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[Any, torch.Tensor]:
        """Get image and label by index."""
        img_path = self.image_paths[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"Failed to load image {img_path}: {e}")
        
        # Apply transforms (assumed to include ToTensor)
        transformed_image = self.transform(image)
        
        # Get label
        label = self.labels[idx]
        
        return transformed_image, label
    
    @classmethod
    def from_config(cls, config_dict: dict, **kwargs):
        """Create dataset from config dictionary."""
        data_dir = config_dict.get('data_dir')
        csv_file = config_dict.get('csv_file')
        
        if not data_dir:
            raise ValueError("config_dict must contain 'data_dir'")
        if not csv_file:
            raise ValueError("config_dict must contain 'csv_file'")

        return cls(data_dir=data_dir, csv_file=csv_file, **kwargs)
    
    def get_info(self) -> dict:
        """Get dataset information as a dictionary."""
        return {
            'num_samples': len(self),
            'data_dir': str(self.data_dir.resolve()),
            'label_columns': ['Base', 'Primary', 'Secondary'],
            'sample_filename': self.df['filename'].iloc[0] if not self.df.empty else None
        }


class HybridSwatchDataset(DigitalSwatchDataset):
    def __init__(
        self, 
        data_dir: str | Path,
        csv_file: str | Path,
        transform: Optional[transforms.Compose] = None,
        feature_cols: Optional[list] = None,
    ):
        """Initialize the hybrid dataset (Digital Swatch + LAB features).
        
        Args:
            data_dir: Directory containing images and CSV file
            csv_file: Name of CSV file with labels
            transform: Image transforms to apply
        """
        super().__init__(data_dir, csv_file, transform)

        if feature_cols is None:
            raise ValueError("feature_cols must be provided for HybridSwatchDataset.")
        self.feature_cols = feature_cols

        missing = [c for c in self.feature_cols if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing feature columns in CSV: {missing}")
        
        self.dense_features = torch.tensor(
            self.df[self.feature_cols].values, dtype=torch.float32
        )
    
    def __getitem__(self, idx: int):
        """Get image, dense features, and label by index."""
        image, label = super().__getitem__(idx)

        features = self.dense_features[idx]

        return image, features, label
    
    def get_info(self) -> dict:
        """Get dataset information as a dictionary."""
        info = super().get_info()
        info['feature_columns'] = self.feature_cols
        return info