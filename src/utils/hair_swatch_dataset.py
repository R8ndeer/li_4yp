import yaml
import pandas as pd
from typing import Tuple
from PIL import Image
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import transforms


CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "config.yaml"


class HairSwatchDataset(Dataset):
    def __init__(self, transform=None):
        with open(CONFIG_PATH, 'r') as f:
            self.__config = yaml.safe_load(f)

        # Dataset path
        self.__ds_path = Path(self.__config['paths']['data']).resolve() / self.__config['dataset']['name']
        # Load dataset
        df = pd.read_csv(self.__ds_path / self.__config['dataset']['label_file'])
        self.__img_paths = [self.__ds_path / fname for fname in df['filename']]
        self.__labels = df[['Base', 'Primary', 'Secondary', 'Tertiary']].to_numpy(dtype='int64')

        if transform is None:
            print("No transform provided, using default ToTensor() transform.")
            self.__transform = transforms.ToTensor()
        else:
            self.__transform = transform
    

    def __len__(self) -> int:
        return len(self.__img_paths)


    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = Image.open(self.__img_paths[idx]).convert('RGB')
        img = self.__transform(img)
        label = torch.from_numpy(self.__labels[idx])
        return img, label