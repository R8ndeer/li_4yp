"""Data handling module for hair swatch datasets."""

from .augment import augment_masked_features
from .datasets import HairSwatchDataset, DigitalSwatchDataset
from .preprocessing import (
    load_data, 
    parse_shade, 
    split_shade, 
    preprocess_lab_data,
    preprocess_formula_data,
    merge_on_shade
)

__all__ = [
    "HairSwatchDataset",
    "DigitalSwatchDataset",
    "load_data",
    "parse_shade", 
    "split_shade",
    "preprocess_lab_data",
    "preprocess_formula_data",
    "merge_on_shade",
    "augment_masked_features"
]
