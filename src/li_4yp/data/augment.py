import numpy as np
import cv2
from pathlib import Path
from typing import Sequence, Tuple
from tqdm import tqdm

from ..utils import mask_specular_and_shadow

def get_masked_feature(bgr_img, drop_top, drop_bottom):
    lab_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
    mask = mask_specular_and_shadow(lab_img, drop_top, drop_bottom)
    return [
        lab_img[:, :, 0][mask].mean(), # L channel
        lab_img[:, :, 1][mask].mean(), # a channel
        lab_img[:, :, 2][mask].mean(), # b channel
    ]


def augment_masked_features(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        thresholds: Sequence[Tuple[float, float]]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Get the masked feature (as mean Lab triplets) by filtering out specular highlights and shadows.

    Args:
        X_train (np.ndarray): Training image paths (N, ) as strings.
        y_train (np.ndarray): Training labels (N, D) as class indices.
        X_test (np.ndarray): Testing image paths (M, ) as strings.
        y_test (np.ndarray): Testing labels (M, D) as class indices.
        thresholds (Sequence[tuple]): List of (lower, upper) thresholds for masking.

    Returns:
        Augmented training and testing features and labels. Order: X_tr, y_tr, X_te, y_te
    """
    if not isinstance(X_train[0], str):
        raise ValueError("X_train should be an array of image file paths as strings.")
    if not isinstance(X_test[0], str):
        raise ValueError("X_test should be an array of image file paths as strings.")

    print(f"Augmenting training data from {len(X_train)} to {len(X_train) * len(thresholds)} samples...")
    X_tr, y_tr = [], []
    for fp, label in tqdm(zip(X_train, y_train), total=len(X_train)):
        if not Path(fp).is_file():
            raise FileNotFoundError(f"Image file not found: {fp}")
        X_tr.extend(
            get_masked_feature(
                cv2.imread(fp), drop_top=dt, drop_bottom=db
            ) for dt, db in thresholds
        )
        y_tr.extend([label] * len(thresholds))

    print(f"Preparing {len(X_test)} test samples...")
    X_te = []
    for fp in tqdm(X_test, total=len(X_test)):
        if not Path(fp).is_file():
            raise FileNotFoundError(f"Image file not found: {fp}")
        lab_img = cv2.cvtColor(cv2.imread(fp), cv2.COLOR_BGR2LAB)
        X_te.append(
            [
                lab_img[:, :, 0].mean(), # L channel
                lab_img[:, :, 1].mean(), # a channel
                lab_img[:, :, 2].mean(), # b channel
            ]
        )


    return np.array(X_tr), np.array(y_tr), np.array(X_te), y_test