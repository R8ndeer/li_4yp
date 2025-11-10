"""Utilities for image I/O and processing using OpenCV. Only use OpenCV images."""

import cv2
import numpy as np

def imread_rgb(path: str) -> np.ndarray:
    imb_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if imb_bgr is None:
        raise FileNotFoundError(f"Image not found at path: {path}")
    return cv2.cvtColor(imb_bgr, cv2.COLOR_BGR2RGB)


def rgb2lab(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img.copy(), cv2.COLOR_RGB2Lab)


def cv2lab_to_lab(img: np.ndarray) -> np.ndarray:
    """Convert OpenCV Lab image to standard Lab image with L in [0, 100], a and b in [-128, 127]."""
    lab_img = img.copy().astype(np.float64)
    lab_img[:, :, 0] = lab_img[:, :, 0] * (100.0 / 255.0)          # L channel
    lab_img[:, :, 1] = lab_img[:, :, 1] - 128.0                    # a channel
    lab_img[:, :, 2] = lab_img[:, :, 2] - 128.0                    # b channel
    return lab_img


def lab2rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img.copy(), cv2.COLOR_Lab2RGB)


def auto_crop_white(img: np.ndarray, kernel_size: int = 10) -> np.ndarray:
    """Auto-crop the white border around the swatch in the image.
    Modified from https://python.plainenglish.io/effortlessly-remove-unwanted-whitespaces-from-images-with-opencv-and-python-e6707aac32c3

    Args:
        img (np.ndarray): Input image in OpenCV RGB color space (H, W, 3).
        kernel_size (int): Size of the morphological kernel to expand the detected swatch area.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, thresh=240, maxval=255, type=cv2.THRESH_BINARY_INV)  # <thresh -> 255 for bbox to detect swatch
    # Apply morphology to expand the non-white area
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(kernel_size, kernel_size))
    expanded_mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel=kernel)
    # Find contours in the expanded binary mask
    contours, _ = cv2.findContours(expanded_mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("Warning: No contours found, returning original image.")
        return img
    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    # Ensure the bounding box is within image dimensions
    x, y = max(x, 0), max(y, 0)
    w, h = min(w, img.shape[1] - x), min(h, img.shape[0] - y)
    # Crop the image using the bounding box
    return img[y:y + h, x:x + w]


def resize(img: np.ndarray, size: int = 224) -> np.ndarray:
    processed_img = cv2.resize(img.copy(), (size, size), interpolation=cv2.INTER_AREA)
    return processed_img


def preprocess(img: np.ndarray, size: int = 224) -> np.ndarray:
    """Preprocess the image by auto-cropping white borders and resizing.

    Args:
        img (np.ndarray): Input image in OpenCV RGB color space (H, W, 3).
        size (int): Desired output size (size x size).
    """
    return resize(auto_crop_white(img), size=size)


def mask_specular_and_shadow(lab_img: np.ndarray, drop_top=0.05, drop_bottom=0.05) -> np.ndarray:
    """Filter out specular highlights and shadows in the Lab color space.
    
    Args:
        lab_img (np.ndarray): Input image in OpenCV Lab color space (H, W, 3) with L, a, b in [0, 255].
        drop_top (float): Fraction of top L values to drop (specular highlights).
        drop_bottom (float): Fraction of bottom L values to drop (shadows).
    
    Returns:
        np.ndarray: Boolean mask with True for valid pixels and False for filtered pixels.
    """
    norm_img = lab_img.copy().astype(np.float32) / 255.0        # Normalize to [0, 1]
    specular_mask = norm_img[:, :, 0] > (1 - drop_top)
    shadow_mask = norm_img[:, :, 0] < drop_bottom
    keep_mask = ~(specular_mask | shadow_mask)          # to keep
    return keep_mask


def get_masked_image(lab_img: np.ndarray, rgb_img: np.ndarray, drop_top=0.05, drop_bottom=0.05) -> np.ndarray:
    """Get the masked image (in RGB) by filtering out specular highlights and shadows.

    Args:
        lab_img (np.ndarray): Resized input image in OpenCV Lab color space (H, W, 3) with L, a, b in [0, 255].
        rgb_img (np.ndarray): Resized input image in OpenCV RGB color space (H, W, 3).
        drop_top (float): Fraction of top L values to drop (specular highlights).
        drop_bottom (float): Fraction of bottom L values to drop (shadows).

    Returns:
        np.ndarray: Masked image with specular highlights and shadows removed.
    """
    assert lab_img.shape == rgb_img.shape, "lab_img and rgb_img must have the same shape"
    mask = mask_specular_and_shadow(lab_img, drop_top, drop_bottom)  # (H, W), bool
    # mask_3d = np.stack([mask] * 3, axis=-1)  # (H, W) -> (H, W, 3) for RGB
    masked_img = rgb_img.copy()
    masked_img[~mask] = 255  # Set specular highlights and shadows to white (broadcast)
    return masked_img
