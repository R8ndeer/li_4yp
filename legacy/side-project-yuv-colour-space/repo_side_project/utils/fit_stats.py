"""Functions for fitting various statistical distributions to image data."""

import numpy as np
from scipy.stats import skewnorm

# =============================================================================
# SKEW NORMAL DISTRIBUTION FITTING
# =============================================================================


def fit_skewnorm(image_data: np.ndarray) -> tuple:
    """Fit skew normal distribution to the channels of an image (channel-agnostic).

    Args:
        image_data (np.ndarray): Image data in shape (height, width, 3).

    Returns:
        tuple: Parameters of the fitted skew normal distributions for 3 channels.
               Order: (R, G, B) for RGB, (L, a, b) for LAB.
    """
    channel_1 = image_data[:, :, 0].flatten()
    channel_2 = image_data[:, :, 1].flatten()
    channel_3 = image_data[:, :, 2].flatten()
    params_1 = skewnorm.fit(channel_1)
    params_2 = skewnorm.fit(channel_2)
    params_3 = skewnorm.fit(channel_3)
    return params_1, params_2, params_3


def skewnorm_mode(a, loc, scale) -> float:
    """Calculate the mode of a skew normal distribution.

    Args:
        a (float): Skewness parameter.
        loc (float): Location parameter.
        scale (float): Scale parameter.

    Returns:
        float: Mode of the skew normal distribution.
    """
    # x = np.linspace(0, 255, 1000)
    x = np.linspace(-128, 255, 1000)  # Adjusted to cover a wider range for RGB and LAB
    pdf = skewnorm.pdf(x, a, loc=loc, scale=scale)
    return x[np.argmax(pdf)]


# =============================================================================
# GAUSSIAN MIXTURE MODEL (GMM) FITTING
# =============================================================================

# TODO: Implement GMM fitting functions
# - fit_gmm(image_data: np.ndarray) -> tuple
# - gmm_mode(weights, means, covariances) -> float
