from fit_stats import *

def __get_mode(image_data: np.ndarray) -> tuple:
    """Get the mode of 3 colour channels from image data using skew normal distribution.
    
    Args:
        image_data (np.ndarray): Image data in shape (height, width, 3).
        
    Returns:
        tuple: Modes of the 3 colour channels.
    """
    params_1, params_2, params_3 = fit_skewnorm(image_data)
    mode_1 = skewnorm_mode(*params_1)
    mode_2 = skewnorm_mode(*params_2)
    mode_3 = skewnorm_mode(*params_3)

    return round(mode_1, 2), round(mode_2, 2), round(mode_3, 2)


def get_channel_stats(image_data: np.ndarray, metric: str) -> tuple:
    """Wrapper function to get different colour channel stats values of an image."""
    if metric == "mode":
        return __get_mode(image_data)
    else:
        raise ValueError(f"Unsupported stats metric: {metric}")