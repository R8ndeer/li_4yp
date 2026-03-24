#!/usr/bin/env python3
"""
Dataset CSV Generator

This script generates a CSV file containing color analysis data
for all individual shade samples.

Date: August 2025
Version: 1.0
"""

import numpy as np
import os, sys, csv
from pathlib import Path
from PIL import Image

import tqdm
from skimage import color

sys.path.append(str(Path(__file__).resolve().parent.parent))
from get_stats import get_channel_stats

# =============================================================================
# Configuration
# =============================================================================

# Paths
DATA_DIR = Path("data").resolve() / "demi_v01"
OUTPUT_CSV = DATA_DIR / "demi_v01.csv"

# CSV Headers
HEADER = [
    "image_id",  # 6.123_001 | ... (representative sample ID)
    "full_shade",  # 6.1 | 7.23 | ...
    "level",  # 1 | 2 | 3 | ... | 10
    "tone",  # .66 | .123 | ...
    "shade_family",  # X | XX | XY | XXX | ...
    "lab_illuminant",  # D65 | ...
    "lab_l",  # range: 0-100
    "lab_a",  # range: -128-127
    "lab_b",  # range: -128-127
    "sRGB_r",  # range: 0-1, normalized
    "sRGB_g",  # range: 0-1, normalized
    "sRGB_b",  # range: 0-1, normalized
    "linear_r",  # range: 0-1
    "linear_g",  # range: 0-1
    "linear_b",  # range: 0-1
]

# =============================================================================
# Color Conversion Functions
# =============================================================================


def srgb_to_linear(c):
    """Convert sRGB color value to linear RGB using proper gamma correction."""
    if c <= 0.04045:
        return c / 12.92
    else:
        return np.clip(((c + 0.055) / 1.055) ** 2.4, 0, 1)


def convert_srgb_to_linear(r, g, b):
    """Convert sRGB values (0-1 range) to linear RGB."""
    r_lin = np.round(srgb_to_linear(r), decimals=3)
    g_lin = np.round(srgb_to_linear(g), decimals=3)
    b_lin = np.round(srgb_to_linear(b), decimals=3)
    return (r_lin, g_lin, b_lin)


# =============================================================================
# Main Processing Function
# =============================================================================


def main():
    """
    Main processing function for dataset generation.
    Processes each image sample individually without averaging.
    """
    print("Starting dataset CSV generation...")

    # Ensure output directory exists
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_CSV, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(HEADER)

        for img_file in tqdm.tqdm(
            list(DATA_DIR.rglob("*.png")), desc="Processing samples"
        ):
            # Load and process image
            with Image.open(img_file) as img:
                img = np.array(img.convert("RGB"))
            img_lab = color.rgb2lab(img, illuminant="D65")

            # Extract metadata from file path
            fname = img_file.name
            render_id = fname.replace(".png", "")
            full_shade = fname.split("_")[0]
            level = img_file.parent.parent.name[1:]  # Remove 'L' prefix
            tone = full_shade.split(".")[1] if "." in full_shade else ""
            shade_family = img_file.parent.name

            # Calculate color values
            mode_r, mode_g, mode_b = get_channel_stats(img, "mode")
            linear_r, linear_g, linear_b = convert_srgb_to_linear(
                mode_r / 255.0, mode_g / 255.0, mode_b / 255.0
            )
            mode_l, mode_a, mode_b_lab = get_channel_stats(img_lab, "mode")

            # Write row to CSV
            writer.writerow(
                [
                    render_id,
                    full_shade,
                    level,
                    tone,
                    shade_family,
                    "D65",  # lab_illuminant
                    round(mode_l, 3),
                    round(mode_a, 3),
                    round(mode_b_lab, 3),
                    round(mode_r / 255.0, 3),  # Normalized to 0-1 range
                    round(mode_g / 255.0, 3),
                    round(mode_b / 255.0, 3),
                    round(linear_r, 3),
                    round(linear_g, 3),
                    round(linear_b, 3),
                ]
            )

    print(f"Dataset CSV generated successfully: {OUTPUT_CSV}")
    print(f"Processing complete!")


if __name__ == "__main__":
    main()
