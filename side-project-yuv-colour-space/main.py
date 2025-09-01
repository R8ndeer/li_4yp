# main.py
"""
Purpose:
    - Query a dataset CSV using a recipe CSV to filter out unrenderable recipes.
    - Create a new CSV file for Blender to render.

Usage:
    python main.py <dataset_csv_path> <recipe_csv_path> <output_csv_path>

Author: Boting Li
Date: August 2025
"""
import csv
import argparse
from pathlib import Path

import tqdm
import numpy as np
import pandas as pd
from skimage import color

# MACRO
VERSION = "v01"

# Parse arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--color_space", type=str, nargs='?', default="srgb", help="Color space to use (srgb, linear, lab)")
    parser.add_argument("--dataset_csv", type=str, nargs='?', help="Path to the dataset CSV file")
    parser.add_argument("--recipe_csv", type=str, nargs='?', help="Path to the recipe CSV file")
    parser.add_argument("--output_csv", type=str, nargs='?', help="Path to the output CSV file")
    return parser.parse_args()


# Color conversion functions
def srgb_to_linear(c):
    if c <= 0.04045:
        return c / 12.92
    else:
        return np.clip(((c + 0.055) / 1.055) ** 2.4, 0, 1)


def convert_srgb_to_linear(r, g, b, a=1.0):
    r_lin = srgb_to_linear(r)
    g_lin = srgb_to_linear(g)
    b_lin = srgb_to_linear(b)
    return (r_lin, g_lin, b_lin)


# Main function
def main():
    print("Creating renderable dataset CSV for Blender...")


    # Parse args
    args = parse_args()
    color_space = args.color_space.lower()
    if color_space not in ["srgb", "linear", "lab"]:
        raise ValueError("color_space must be one of: srgb, linear, lab")
    print(f"Using color space: {color_space}")

    ROOT_DIR = Path(__file__).resolve().parent
    DATA_CSV = Path(args.dataset_csv).resolve() if args.dataset_csv else ROOT_DIR / f"data/demi_{VERSION}/demi_{VERSION}.csv"
    RECIPE_CSV = Path(args.recipe_csv).resolve() if args.recipe_csv else ROOT_DIR / f"data/recipes/recipes_{VERSION}.csv"
    OUTPUT_CSV = Path(args.output_csv).resolve() if args.output_csv else ROOT_DIR / f"data/renders/{VERSION}/pred_renders_{color_space}_{VERSION}.csv"
    print("Using")
    print(f"  dataset CSV: {DATA_CSV}\n  recipe CSV:  {RECIPE_CSV}\n  output CSV:  {OUTPUT_CSV}")


    # Load dataset and recipes
    ds_df = pd.read_csv(DATA_CSV, dtype={"full_shade": str, "level": str, "tone": str, "shade_family": str})
    recipe_df = pd.read_csv(RECIPE_CSV, dtype={"render_id": str, "full_shade": str, "level": str, "tone": str, "shade_family": str})
    print(f"Loaded {len(ds_df)} dataset samples and {len(recipe_df)} recipes")

    # Prepare output file
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_CSV, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "render_id", 
            "full_shade", 
            "level",
            "tone",
            "shade_family",
            "mix_color_space",
            "pred_linear_r", 
            "pred_linear_g", 
            "pred_linear_b"
        ])

        # Filter to find renderable recipes
        for _, recipe in tqdm.tqdm(list(recipe_df.iterrows())):
            if recipe["is_primary"]:
                continue

            parents = recipe["parents"].strip("[]").replace('"', "").replace("'", "").split(", ")
            # Check if all parents exist in dataset
            if not all(parent in ds_df["full_shade"].values for parent in parents):
                continue

            # Get parent colors
            if color_space == "srgb":
                channels = ["sRGB_r", "sRGB_g", "sRGB_b"]
            elif color_space == "linear":
                channels = ["linear_r", "linear_g", "linear_b"]
            else:  # lab
                channels = ["lab_l", "lab_a", "lab_b"]
            
            parent_colors = []
            for parent in parents:
                parent_row = ds_df.loc[ds_df["full_shade"] == parent][channels].iloc[0].values
                parent_colors.append(parent_row)

            # Calculate mixed color
            mix_ratios = [int(r) for r in recipe["mix_ratios"].split("-")]  # e.g., "50-50" -> [50, 50]
            assert len(parent_colors) == len(mix_ratios), "Number of parents and mix ratios must match"
            predicted_color = np.average(parent_colors, axis=0, weights=mix_ratios)

            if color_space == "srgb":
                predicted_color = convert_srgb_to_linear(*predicted_color)
            elif color_space == "lab":
                predicted_color = color.lab2rgb(predicted_color, illuminant="D65")
                predicted_color = convert_srgb_to_linear(*predicted_color)
            
            writer.writerow([
                f"{recipe['render_id']}_pred_{color_space}",
                recipe["full_shade"],
                recipe["level"],
                recipe["tone"],
                recipe["shade_family"],
                color_space,
                np.round(predicted_color[0], 3),
                np.round(predicted_color[1], 3),
                np.round(predicted_color[2], 3)
            ])

if __name__ == "__main__":
    main()
