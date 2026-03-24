#!/usr/bin/env python3
"""
CSV Data Averager

This script reads the individual dataset CSV and generates an averaged version
where multiple samples of the same shade are averaged together.

Date: August 2025
Version: 1.0
"""

import pandas as pd
from pathlib import Path

# =============================================================================
# Configuration
# =============================================================================

# Paths
INPUT_CSV = Path(__file__).resolve().parent.parent.parent / "data/demi_v1/demi_v01.csv"
OUTPUT_CSV = (
    Path(__file__).resolve().parent.parent.parent / "data/demi_v1/demi_v01_avg.csv"
)


def main():
    # Read the CSV file
    df = pd.read_csv(
        INPUT_CSV,
        dtype={"full_shade": str, "level": str, "tone": str, "shade_family": str},
    )

    print(f"Loaded {len(df)} individual samples from {INPUT_CSV}")
    print(f"Found {df['full_shade'].nunique()} unique shades")

    # Group by full_shade and calculate averages
    print("Grouping and averaging samples by full_shade...")

    # Define columns to average (all numeric color data)
    color_columns = [
        "lab_l",
        "lab_a",
        "lab_b",
        "sRGB_r",
        "sRGB_g",
        "sRGB_b",
        "linear_r",
        "linear_g",
        "linear_b",
    ]

    # Group by full_shade and aggregate
    grouped = (
        df.groupby("full_shade")
        .agg(
            {
                # Keep first value for metadata (should be consistent within shade)
                "level": "first",
                "tone": "first",
                "shade_family": "first",
                "lab_illuminant": "first",
                # Average all color values
                "lab_l": "mean",
                "lab_a": "mean",
                "lab_b": "mean",
                "sRGB_r": "mean",
                "sRGB_g": "mean",
                "sRGB_b": "mean",
                "linear_r": "mean",
                "linear_g": "mean",
                "linear_b": "mean",
                # Count samples
                "image_id": "count",
            }
        )
        .reset_index()
    )

    # Rename count column
    grouped.rename(columns={"image_id": "sample_count"}, inplace=True)

    # Round the averaged values for cleaner output
    for col in color_columns:
        grouped[col] = grouped[col].round(3)

    # Reorder columns for better readability
    column_order = [
        "full_shade",
        "level",
        "tone",
        "shade_family",
        "lab_illuminant",
        "lab_l",
        "lab_a",
        "lab_b",
        "sRGB_r",
        "sRGB_g",
        "sRGB_b",
        "linear_r",
        "linear_g",
        "linear_b",
        "sample_count",
    ]

    averaged_df = grouped[column_order]

    # Ensure output directory exists
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    # Save averaged dataset
    print(f"Saving averaged dataset to: {OUTPUT_CSV}")
    averaged_df.to_csv(OUTPUT_CSV, index=False)

    # Print summary statistics
    print(f"\nAveraging complete!")
    print(f"Original samples: {len(df)}")
    print(f"Unique shades: {len(averaged_df)}")
    print(
        f"Samples reduced by: {len(df) - len(averaged_df)} ({((len(df) - len(averaged_df)) / len(df) * 100):.1f}%)"
    )

    # Show sample count distribution
    sample_counts = averaged_df["sample_count"].value_counts().sort_index()
    print(f"\nSample count distribution:")
    for count, freq in sample_counts.items():
        print(f"  {count} sample(s): {freq} shades")

    # Show examples of multi-sample shades
    multi_sample_shades = averaged_df[averaged_df["sample_count"] > 1].sort_values(
        "sample_count", ascending=False
    )
    if not multi_sample_shades.empty:
        print(f"\nTop shades with multiple samples:")
        for _, shade in multi_sample_shades.head(10).iterrows():
            print(f"  {shade['full_shade']}: {shade['sample_count']} samples")

    print(f"\nAveraged dataset saved successfully!")


if __name__ == "__main__":
    main()
