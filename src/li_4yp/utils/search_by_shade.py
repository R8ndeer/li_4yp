# Simple script to search for shades in a dataset
from pathlib import Path
import pandas as pd

def main():
    file_path = Path(__file__).resolve().parent.parent.parent / "data" / "tone_to_mix_ratio_demi.csv"
    df = pd.read_csv(file_path)
    input_shade = input("Enter the 2 shade codes to search for (e.g., '10.123, 7.8'): ")
    shade_codes = [code.strip() for code in input_shade.split(",")]
    for code in shade_codes:
        if code not in df['shade'].values:
            print(f"Shade '{code}' not found in the dataset.")
            return
    shade_data = df[df['shade'].isin(shade_codes)]
    print(f"Data for shades {shade_codes}:")
    print(shade_data)


if __name__ == "__main__":
    main()