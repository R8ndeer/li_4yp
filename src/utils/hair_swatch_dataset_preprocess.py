import os
import re
import pandas as pd
from pathlib import Path

def parse_shade_from_filename(fname):
    # First check for "half" bases - if there's "1_2" before the first dot, ignore
    dot_pos = fname.find('.')
    if dot_pos != -1:
        part_before_dot = fname[:dot_pos]
        if '1_2' in part_before_dot:  # "half" base like "10 1_2.029" - ignore these
            return [-1, -1, -1, -1]
    
    # Case 1: Base digit only (no dot) - match digits followed by non-digit/non-dot
    base_only_match = re.match(r"(\d+)(?=\s|_|\(|\.png|$)", fname)
    if base_only_match and '.' not in fname[:len(base_only_match.group(1)) + 1]:
        base = base_only_match.group(1)
        return [base, -1, -1, -1]
    
    # Case 2: Base + decimal digits - base (1-2 digits) followed by dot and 1-3 digits
    decimal_match = re.match(r"(\d{1,2})\.(\d{1,3})", fname)
    if decimal_match:
        base = decimal_match.group(1)
        decimal_part = decimal_match.group(2)
        parts = [base] + list(decimal_part)
        while len(parts) < 4:
            parts.append(-1)  # pad missing
        return parts
    
    # Could not parse
    return [-1, -1, -1, -1]

def main():
    # Loop through folder
    root_dir = Path(__file__).resolve().parent.parent.parent / "data"
    data_dir = root_dir / "ColourCorrectedImages"
    rows = []
    for fname in os.listdir(data_dir):
        if fname.lower().endswith(".png"):
            base, p1, p2, p3 = parse_shade_from_filename(fname)
            if base == -1:
                print(f"Skipping file with unparseable name: {fname}")
                continue
            rows.append({"filename": fname, "Base": base, "Primary": p1, "Secondary": p2, "Tertiary": p3})

    df_labels = pd.DataFrame(rows)
    df_labels.to_csv(data_dir / "hair_swatch_labels_test_2.csv", index=False)


if __name__ == "__main__":
    main()