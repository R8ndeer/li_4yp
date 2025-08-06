import os
import re
import pandas as pd
from pathlib import Path

def parse_shade_from_filename(fname):
    # Match shade codes like "1.4", "1.34", "10.444" - only one dot with 1-3 digits after
    match = re.match(r"(\d+\.\d{1,3})", fname)
    if match:
        parts = match.group(1).split(".")
        parts = [parts[0]] + list(parts[1])
        while len(parts) < 4:
            parts.append(-1)  # pad missing
        return parts
    else:
        return [-1, -1, -1, -1]  # could not parse

# Loop through folder

root_dir = Path(__file__).resolve().parent.parent.parent / "data"
data_dir = root_dir / "ColourCorrectedImages"
rows = []
for fname in os.listdir(data_dir):
    if fname.lower().endswith(".png"):
        base, p1, p2, p3 = parse_shade_from_filename(fname)
        rows.append({"filename": fname, "Base": base, "Primary": p1, "Secondary": p2, "Tertiary": p3})

df_labels = pd.DataFrame(rows)
df_labels.to_csv(data_dir / "hair_swatch_labels.csv", index=False)
