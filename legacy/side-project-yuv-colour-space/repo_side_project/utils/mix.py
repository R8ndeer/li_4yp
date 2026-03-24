# Colour space mixing
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data/demi_v1"
DATA_CSV = ROOT / "data/demi_v1/demi_v1.csv"
RECIPE_CSV = ROOT / "data/recipes/recipe_full_v01.csv"


def load_recipes(csv_path=RECIPE_CSV):
    """Load non-trivial mixing recipes from a CSV file."""
    recipes = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["parents"] == "[]":
                continue
            parents = list(
                row["parents"].strip("[]").replace('"', "").replace("'", "").split(", ")
            )
            render_id = row["render_id"]
            mix_logic = row["mix_logic"]
            mix_ratios = row["mix_ratios"]
            recipes[render_id] = {
                "parents": parents,
                "mix_logic": mix_logic,
                "mix_ratios": mix_ratios,
            }
    return recipes


if __name__ == "__main__":
    recipes = load_recipes()
    print(f"Loaded {len(recipes)} recipes.")
    for k, v in recipes.items():
        print(f"{k}: {v}")
