#!/usr/bin/env python3
"""
Theoretical Recipe Generator

This script generates theoretical mixing recipes for all non-primary matrix shades.
The primary matrix (X, XX, XXX shade families) serves as the base ingredients
for creating all other possible shades through mixing logic.

Author: Color Analysis Pipeline
Date: August 2025
Version: 1.0

Features:
- Theoretical recipe generation for non-primary shades
- Mixing logic determination based on shade families
- Parent shade validation and dependency mapping
- No dataset dependency - pure algorithmic generation
"""

import numpy as np
import os, sys, csv
from pathlib import Path
from itertools import product

import tqdm

# =============================================================================
# Configuration
# =============================================================================

# Paths
OUTPUT_CSV = Path("data").resolve() / "recipes" / "recipes_v01.csv"

# Shade generation parameters
LEVELS = list(range(1, 11))  # 1-10
TONE_DIGITS = list(range(10))  # 0-9

# CSV Headers - Theoretical recipes only need mixing info, no color data
HEADER = [
    "render_id",                # 6.123 | 7.05 | ... (unique shade identifier)
    "full_shade",               # 6.123 | 7.05 | ...
    "level",                    # 1 | 2 | 3 | ... | 10
    "tone",                     # .123 | .05 | ...
    "shade_family",             # 0X | XY | XXX | XYZ | etc.
    "mix_logic",                # L-X | XX-Y | PM | ...
    "mix_ratios",               # 50-50 | 67-33 | ...
    "parents",                  # ["6.1", "7.2"] | ...
    "is_primary",               # True/False - whether this is a primary matrix shade
]

# Mixing Logic Maps
MIX_LOGIC_MAP = {
    "X": "PM",          # Primary Matrix
    "XX": "PM",         # Primary Matrix  
    "XXX": "PM",        # Primary Matrix
    "0X": "L-X",        # Level - X (50-50 ratio)
    "XY": "XX-Y",       # XX - Y (67-33 ratio)
    "XYZ": "XXX-YY-Z",  # XXX - YY - Z (57-28-15 ratio)
    "X0Y": "X-Y",       # X - Y
    "0XY": "0X-YYY",    # 0X - YYY
    "00X": "L-X",       # Level - X (67-33 ratio)
    "L": "L"            # Level (for whole numbers without decimals)
}

MIX_RATIOS_MAP = {
    "X": "",            # No mixing for primary
    "XX": "",           # No mixing for primary
    "XXX": "",          # No mixing for primary
    "0X": "50-50",      # Even mix
    "XY": "67-33",      # Weighted mix
    "XYZ": "57-28-15",  # Three-way mix
    "X0Y": "",          # Custom ratio (varies)
    "0XY": "",          # Custom ratio (varies) 
    "00X": "67-33",     # Weighted mix
    "L": ""             # No mixing for level
}

# =============================================================================
# Shade Generation Functions
# =============================================================================

def classify_shade_family(tone_str):
    """Classify shade family based on tone pattern."""
    if not tone_str:
        return "L"  # Level only (no tone)

    if len(tone_str) == 1:
        if tone_str == '0':
            return "L"  # Level only
        else:
            return "X"
    elif len(tone_str) == 2:
        if tone_str[0] == '0':
            return "0X"
        elif tone_str[0] == tone_str[1]:
            return "XX"
        else:
            return "XY"
    elif len(tone_str) == 3:
        if tone_str[:2] == "00":
            return "00X"
        elif tone_str[0] == '0':
            return "0XY"
        elif tone_str[0] == tone_str[1] == tone_str[2]:
            return "XXX"
        elif tone_str[0] != '0' and tone_str[1] == '0':
            return "X0Y"
        else:
            return "XYZ"

    return "UNKNOWN"


def generate_all_possible_shades():
    """Generate all possible shade combinations, avoiding redundant trailing zeros."""
    shades = []
    
    for level in LEVELS:
        # Level only (no tone)
        shades.append({
            'full_shade': str(level),
            'level': str(level),
            'tone': "",
            'shade_family': "L"
        })
        
        # Single digit tones (1-digit)
        for d1 in TONE_DIGITS:
            if d1 == 0:  # Skip 0 as single digit, it's covered by level only
                continue
            tone_str = str(d1)
            shades.append({
                'full_shade': f"{level}.{tone_str}",
                'level': str(level),
                'tone': tone_str,
                'shade_family': classify_shade_family(tone_str)
            })
        
        # Two digit tones (2-digit combinations)
        for d1, d2 in product(TONE_DIGITS, repeat=2):
            if d1 == 0 and d2 == 0:  # Skip 00, it's essentially level only
                continue
            if d2 == 0:  # Skip trailing zero (e.g., 10, 20, etc.)
                continue
            tone_str = f"{d1}{d2}"
            shades.append({
                'full_shade': f"{level}.{tone_str}",
                'level': str(level),
                'tone': tone_str,
                'shade_family': classify_shade_family(tone_str)
            })
        
        # Three digit tones (3-digit combinations)
        for d1, d2, d3 in product(TONE_DIGITS, repeat=3):
            if d1 == 0 and d2 == 0 and d3 == 0:  # Skip 000
                continue
            if d3 == 0:  # Skip trailing zero (e.g., 120, 340, etc.)
                continue
            if d2 == 0 and d3 == 0:  # Skip double trailing zeros (e.g., 100, 200, etc.)
                continue
            tone_str = f"{d1}{d2}{d3}"
            shades.append({
                'full_shade': f"{level}.{tone_str}",
                'level': str(level),
                'tone': tone_str,
                'shade_family': classify_shade_family(tone_str)
            })
    
    return shades

# =============================================================================
# Mixing Logic Functions
# =============================================================================

def get_mix_info(shade_family):
    """Determine mixing logic and ratios based on shade family pattern."""
    mix_logic = MIX_LOGIC_MAP.get(shade_family, "UNKNOWN")
    mix_ratios = MIX_RATIOS_MAP.get(shade_family, "")
    return mix_logic, mix_ratios


def determine_parents(full_shade, shade_family):
    """Determine parent shades required for mixing based on shade family."""
    if shade_family in ["X", "XX", "XXX", "L"]:
        return []  # Primary shades have no parents
    
    level = full_shade.split('.')[0] if '.' in full_shade else full_shade
    tone = full_shade.split('.')[1] if '.' in full_shade else ""
    
    if shade_family == "0X":
        # Mix of level and X tone
        tone = full_shade.split('.')[1][1:] if '.' in full_shade else ""
        return [level, f"{level}.{tone}"]
    elif shade_family == "XY":
        # Mix of XX and Y
        tone = full_shade.split('.')[1] if '.' in full_shade else ""
        if len(tone) >= 2:
            return [f"{level}.{tone[0]}{tone[0]}", f"{level}.{tone[1]}"]
    elif shade_family == "00X":
        # Mix of level and X tone
        tone = full_shade.split('.')[1][2:] if '.' in full_shade else ""
        return [level, f"{level}.{tone}"]
    elif shade_family == "XYZ":
        # Mix of XXX, YY, and Z
        return [f"{level}.{tone[0]}{tone[0]}{tone[0]}", f"{level}.{tone[1]}{tone[1]}", f"{level}.{tone[2]}"]
    
    return []  # Default: no parents identified

# =============================================================================
# Main Processing Function
# =============================================================================

def main():
    """
    Main function to generate theoretical recipes for all possible shades.
    Only non-primary matrix shades (non X, XX, XXX) will have mixing recipes.
    """
    print("Starting recipe generation...")
    
    # Ensure output directory exists
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate all possible shades
    print("Generating all possible shade combinations...")
    all_shades = generate_all_possible_shades()
    
    print(f"Generated {len(all_shades)} possible shades")
    
    # Filter and process shades
    with open(OUTPUT_CSV, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(HEADER)
        
        for shade in tqdm.tqdm(all_shades, desc="Processing shades"):
            full_shade = shade['full_shade']
            level = shade['level']
            tone = shade['tone']
            shade_family = shade['shade_family']
            
            # Determine if this is a primary matrix shade
            is_primary = shade_family in ["X", "XX", "XXX", "L"]
            
            # Get mixing information
            mix_logic, mix_ratios = get_mix_info(shade_family)
            parents = determine_parents(full_shade, shade_family)
            
            # Use full_shade as render_id (no _001 suffix needed)
            render_id = full_shade
            
            # Write row to CSV
            writer.writerow([
                render_id,
                full_shade,
                level,
                tone,
                shade_family,
                mix_logic,
                mix_ratios,
                str(parents),  # Convert list to string
                is_primary
            ])
    
    # Generate summary statistics
    primary_count = sum(1 for shade in all_shades if shade['shade_family'] in ["X", "XX", "XXX", "L"])
    recipe_count = len(all_shades) - primary_count
    
    print(f"Recipe CSV generated successfully: {OUTPUT_CSV}")
    print(f"Total shades: {len(all_shades)}")
    print(f"Primary matrix shades: {primary_count}")
    print(f"Recipe shades (non-primary): {recipe_count}")
    
    # Print breakdown by shade family
    family_counts = {}
    for shade in all_shades:
        family = shade['shade_family']
        family_counts[family] = family_counts.get(family, 0) + 1
    
    print("\nShade family breakdown:")
    for family, count in family_counts.items():
        primary_status = "PRIMARY" if family in ["X", "XX", "XXX", "L"] else "RECIPE"
        print(f"  {family}: {count} shades ({primary_status})")


if __name__ == "__main__":
    main()
