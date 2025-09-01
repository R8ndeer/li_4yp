"""Reorganize files (from ColourCorrectedImages) into a new directory structure based on shade codes and patterns."""

import os
import re
import shutil
from collections import defaultdict
from pathlib import Path

VERSION = "v01"  # Directory version

def parse_shade_from_filename(fname):
    """Extract shade code from filename (e.g., '6.1', '7.23', '10.444')"""
    # Match shade codes like "1.4", "1.34", "10.444" - only one dot with 1-3 digits after
    match = re.match(r"(\d+\.\d{1,3})", fname)
    if match:
        return match.group(1)
    else:
        return None

def reorganize_files():
    """Reorganize files into the new directory structure"""
    root_dir = Path(__file__).resolve().parent.parent.parent / "data"
    source_dir = root_dir / "ColourCorrectedImages"
    output_dir = root_dir / f"demi_{VERSION}"  # New output directory

    print(f"Number of files in source directory: {len(list(source_dir.iterdir()))}")
    
    # Remove existing output directory if it exists
    if output_dir.exists():
        shutil.rmtree(output_dir)
        print(f"Removed existing directory: {output_dir}")
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Group files by shade code
    shade_files = defaultdict(list)
    
    # Collect all PNG files and group them by shade code
    for fname in os.listdir(source_dir):
        if fname.lower().endswith(".png"):
            shade_code = parse_shade_from_filename(fname)
            if shade_code:
                shade_files[shade_code].append(fname)
            else:
                print(f"Warning: Could not parse shade code from {fname}")
    
    def categorize_decimal_pattern(decimal_part):
        """Categorize decimal part based on digit patterns"""
        length = len(decimal_part)
        
        if length == 1:
            return "X"
        elif length == 2:
            d1, d2 = decimal_part[0], decimal_part[1]
            if d1 == d2:
                return "XX"  # Same digits
            elif d1 == '0':
                return "0X"  # Zero followed by non-zero
            else:
                return "XY"  # Two different non-zero digits
        elif length == 3:
            d1, d2, d3 = decimal_part[0], decimal_part[1], decimal_part[2]
            
            if d1 == '0' and d2 == '0':
                return "00X"  # Pattern: 00X (e.g., 001, 002, 009)
            elif d1 == '0':
                return "0XY"  # Pattern: 0XY (e.g., 012, 023, 034)
            elif d2 == '0':
                return "X0Y"  # Pattern: X0Y (e.g., 102, 203, 304)
            elif d1 == d2 == d3:
                return "XXX"  # Pattern: XXX (e.g., 111, 222, 333)
            else:
                return "XYZ"  # Pattern: XYZ (e.g., 123, 234, 567)
        else:
            # Fallback for unexpected lengths (4+ digits)
            return "X" * length

    # Process each shade group
    for shade_code, files in shade_files.items():
        # Extract base level (e.g., "6" from "6.1", "7" from "7.23")
        base_level = shade_code.split('.')[0]
        
        # Determine decimal precision category
        decimal_part = shade_code.split('.')[1]
        precision_dir = categorize_decimal_pattern(decimal_part)
        
        # Create directory structure: real/L{base}/{precision}/
        level_dir = output_dir / f"L{base_level}"
        precision_subdir = level_dir / precision_dir
        
        # Create directories
        precision_subdir.mkdir(parents=True, exist_ok=True)
        
        # Copy and rename files with numbering
        for i, fname in enumerate(sorted(files), 1):
            # Create new filename: shade_code_001.png, shade_code_002.png, etc.
            new_fname = f"{shade_code}_{i:03d}.png"
            
            source_path = source_dir / fname
            dest_path = precision_subdir / new_fname
            
            # Copy file to new location
            shutil.copy2(source_path, dest_path)
            # print(f"Copied: {fname} -> {dest_path.relative_to(output_dir)}")
    
    print(f"\nReorganization complete!")
    print(f"Files organized in: {output_dir}")
    print(f"Total shade codes processed: {len(shade_files)}")

if __name__ == "__main__":
    reorganize_files()
