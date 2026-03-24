"""
Render shades with primary and secondary colors only.
"""

import bpy, csv, os
from math import pow
import numpy as np


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


def apply_shader_data(red, green, blue):
    mat = bpy.data.materials.get("Hair Material")
    if mat:
        nodes = mat.node_tree.nodes
        principled_hair_bsdf = nodes.get("Principled Hair BSDF")
        if principled_hair_bsdf:
            principled_hair_bsdf.inputs["Color"].default_value = (red, green, blue, 1.0)


def apply_shader_data_hex(hex):
    mat = bpy.data.materials.get("Hair Material")
    if mat:
        nodes = mat.node_tree.nodes
        principled_hair_bsdf = nodes.get("Principled Hair BSDF")
        if principled_hair_bsdf:
            principled_hair_bsdf.inputs["Color"].default_value = hex


# --- Load CSV ---
csv_path = "/Users/boting/li_4yp/side-project-yuv-colour-space/data/renders/v01/pred_renders_lab_v01.csv"
output_folder = "/Users/boting/li_4yp/side-project-yuv-colour-space/blender/output"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

with open(csv_path, newline="") as csvfile:
    reader = csv.reader(csvfile)
    header = next(reader)  # Skip header if exists

    row_cnt = 0
    for row in reader:
        render_id = row[0].strip()
        shade = row[0].strip().split("_")[0]

        # render testable shades
        if len(shade.split(".")[1]) > 2:
            continue

        # If the LAB values are all in one cell
        red = float(row[-3])
        green = float(row[-2])
        blue = float(row[-1])

        # # Apply the empirical transformation
        # (r,g,b) = convert_srgb_to_linear(red/255, green/255, blue/255)
        r, g, b = red, green, blue

        apply_shader_data(r, g, b)

        # Set Blender render output
        bpy.context.scene.render.filepath = os.path.join(output_folder, render_id)
        bpy.ops.render.render(write_still=True)

        # # Test the first n
        # row_cnt += 1
        # if row_cnt >= 1:
        #     break
