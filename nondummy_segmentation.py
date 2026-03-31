# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 13:42:07 2026

@author: julia
"""

# -*- coding: utf-8 -*-
"""
Run DeepCell NuclearSegmentation on QuPath-exported tiles and save polygons
for each tile as a .txt file with the same basename.

Each output .txt contains one polygon per line:
x1,y1;x2,y2;x3,y3;...

Coordinates are LOCAL to the tile.
QuPath will shift them back using x/y from the filename.

@author: julia
"""

from pathlib import Path
import sys
import re
import math

import numpy as np
import tifffile as tiff

from deepcell.applications import NuclearSegmentation

# contour extraction
from skimage.measure import find_contours, label
from skimage.morphology import remove_small_objects


# =========================
# SETTINGS
# =========================
# fallback for running from Spyder without CLI args
DEFAULT_TILE_DIR = r"C:\Users\julia\OneDrive\Desktop\tiles_roi"

# model / prediction settings
WEIGHTS_PATH = r"C:\Users\julia\OneDrive\Desktop\melanoma\nuclear_finetuned_best.weights.h5"
IMAGE_MPP = 0.65

POSTPROCESS_KWARGS = {
    "maxima_threshold": 0.05,
    "interior_threshold": 0.3,
}

# polygon filtering
MIN_OBJECT_AREA = 40          # minimum object size in pixels
MIN_POLYGON_POINTS = 6        # discard tiny/degenerate contours
SIMPLIFY_EVERY_NTH = 2        # keep every nth contour point to make polygons lighter


# =========================
# IMAGE LOADING
# =========================
def load_nuclear_image(path):
    """
    Load image for NuclearSegmentation and return float32 array of shape (H, W, 1),
    normalized to [0, 1].

    Accepted:
    - 2D grayscale: (H, W)
    - 3D channels-last: (H, W, C)
    - 3D channels-first: (C, H, W)

    For multichannel images, takes channel 0 by default.
    """
    im = tiff.imread(path)
    if im is None:
        raise FileNotFoundError(f"Cannot read: {path}")

    im = np.asarray(im)
    im = np.squeeze(im)

    if im.ndim == 2:
        nuc = im

    elif im.ndim == 3:
        # channels-first
        if im.shape[0] in (1, 2, 3, 4) and im.shape[0] < im.shape[-1]:
            nuc = im[0]

        # channels-last
        elif im.shape[-1] in (1, 2, 3, 4):
            nuc = im[..., 0]

        else:
            raise ValueError(
                f"Unsupported 3D image shape {im.shape}. "
                "Expected (H, W, C) or (C, H, W) with small C."
            )
    else:
        raise ValueError(
            f"Unsupported image ndim={im.ndim}, shape={im.shape}"
        )

    nuc = nuc.astype(np.float32)
    nuc -= nuc.min()
    max_val = nuc.max()
    if max_val > 0:
        nuc /= max_val

    nuc = nuc[..., np.newaxis]  # (H, W, 1)
    return nuc


# =========================
# MASK -> POLYGONS
# =========================
def mask_to_polygons(label_mask,
                     min_area=MIN_OBJECT_AREA,
                     min_points=MIN_POLYGON_POINTS,
                     simplify_every_nth=SIMPLIFY_EVERY_NTH):
    """
    Convert integer label mask into a list of polygons.
    Each polygon is a list of (x, y) points in tile-local coordinates.
    """
    polygons = []

    label_mask = np.asarray(label_mask)
    if label_mask.ndim != 2:
        raise ValueError(f"label_mask must be 2D, got shape {label_mask.shape}")

    # remove tiny objects from the labeled mask
    cleaned = np.zeros_like(label_mask, dtype=np.int32)

    ids = np.unique(label_mask)
    ids = ids[ids > 0]

    next_id = 1
    for obj_id in ids:
        obj = label_mask == obj_id
        if obj.sum() < min_area:
            continue
        cleaned[obj] = next_id
        next_id += 1

    if cleaned.max() == 0:
        return polygons

    for obj_id in range(1, cleaned.max() + 1):
        obj = cleaned == obj_id
        if obj.sum() < min_area:
            continue

        # find_contours returns coordinates as (row, col) = (y, x)
        contours = find_contours(obj.astype(np.uint8), level=0.5)
        if not contours:
            continue

        # choose longest contour
        contour = max(contours, key=len)

        if simplify_every_nth > 1:
            contour = contour[::simplify_every_nth]

        if len(contour) < min_points:
            continue

        poly = []
        for y, x in contour:
            poly.append((float(x), float(y)))

        if len(poly) >= min_points:
            polygons.append(poly)

    return polygons


def polygon_to_line(poly):
    return ";".join(f"{x:.1f},{y:.1f}" for x, y in poly)


# =========================
# MAIN
# =========================
def main(tile_dir):
    tile_dir = Path(tile_dir)
    if not tile_dir.exists():
        raise FileNotFoundError(f"Folder not found: {tile_dir}")

    tif_files = sorted(tile_dir.glob("tile_x*_y*.tif"))
    if not tif_files:
        print("No tile tif files found.")
        return

    print(f"Found {len(tif_files)} tiles")
    print("Loading DeepCell NuclearSegmentation...")

    app = NuclearSegmentation()
    app.model.load_weights(WEIGHTS_PATH)
    print(f"Weights loaded: {WEIGHTS_PATH}")
    print(f"Model training resolution (mpp): {app.model_mpp}")

    for i, tif_path in enumerate(tif_files, start=1):
        print(f"[{i}/{len(tif_files)}] Processing {tif_path.name}")

        try:
            data = load_nuclear_image(tif_path)      # (H, W, 1)
            X = np.expand_dims(data, axis=0)         # (1, H, W, 1)

            seg = app.predict(
                X,
                image_mpp=IMAGE_MPP,
                postprocess_kwargs=POSTPROCESS_KWARGS
            )

            # expected shape usually (1, H, W, 1)
            seg = np.asarray(seg)

            if seg.ndim == 4:
                seg2d = seg[0, :, :, 0]
            elif seg.ndim == 3:
                seg2d = seg[0]
            else:
                raise ValueError(f"Unexpected prediction shape: {seg.shape}")

            seg2d = seg2d.astype(np.int32)

            polygons = mask_to_polygons(seg2d)

            out_txt = tif_path.with_suffix(".txt")
            if polygons:
                out_txt.write_text(
                    "\n".join(polygon_to_line(poly) for poly in polygons),
                    encoding="utf-8"
                )
            else:
                # write empty file so QuPath knows tile was processed
                out_txt.write_text("", encoding="utf-8")

            print(f"    objects: {len(polygons)} -> {out_txt.name}")

        except Exception as e:
            print(f"    ERROR in {tif_path.name}: {e}")
            # keep going, but still write empty result file
            out_txt = tif_path.with_suffix(".txt")
            out_txt.write_text("", encoding="utf-8")

    print("Done.")


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        tile_dir = sys.argv[1]
    else:
        print("No command-line tile_dir provided, using DEFAULT_TILE_DIR")
        tile_dir = DEFAULT_TILE_DIR

    main(tile_dir)