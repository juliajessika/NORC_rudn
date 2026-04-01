# -*- coding: utf-8 -*-
"""
Run DeepCell NuclearSegmentation on QuPath-exported tiles and save polygons
for each tile as a .txt file with the same basename.

Each output .txt contains one polygon per line:
x1,y1;x2,y2;x3,y3;...

Coordinates are LOCAL to the tile.
QuPath will shift them back using x/y from the filename.
"""

from pathlib import Path
import argparse
import os
import sys

import numpy as np
import tifffile as tiff

from deepcell.applications import NuclearSegmentation
from skimage.measure import find_contours


def getenv_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value is not None and value != "" else default


def getenv_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return float(value)


def getenv_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


# =========================
# DEFAULTS
# =========================
DEFAULT_TILE_DIR = getenv_str("TILE_DIR", "/data")
DEFAULT_WEIGHTS_PATH = getenv_str("WEIGHTS_PATH", "/model/model.weights.h5")
DEFAULT_IMAGE_MPP = getenv_float("IMAGE_MPP", 0.65)

DEFAULT_MAXIMA_THRESHOLD = getenv_float("MAXIMA_THRESHOLD", 0.05)
DEFAULT_INTERIOR_THRESHOLD = getenv_float("INTERIOR_THRESHOLD", 0.3)

DEFAULT_MIN_OBJECT_AREA = getenv_int("MIN_OBJECT_AREA", 40)
DEFAULT_MIN_POLYGON_POINTS = getenv_int("MIN_POLYGON_POINTS", 6)
DEFAULT_SIMPLIFY_EVERY_NTH = getenv_int("SIMPLIFY_EVERY_NTH", 2)


# =========================
# IMAGE LOADING
# =========================
def load_nuclear_image(path: Path) -> np.ndarray:
    """
    Load image for NuclearSegmentation and return float32 array of shape (H, W, 1),
    normalized to [0, 1].

    Accepted:
    - 2D grayscale: (H, W)
    - 3D channels-last: (H, W, C)
    - 3D channels-first: (C, H, W)

    For multichannel images, takes channel 0 by default.
    """
    im = tiff.imread(str(path))
    if im is None:
        raise FileNotFoundError(f"Cannot read: {path}")

    im = np.asarray(im)
    im = np.squeeze(im)

    if im.ndim == 2:
        nuc = im

    elif im.ndim == 3:
        if im.shape[0] in (1, 2, 3, 4) and im.shape[0] < im.shape[-1]:
            nuc = im[0]
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
def mask_to_polygons(
    label_mask: np.ndarray,
    min_area: int,
    min_points: int,
    simplify_every_nth: int,
):
    """
    Convert integer label mask into a list of polygons.
    Each polygon is a list of (x, y) points in tile-local coordinates.
    """
    polygons = []

    label_mask = np.asarray(label_mask)
    if label_mask.ndim != 2:
        raise ValueError(f"label_mask must be 2D, got shape {label_mask.shape}")

    ids = np.unique(label_mask)
    ids = ids[ids > 0]

    for obj_id in ids:
        obj = (label_mask == obj_id)
        if int(obj.sum()) < min_area:
            continue

        contours = find_contours(obj.astype(np.uint8), level=0.5)
        if not contours:
            continue

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


def parse_args():
    parser = argparse.ArgumentParser(description="DeepCell segmentation on QuPath tiles")

    parser.add_argument(
        "--tile-dir",
        default=DEFAULT_TILE_DIR,
        help="Folder containing tile_x*_y*.tif files"
    )
    parser.add_argument(
        "--weights",
        default=DEFAULT_WEIGHTS_PATH,
        help="Path to .h5 or .weights.h5 model weights"
    )
    parser.add_argument(
        "--image-mpp",
        type=float,
        default=DEFAULT_IMAGE_MPP,
        help="Microns per pixel for DeepCell prediction"
    )
    parser.add_argument(
        "--maxima-threshold",
        type=float,
        default=DEFAULT_MAXIMA_THRESHOLD,
        help="DeepCell postprocess maxima_threshold"
    )
    parser.add_argument(
        "--interior-threshold",
        type=float,
        default=DEFAULT_INTERIOR_THRESHOLD,
        help="DeepCell postprocess interior_threshold"
    )
    parser.add_argument(
        "--min-object-area",
        type=int,
        default=DEFAULT_MIN_OBJECT_AREA,
        help="Minimum object area in pixels"
    )
    parser.add_argument(
        "--min-polygon-points",
        type=int,
        default=DEFAULT_MIN_POLYGON_POINTS,
        help="Minimum contour points to keep polygon"
    )
    parser.add_argument(
        "--simplify-every-nth",
        type=int,
        default=DEFAULT_SIMPLIFY_EVERY_NTH,
        help="Keep every nth contour point"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    tile_dir = Path(args.tile_dir)
    weights_path = Path(args.weights)

    if not tile_dir.exists():
        raise FileNotFoundError(f"Tile folder not found: {tile_dir}")

    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    tif_files = sorted(tile_dir.glob("tile_x*_y*.tif"))
    if not tif_files:
        print(f"No tile tif files found in {tile_dir}")
        return 0

    print(f"Found {len(tif_files)} tiles")
    print("Loading DeepCell NuclearSegmentation...")

    app = NuclearSegmentation()
    app.model.load_weights(str(weights_path))

    print(f"Weights loaded: {weights_path}")
    print(f"Model training resolution (model_mpp): {app.model_mpp}")
    print(f"Prediction image_mpp: {args.image_mpp}")

    postprocess_kwargs = {
        "maxima_threshold": args.maxima_threshold,
        "interior_threshold": args.interior_threshold,
    }

    error_count = 0

    for i, tif_path in enumerate(tif_files, start=1):
        print(f"[{i}/{len(tif_files)}] Processing {tif_path.name}")

        out_txt = tif_path.with_suffix(".txt")

        try:
            data = load_nuclear_image(tif_path)      # (H, W, 1)
            X = np.expand_dims(data, axis=0)         # (1, H, W, 1)

            seg = app.predict(
                X,
                image_mpp=args.image_mpp,
                postprocess_kwargs=postprocess_kwargs
            )

            seg = np.asarray(seg)

            if seg.ndim == 4:
                seg2d = seg[0, :, :, 0]
            elif seg.ndim == 3:
                seg2d = seg[0]
            else:
                raise ValueError(f"Unexpected prediction shape: {seg.shape}")

            seg2d = seg2d.astype(np.int32)

            polygons = mask_to_polygons(
                seg2d,
                min_area=args.min_object_area,
                min_points=args.min_polygon_points,
                simplify_every_nth=args.simplify_every_nth,
            )

            if polygons:
                out_txt.write_text(
                    "\n".join(polygon_to_line(poly) for poly in polygons),
                    encoding="utf-8"
                )
            else:
                out_txt.write_text("", encoding="utf-8")

            print(f"    objects: {len(polygons)} -> {out_txt.name}")

        except Exception as e:
            error_count += 1
            print(f"    ERROR in {tif_path.name}: {e}")
            out_txt.write_text("", encoding="utf-8")

    print(f"Done. Tiles with errors: {error_count}")
    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"FATAL ERROR: {e}")
        sys.exit(2)
