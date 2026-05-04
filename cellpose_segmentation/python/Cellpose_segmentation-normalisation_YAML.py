# -*- coding: utf-8 -*-
"""
Cellpose segmentation for QuPath-exported tiles.
Reads segmentation/postprocess parameters from YAML.
"""

from pathlib import Path
import sys
import traceback
import warnings

import numpy as np
import tifffile as tiff
from skimage.measure import find_contours
import yaml

from cellpose import models, core

warnings.filterwarnings("ignore")


# =========================
# DEFAULT CONFIG
# =========================
DEFAULTS = {
    "cellpose": {
        # launcher keys may also exist in YAML (conda_exe, conda_env, python_script),
        # but they are not used directly by this script
        "model_type": "cyto3",
        "pretrained_model": None,        # if set, overrides model_type
        "diameter": 30,                  # set None for automatic diameter
        "flow_threshold": 0.8,
        "cellprob_threshold": 0.0,
        "tile_norm_blocksize": 0,
        "batch_size": 32,
        "channels_mode": "auto",         # auto | explicit
        "explicit_channels": [0, 0],     # used if channels_mode == explicit
    },
    "postprocess": {
        "min_object_area": 40,
        "min_polygon_points": 6,
        "simplify_every_nth": 2,
    },
}


# =========================
# CONFIG HELPERS
# =========================
def deep_update(base: dict, upd: dict) -> dict:
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(config_path=None):
    cfg = {
        "cellpose": dict(DEFAULTS["cellpose"]),
        "postprocess": dict(DEFAULTS["postprocess"]),
    }

    if config_path is None:
        return cfg

    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")

    with p.open("r", encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f) or {}

    if not isinstance(user_cfg, dict):
        raise ValueError("YAML root must be a mapping/object")

    deep_update(cfg, user_cfg)
    return cfg


# =========================
# IMAGE LOADING & CHANNEL SELECTION
# =========================
def load_and_select_channels(path, channel_indices):
    """
    Load image and select specified channels.
    Matches robust logic for 2D/3D, channels-last/channels-first.
    """
    im = tiff.imread(path)
    im = np.asarray(im)
    im = np.squeeze(im)

    print(f"  Original shape: {im.shape}")

    if im.ndim == 2:
        print("  Detected single-channel (grayscale) image")
        im = im[:, :, np.newaxis]  # (H, W) -> (H, W, 1)
        print(f"  Reshaped to: {im.shape}")

    elif im.ndim == 3:
        # If first dim looks like channel dim and last dim is larger -> transpose to HWC
        if im.shape[0] <= 4 and im.shape[0] < im.shape[-1]:
            print("  Detected channels-first format, transposing...")
            im = np.transpose(im, (1, 2, 0))
            print(f"  New shape: {im.shape}")

    else:
        raise ValueError(f"Unsupported image ndim={im.ndim}, shape={im.shape}")

    n_channels = im.shape[-1]
    print(f"  Number of channels: {n_channels}")

    if n_channels == 1:
        print("  Single-channel image")
        return im.astype(np.float32)

    valid_indices = [i for i in channel_indices if 0 <= i < n_channels]
    if not valid_indices:
        raise ValueError(f"No valid channels! Requested {channel_indices}, image has {n_channels} channels")

    if len(valid_indices) < len(channel_indices):
        print(f"  Warning: Using only {valid_indices} (requested {channel_indices})")

    selected = im[:, :, valid_indices]
    print(f"  Selected channels shape: {selected.shape}")
    return selected.astype(np.float32)


# =========================
# MASK -> POLYGONS
# =========================
def mask_to_polygons(label_mask, min_area=40, min_points=6, simplify_every_nth=2):
    """Convert integer label mask into a list of polygons."""
    from skimage import morphology

    polygons = []
    label_mask = np.asarray(label_mask)

    if label_mask.ndim != 2:
        raise ValueError(f"label_mask must be 2D, got shape {label_mask.shape}")

    if min_area > 0:
        cleaned = morphology.remove_small_objects(label_mask, min_size=min_area)
    else:
        cleaned = label_mask

    max_id = int(cleaned.max()) if cleaned.size else 0
    if max_id == 0:
        return polygons

    for obj_id in range(1, max_id + 1):
        obj = cleaned == obj_id
        if obj.sum() < min_area:
            continue

        contours = find_contours(obj.astype(np.uint8), level=0.5)
        if not contours:
            continue

        contour = max(contours, key=len)

        if simplify_every_nth > 1:
            contour = contour[::simplify_every_nth]

        if len(contour) < min_points:
            continue

        # (y, x) -> (x, y)
        poly = [(round(float(x), 1), round(float(y), 1)) for y, x in contour]
        if len(poly) >= min_points:
            polygons.append(poly)

    return polygons


def polygon_to_line(poly):
    """Convert polygon to line format: x1,y1;x2,y2;..."""
    return ";".join(f"{x:.1f},{y:.1f}" for x, y in poly)


# =========================
# MAIN
# =========================
def main(tile_dir, channels_str="0", config_path=None):
    tile_dir = Path(tile_dir)
    if not tile_dir.exists():
        raise FileNotFoundError(f"Folder not found: {tile_dir}")

    cfg = load_config(config_path)
    cp = cfg["cellpose"]
    pp = cfg["postprocess"]

    print("Loaded config:")
    print(f"  cellpose: {cp}")
    print(f"  postprocess: {pp}")

    # Parse channels passed by Groovy (post-export channel indices)
    channel_indices = [int(c.strip()) for c in channels_str.split(",") if c.strip() != ""]
    if not channel_indices:
        channel_indices = [0]
    print(f"Groovy channels_to_use: {channel_indices}")

    tif_files = sorted(tile_dir.glob("tile_x*_y*.tif"))
    if not tif_files:
        tif_files = sorted(tile_dir.glob("tile_x*_y*.tiff"))

    if not tif_files:
        print("No tile tif files found.")
        return

    print(f"Found {len(tif_files)} tiles")

    print("Loading Cellpose model...")
    use_gpu = core.use_gpu()
    print(f"GPU available: {use_gpu}")

    pretrained = cp.get("pretrained_model", None)
    if pretrained in ("", "null", "None"):
        pretrained = None

    if pretrained:
        model = models.CellposeModel(gpu=use_gpu, pretrained_model=pretrained)
        print(f"Model loaded: custom -> {pretrained}")
    else:
        model_type = cp.get("model_type", "cyto3")
        model = models.CellposeModel(gpu=use_gpu, model_type=model_type)
        print(f"Model loaded: {model_type}")

    batch_size = int(cp.get("batch_size", 32))
    diameter = cp.get("diameter", 30)
    diameter = None if diameter is None else float(diameter)
    flow_threshold = float(cp.get("flow_threshold", 0.8))
    cellprob_threshold = float(cp.get("cellprob_threshold", 0.0))
    tile_norm_blocksize = int(cp.get("tile_norm_blocksize", 0))

    channels_mode = str(cp.get("channels_mode", "auto")).strip().lower()
    explicit_channels = cp.get("explicit_channels", [0, 0])

    min_object_area = int(pp.get("min_object_area", 40))
    min_polygon_points = int(pp.get("min_polygon_points", 6))
    simplify_every_nth = int(pp.get("simplify_every_nth", 2))

    for i, tif_path in enumerate(tif_files, start=1):
        print(f"[{i}/{len(tif_files)}] Processing {tif_path.name}")
        out_txt = tif_path.with_suffix(".txt")

        try:
            img_data = load_and_select_channels(tif_path, channel_indices)

            if img_data.dtype != np.float32:
                img_data = img_data.astype(np.float32)

            n_channels = img_data.shape[-1] if img_data.ndim == 3 else 1

            if channels_mode == "explicit":
                if not (isinstance(explicit_channels, list) and len(explicit_channels) == 2):
                    raise ValueError("cellpose.explicit_channels must be a list of 2 ints, e.g. [0,0]")
                channels_param = [int(explicit_channels[0]), int(explicit_channels[1])]
            else:
                # auto mode: keep your previous behavior
                if n_channels == 1:
                    channels_param = [0, 0]
                else:
                    channels_param = [int(channel_indices[0]), 0]

            print(f"  Cellpose channels={channels_param}")

            masks, flows, styles = model.eval(
                img_data,
                batch_size=batch_size,
                channels=channels_param,
                diameter=diameter,
                flow_threshold=flow_threshold,
                cellprob_threshold=cellprob_threshold,
                normalize={"tile_norm_blocksize": tile_norm_blocksize},
            )

            polygons = mask_to_polygons(
                masks,
                min_area=min_object_area,
                min_points=min_polygon_points,
                simplify_every_nth=simplify_every_nth,
            )

            if polygons:
                out_txt.write_text(
                    "\n".join(polygon_to_line(poly) for poly in polygons),
                    encoding="utf-8",
                )
                print(f"    ✅ {len(polygons)} objects saved")
            else:
                out_txt.write_text("", encoding="utf-8")
                print("    ⚠️ No objects found")

        except Exception as e:
            print(f"    ❌ ERROR: {e}")
            traceback.print_exc()
            out_txt.write_text("", encoding="utf-8")

    print("Done.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <tile_dir> [channels] [config_yaml]")
        print("Example: python script.py ./tiles 0 ./config.yaml")
        sys.exit(1)

    tile_dir_arg = sys.argv[1]
    channels_arg = sys.argv[2] if len(sys.argv) >= 3 else "0"
    config_arg = sys.argv[3] if len(sys.argv) >= 4 else None

    main(tile_dir_arg, channels_arg, config_arg)
