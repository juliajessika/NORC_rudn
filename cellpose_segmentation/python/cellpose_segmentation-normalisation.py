# -*- coding: utf-8 -*-
"""
Cellpose segmentation for QuPath-exported tiles.
"""

from pathlib import Path
import sys
import numpy as np
import tifffile as tiff
from skimage.measure import find_contours
import warnings
warnings.filterwarnings('ignore')

from cellpose import models, core, io

# =========================
# SETTINGS
# =========================
MODEL_TYPE = 'nuclei'
DIAMETER = None  # Let Cellpose estimate automatically (better for varying sizes)

FLOW_THRESHOLD = 0.4  # Same as working script
CELLPROB_THRESHOLD = 0.0  # Same as working script
TILE_NORM_BLOCKSIZE = 0  # Same as working script - 0 means global normalization

# Polygon filtering
MIN_OBJECT_AREA = 40
MIN_POLYGON_POINTS = 6
SIMPLIFY_EVERY_NTH = 2

# =========================
# IMAGE LOADING & CHANNEL SELECTION
# =========================
def load_and_select_channels(path, channel_indices):
    """
    Load image and select specified channels.
    Matches the successful script's approach.
    """
    im = tiff.imread(path)
    im = np.asarray(im)
    im = np.squeeze(im)
    
    print(f"  Original shape: {im.shape}")
    
    # IMPORTANT: Match the successful script's logic
    # For single-channel images (2D or with one channel)
    if im.ndim == 2:
        print(f"  Detected single-channel (grayscale) image")
        im = im[:, :, np.newaxis]  # Add channel dimension: (H, W) -> (H, W, 1)
        print(f"  Reshaped to: {im.shape}")
    
    # For 3D arrays, assume channels-last if last dimension is small
    elif im.ndim == 3:
        # If first dimension is small (<=4) and last dimension is large, it's likely channels-first
        if im.shape[0] <= 4 and im.shape[0] < im.shape[-1]:
            print(f"  Detected channels-first format, transposing...")
            im = np.transpose(im, (1, 2, 0))
            print(f"  New shape: {im.shape}")
    
    n_channels = im.shape[-1] if im.ndim == 3 else 1
    print(f"  Number of channels: {n_channels}")
    
    # For single-channel images
    if im.ndim == 2 or n_channels == 1:
        if im.ndim == 3 and im.shape[-1] == 1:
            print(f"  Single-channel image (shape: {im.shape})")
        else:
            print(f"  Single-channel image")
        return im.astype(np.float32)
    
    # For multi-channel images, select requested channels
    valid_indices = [i for i in channel_indices if i < n_channels]
    
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
def mask_to_polygons(label_mask,
                     min_area=MIN_OBJECT_AREA,
                     min_points=MIN_POLYGON_POINTS,
                     simplify_every_nth=SIMPLIFY_EVERY_NTH):
    """Convert integer label mask into a list of polygons."""
    from skimage import morphology
    
    polygons = []
    label_mask = np.asarray(label_mask)
    
    if label_mask.ndim != 2:
        raise ValueError(f"label_mask must be 2D, got shape {label_mask.shape}")
    
    # Remove small objects
    if min_area > 0:
        cleaned = morphology.remove_small_objects(label_mask, min_size=min_area)
    else:
        cleaned = label_mask
    
    if cleaned.max() == 0:
        return polygons
    
    # Find contours for each object
    for obj_id in range(1, cleaned.max() + 1):
        obj = cleaned == obj_id
        if obj.sum() < min_area:
            continue
        
        contours = find_contours(obj.astype(np.uint8), level=0.5)
        if not contours:
            continue
        
        # Choose longest contour
        contour = max(contours, key=len)
        
        if simplify_every_nth > 1:
            contour = contour[::simplify_every_nth]
        
        if len(contour) < min_points:
            continue
        
        # Convert from (y, x) to (x, y) and round to 1 decimal
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
def main(tile_dir, channels_str="0"):
    """Main segmentation function."""
    tile_dir = Path(tile_dir)
    if not tile_dir.exists():
        raise FileNotFoundError(f"Folder not found: {tile_dir}")
    
    # Parse channel indices
    channel_indices = [int(c.strip()) for c in channels_str.split(",")]
    print(f"Using channels: {channel_indices}")
    
    # Find tile files
    tif_files = sorted(tile_dir.glob("tile_x*_y*.tif"))
    if not tif_files:
        tif_files = sorted(tile_dir.glob("tile_x*_y*.tiff"))
    
    if not tif_files:
        print("No tile tif files found.")
        return
    
    print(f"Found {len(tif_files)} tiles")
    
    # Initialize Cellpose model
    print("Loading Cellpose model...")
    use_gpu = core.use_gpu()
    print(f"GPU available: {use_gpu}")
    
    # Use CellposeModel for more control (same as working script)
    model = models.CellposeModel(gpu=use_gpu, model_type=MODEL_TYPE)
    print(f"Model loaded: {MODEL_TYPE}")
    
    # Process each tile
    for i, tif_path in enumerate(tif_files, start=1):
        print(f"[{i}/{len(tif_files)}] Processing {tif_path.name}")
        
        out_txt = tif_path.with_suffix(".txt")
        
        try:
            # Load and prepare image
            img_data = load_and_select_channels(tif_path, channel_indices)
            
            # CRITICAL FIX: Don't normalize separately - let Cellpose handle it with tile_norm_blocksize
            # Just ensure it's float32
            if img_data.dtype != np.float32:
                img_data = img_data.astype(np.float32)
            
            # Determine channels parameter for Cellpose (MATCH WORKING SCRIPT)
            n_channels = img_data.shape[-1] if img_data.ndim == 3 else 1
            
            if n_channels == 1:
                # Single-channel: use [0,0] exactly like working script
                channels_param = [0, 0]
                print(f"  Single-channel mode: channels={channels_param}")
            else:
                # For multi-channel, first channel (index 0) is nuclear channel
                # Adjust based on your image: [nuclear_channel, cytoplasmic_channel]
                # If you have DAPI in channel 0: [0, 0] or [0, 1] for membrane
                channels_param = [channel_indices[0], 0]
                print(f"  Multi-channel mode: channels={channels_param}")
            
            # Run Cellpose segmentation with EXACT same parameters as working script
            masks, flows, styles = model.eval(
                img_data,
                batch_size=32,  # Added - same as working script
                channels=channels_param,
                flow_threshold=FLOW_THRESHOLD,
                cellprob_threshold=CELLPROB_THRESHOLD,
                normalize={"tile_norm_blocksize": TILE_NORM_BLOCKSIZE}  # CRITICAL: Use same normalization
            )
            
            # Convert mask to polygons
            polygons = mask_to_polygons(masks)
            
            # Save results
            if polygons:
                out_txt.write_text(
                    "\n".join(polygon_to_line(poly) for poly in polygons),
                    encoding="utf-8"
                )
                print(f"    ✅ {len(polygons)} objects saved")
            else:
                out_txt.write_text("", encoding="utf-8")
                print(f"    ⚠️ No objects found")
            
        except Exception as e:
            print(f"    ❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            out_txt.write_text("", encoding="utf-8")
    
    print("Done.")


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        tile_dir = sys.argv[1]
    else:
        print("Usage: python script.py <tile_dir> [channels]")
        print("Example: python script.py ./tiles 0")
        sys.exit(1)
    
    channels = sys.argv[2] if len(sys.argv) >= 3 else "0"
    
    main(tile_dir, channels)
