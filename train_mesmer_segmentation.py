# -*- coding: utf-8 -*-
"""
Fine-tune old-style DeepCell NuclearSegmentation on single-channel nuclear TIFF data.

Folder structure
----------------
data/
├── images/
│   ├── sample_001.tif
│   ├── sample_002.tif
│   └── ...
└── masks/
    ├── sample_001.tif
    ├── sample_002.tif
    └── ...

Requirements
------------
pip install numpy tifffile scikit-image scikit-learn tensorflow deepcell-toolbox deepcell-tf

Important
---------
- masks must be INSTANCE masks, not binary masks
- input images are grayscale nuclear images
- this version is written for older DeepCell app/model style
- it automatically adapts to 2-head or 3-head NuclearSegmentation models
- uses save_weights / load_weights to avoid custom-layer deserialization issues
"""

import math
import random
from pathlib import Path

import numpy as np
import tifffile as tiff
import tensorflow as tf

from sklearn.model_selection import train_test_split

from deepcell.applications import NuclearSegmentation
from deepcell.image_generators import CroppingDataGenerator
from deepcell.losses import weighted_categorical_crossentropy
from deepcell.utils.train_utils import rate_scheduler


# =========================
# User settings
# =========================
DATA_DIR = Path("./training_dataset")
IMAGE_DIR = DATA_DIR / "images"
MASK_DIR = DATA_DIR / "masks"

MODEL_DIR = Path("./melanoma")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
VAL_FRAC = 0.2
BATCH_SIZE = 4
CROP_SIZE = 256

EPOCHS_STAGE1 = 16
EPOCHS_STAGE2 = 16
LR_STAGE1 = 1e-5
LR_STAGE2 = 2e-6

# inference nominal scale
IMAGE_MPP = 0.25

STAGE1_BEST_WEIGHTS = MODEL_DIR / "stage1_best.weights.h5"
STAGE2_BEST_WEIGHTS = MODEL_DIR / "nuclear_finetuned_best.weights.h5"
FINAL_WEIGHTS = MODEL_DIR / "nuclear_finetuned_final.weights.h5"


# =========================
# Reproducibility
# =========================
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# =========================
# I/O helpers
# =========================
def pad_to_shape(arr, target_shape, constant_values=0):
    """
    Pad 2D or 3D array to target spatial shape (H, W).
    Padding is added to bottom/right only.
    """
    arr = np.asarray(arr)
    th, tw = target_shape
    h, w = arr.shape[:2]

    if h > th or w > tw:
        raise ValueError(f"Array shape {arr.shape} is larger than target {target_shape}")

    pad_h = th - h
    pad_w = tw - w

    if arr.ndim == 2:
        pad_width = ((0, pad_h), (0, pad_w))
    elif arr.ndim == 3:
        pad_width = ((0, pad_h), (0, pad_w), (0, 0))
    else:
        raise ValueError(f"pad_to_shape expects 2D or 3D array, got {arr.ndim}D")

    return np.pad(arr, pad_width, mode="constant", constant_values=constant_values)


def read_tiff(path):
    arr = tiff.imread(str(path))
    arr = np.asarray(arr)
    arr = np.squeeze(arr)

    if arr.ndim != 2:
        raise ValueError(f"{path} must be a 2D TIFF after squeeze, got shape {arr.shape}")

    return arr


def normalize_single_image(img):
    img = img.astype(np.float32)

    vmin = np.min(img)
    vmax = np.max(img)
    if vmax <= vmin:
        return np.zeros_like(img, dtype=np.float32)

    lo, hi = np.percentile(img, [1, 99])
    img = np.clip(img, lo, hi)

    if hi > lo:
        img = (img - lo) / (hi - lo)
    else:
        img = np.zeros_like(img, dtype=np.float32)

    return img.astype(np.float32)


def ensure_instance_mask(mask):
    """
    Accept instance mask directly.
    Does not relabel binary masks automatically.
    """
    mask = np.asarray(mask)

    if mask.ndim != 2:
        raise ValueError(f"Mask must be 2D, got {mask.shape}")

    if np.issubdtype(mask.dtype, np.floating):
        if not np.allclose(mask, np.round(mask)):
            raise ValueError("Mask looks non-integer. Need instance labels, not probability map.")
        mask = np.round(mask).astype(np.int32)
    else:
        mask = mask.astype(np.int32)

    return mask


def build_input_image(nuclear_img):
    """
    Build 1-channel input for NuclearSegmentation.
    Output shape: (H, W, 1)
    """
    ch0 = normalize_single_image(nuclear_img)
    x = ch0[..., np.newaxis].astype(np.float32)
    return x


# =========================
# Dataset loading
# =========================
def collect_pairs(image_dir, mask_dir):
    image_paths = sorted(list(image_dir.glob("*.tif")) + list(image_dir.glob("*.tiff")))
    if not image_paths:
        raise FileNotFoundError(f"No TIFF images found in {image_dir}")

    pairs = []
    missing = []

    for img_path in image_paths:
        mask_path = mask_dir / img_path.name
        if mask_path.exists():
            pairs.append((img_path, mask_path))
        else:
            missing.append(img_path.name)

    if missing:
        print(f"Warning: {len(missing)} images have no matching mask and will be skipped.")
        for name in missing[:10]:
            print("  missing mask for:", name)
        if len(missing) > 10:
            print("  ...")

    if not pairs:
        raise FileNotFoundError("No matched image/mask pairs found.")

    return pairs


def load_dataset(pairs):
    X = []
    y_nuc = []
    spatial_shapes = []

    for img_path, mask_path in pairs:
        img = read_tiff(img_path)
        mask = ensure_instance_mask(read_tiff(mask_path))

        if img.shape != mask.shape:
            raise ValueError(
                f"Shape mismatch for {img_path.name}: image {img.shape}, mask {mask.shape}"
            )

        x = build_input_image(img)
        y = mask[..., np.newaxis].astype(np.int32)

        X.append(x)
        y_nuc.append(y)
        spatial_shapes.append(img.shape)

    max_h = max(s[0] for s in spatial_shapes)
    max_w = max(s[1] for s in spatial_shapes)

    X = [pad_to_shape(x, (max_h, max_w), constant_values=0) for x in X]
    y_nuc = [pad_to_shape(y, (max_h, max_w), constant_values=0) for y in y_nuc]

    X = np.stack(X, axis=0).astype(np.float32)
    y_nuc = np.stack(y_nuc, axis=0).astype(np.int32)

    return X, y_nuc


# =========================
# Model/head inspection
# =========================
def _shape_last_dim(shape_obj):
    if hasattr(shape_obj, "as_list"):
        shape_obj = shape_obj.as_list()
    return int(shape_obj[-1])


def inspect_semantic_heads(model):
    output_names = list(model.output_names)
    output_shapes = model.output_shape

    if not isinstance(output_shapes, (list, tuple)):
        output_shapes = [output_shapes]

    heads = []
    for name, shape in zip(output_names, output_shapes):
        n_classes = _shape_last_dim(shape)
        heads.append({"name": name, "n_classes": n_classes})

    return heads


def choose_transforms_for_heads(heads):
    n_heads = len(heads)

    if n_heads == 2:
        return ["inner-distance", "outer-distance"]
    elif n_heads == 3:
        return ["inner-distance", "outer-distance", "fgbg"]
    else:
        raise RuntimeError(
            f"Unsupported number of model outputs: {n_heads}. "
            f"Output names: {[h['name'] for h in heads]}"
        )


def print_model_head_info(model, title="Model"):
    heads = inspect_semantic_heads(model)
    print(f"{title} input shape: {model.input_shape}")
    print(f"{title} output names: {model.output_names}")
    print(f"{title} output shapes: {model.output_shape}")
    print(f"{title} semantic heads:")
    for i, h in enumerate(heads):
        print(f"  [{i}] name={h['name']}, n_classes={h['n_classes']}")


# =========================
# Training target generator
# =========================
class NuclearSequence(tf.keras.utils.Sequence):
    """
    Produces old-style NuclearSegmentation-compatible outputs.

    Supports:
      - 2-head model: inner-distance, outer-distance
      - 3-head model: inner-distance, outer-distance, fgbg
    """
    def __init__(
        self,
        X,
        y_nuc,
        model,
        batch_size=4,
        crop_size=256,
        training=True,
        seed=42,
    ):
        self.X = X
        self.y_nuc = y_nuc
        self.batch_size = batch_size
        self.crop_size = crop_size
        self.training = training
        self.seed = seed

        self.heads = inspect_semantic_heads(model)
        self.transforms = choose_transforms_for_heads(self.heads)

        if training:
            self.datagen = CroppingDataGenerator(
                rotation_range=180,
                zoom_range=(0.8, 1.2),
                horizontal_flip=True,
                vertical_flip=True,
                crop_size=(crop_size, crop_size),
            )
        else:
            self.datagen = CroppingDataGenerator(
                crop_size=(crop_size, crop_size)
            )

        self.transforms_kwargs = {
            "outer-distance": {"erosion_width": 1},
            "inner-distance": {
                "alpha": "auto",
                "beta": 1,
                "erosion_width": 0,
            },
        }

        self.gen_nuc = self.datagen.flow(
            {"X": self.X, "y": self.y_nuc},
            seed=seed,
            batch_size=batch_size,
            min_objects=1,
            transforms=self.transforms,
            transforms_kwargs=self.transforms_kwargs,
        )

        self.n = X.shape[0]

    def __len__(self):
        return max(1, math.ceil(self.n / self.batch_size))

    def __getitem__(self, idx):
        x, y_n = next(self.gen_nuc)
        x = x.astype(np.float32)

        outputs = [arr.astype(np.float32) for arr in y_n]

        if len(outputs) != len(self.heads):
            raise RuntimeError(
                f"Generator/model output mismatch: generator produced {len(outputs)} targets "
                f"for transforms {self.transforms}, but model has {len(self.heads)} outputs "
                f"{[h['name'] for h in self.heads]}"
            )

        return x, outputs


# =========================
# Loss helpers
# =========================
def semantic_loss(n_classes):
    def _loss(y_true, y_pred):
        if n_classes > 1:
            return 0.01 * weighted_categorical_crossentropy(
                y_true, y_pred, n_classes=n_classes
            )
        return tf.keras.losses.MSE(y_true, y_pred)
    return _loss


def build_loss_dict(model):
    heads = inspect_semantic_heads(model)

    loss = {}
    for head in heads:
        loss[head["name"]] = semantic_loss(head["n_classes"])

    if len(loss) == 0:
        raise RuntimeError("No model outputs found for loss construction.")

    return loss


def set_trainable_backbone(model, freeze_ratio=0.85):
    n_layers = len(model.layers)
    cutoff = int(n_layers * freeze_ratio)

    for i, layer in enumerate(model.layers):
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = i >= cutoff


def compile_model(model, lr):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=0.001),
        loss=build_loss_dict(model),
    )


# =========================
# Main
# =========================
def main():
    pairs = collect_pairs(IMAGE_DIR, MASK_DIR)
    print(f"Matched pairs: {len(pairs)}")

    if len(pairs) < 2:
        raise ValueError("Need at least 2 matched image/mask pairs.")

    train_pairs, val_pairs = train_test_split(
        pairs, test_size=VAL_FRAC, random_state=SEED, shuffle=True
    )

    X_train, y_train_nuc = load_dataset(train_pairs)
    X_val, y_val_nuc = load_dataset(val_pairs)

    X_train = np.clip(X_train, 0.0, 1.0).astype(np.float32)
    X_val = np.clip(X_val, 0.0, 1.0).astype(np.float32)

    print("Train X:", X_train.shape)
    print("Train nuclear masks:", y_train_nuc.shape)
    print("Val X:", X_val.shape)
    print("Val nuclear masks:", y_val_nuc.shape)

    # pretrained old-style app/model
    app = NuclearSegmentation()
    model = app.model

    print_model_head_info(model, title="NuclearSegmentation")
    print("NuclearSegmentation model_mpp:", getattr(app, "model_mpp", "unknown"))

    train_seq = NuclearSequence(
        X_train,
        y_train_nuc,
        model=model,
        batch_size=BATCH_SIZE,
        crop_size=CROP_SIZE,
        training=True,
        seed=SEED,
    )

    val_seq = NuclearSequence(
        X_val,
        y_val_nuc,
        model=model,
        batch_size=BATCH_SIZE,
        crop_size=CROP_SIZE,
        training=False,
        seed=SEED,
    )

    print("Training transforms:", train_seq.transforms)

    # sanity check one batch
    x0, y0 = train_seq[0]
    print("Example batch X:", x0.shape)
    print("Example batch number of targets:", len(y0))
    for i, arr in enumerate(y0):
        print(f"  target[{i}] shape: {arr.shape}")

    # ---------- stage 1 ----------
    set_trainable_backbone(model, freeze_ratio=0.85)
    compile_model(model, LR_STAGE1)

    callbacks_stage1 = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(STAGE1_BEST_WEIGHTS),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(MODEL_DIR / "train_log.csv"), append=False),
        tf.keras.callbacks.LearningRateScheduler(
            rate_scheduler(lr=LR_STAGE1, decay=0.99)
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=3,
            verbose=1,
            min_delta=1e-4,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=6,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    model.fit(
        train_seq,
        validation_data=val_seq,
        epochs=EPOCHS_STAGE1,
        callbacks=callbacks_stage1,
        verbose=1,
    )

    # optionally persist stage1 final in-memory weights too
    model.save_weights(str(MODEL_DIR / "stage1_last.weights.h5"))

    # ---------- stage 2 ----------
    set_trainable_backbone(model, freeze_ratio=0.3)
    compile_model(model, LR_STAGE2)

    callbacks_stage2 = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(STAGE2_BEST_WEIGHTS),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(MODEL_DIR / "train_log.csv"), append=True),
        tf.keras.callbacks.LearningRateScheduler(
            rate_scheduler(lr=LR_STAGE2, decay=0.995)
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=4,
            verbose=1,
            min_delta=1e-4,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    model.fit(
        train_seq,
        validation_data=val_seq,
        epochs=EPOCHS_STAGE2,
        callbacks=callbacks_stage2,
        verbose=1,
    )

    # save final weights from current in-memory model
    model.save_weights(str(FINAL_WEIGHTS))
    print(f"Saved final weights to: {FINAL_WEIGHTS}")

    # =========================================================
    # Inference route A: use current in-memory model directly
    # =========================================================
    preds_mem = app.predict(
        X_val[:2],
        image_mpp=IMAGE_MPP,
    )
    print("Prediction shape from in-memory model:", preds_mem.shape)

    # =========================================================
    # Inference route B: rebuild fresh app and load BEST weights
    # =========================================================
    fresh_app = NuclearSegmentation()
    fresh_model = fresh_app.model

    print_model_head_info(fresh_model, title="Fresh NuclearSegmentation before weight load")

    # load best stage2 weights into freshly constructed architecture
    fresh_model.load_weights(str(STAGE2_BEST_WEIGHTS))
    print(f"Loaded best weights from: {STAGE2_BEST_WEIGHTS}")

    preds_best = fresh_app.predict(
        X_val[:2],
        image_mpp=IMAGE_MPP,
    )
    print("Prediction shape from reloaded best weights:", preds_best.shape)


if __name__ == "__main__":
    main()