"""
CC1 Vision - local CNN stain classifier training.

Trains the lightweight convolutional network that decides whether an OpenCV
candidate region is a real stain and, if so, which kind. Everything runs
locally on the CPU with TensorFlow/Keras. No pretrained weights are downloaded,
no cloud service is contacted and no API key is required.

Usage
-----

    python -m pip install -r requirements-ml.txt
    python dataset_generator.py            # build the labelled crops first
    python train_classifier.py --epochs 40

Outputs (all inside ``models/``)
--------------------------------

    stain_classifier.keras   trained Keras 3 model
    labels.json              class names in model output order
    model_config.json        input size, class order, display names, thresholds
    training_history.json    per-epoch accuracy and loss
    metrics.json             test accuracy, precision, recall, F1, confusion matrix

The architecture is deliberately small (roughly 0.2M parameters). The task is
six visually distinct classes of 96x96 crops, so a compact network trains in a
few minutes on a CPU and runs fast enough to classify every candidate region
interactively.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from dataset_generator import CLASS_NAMES, CROP_SIZE, DATASET_DIR, STAIN_CLASSES

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"

#: Folder-name -> the label shown in the dashboard and used by the cleaning
#: engine. ``clean`` has no display name because such regions are rejected.
DISPLAY_NAMES: Dict[str, str] = {
    "clean": "Clean Floor",
    "mud_dirt": "Mud / Dirt",
    "liquid_spill": "Liquid Spill",
    "grease_oil": "Grease / Oil",
    "dark_stain": "Dark Stain",
    "scattered_dirt": "Scattered Dirt",
}

#: A candidate is only accepted as a stain when the winning stain class scores
#: at least this probability. Raised from 0.55 to 0.60 after 0.55 let a JPEG
#: compression artifact through on the single-stain regression image; the
#: dashboard keeps it adjustable per run.
DEFAULT_MIN_STAIN_PROBABILITY: float = 0.60


def require_tensorflow():
    """Import TensorFlow with a clear message if it is missing."""
    try:
        import tensorflow as tf
    except ImportError:
        print(
            "TensorFlow is not installed.\n"
            "Install the optional ML requirements first:\n"
            "    python -m pip install -r requirements-ml.txt",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return tf


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------


def load_splits(tf, dataset_dir: Path, batch_size: int, seed: int):
    """Load train/val/test image folders with a fixed class order."""
    missing = [
        split for split in ("train", "val", "test")
        if not (dataset_dir / split).is_dir()
    ]
    if missing:
        raise SystemExit(
            f"Dataset split(s) {missing} not found under {dataset_dir}.\n"
            "Generate the dataset first:  python dataset_generator.py"
        )

    def load(split: str, shuffle: bool):
        return tf.keras.utils.image_dataset_from_directory(
            dataset_dir / split,
            labels="inferred",
            label_mode="int",
            class_names=list(CLASS_NAMES),       # fixes the output order
            color_mode="rgb",
            batch_size=batch_size,
            image_size=(CROP_SIZE, CROP_SIZE),
            shuffle=shuffle,
            seed=seed,
        )

    return load("train", True), load("val", False), load("test", False)


def build_augmentation(tf):
    """Augmentation pipeline applied to the training split only.

    Floor photographs have no canonical orientation and no canonical exposure,
    so full rotation, flips and photometric jitter are all label-preserving.
    """
    layers = tf.keras.layers
    return tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.5, fill_mode="reflect"),
            layers.RandomTranslation(0.10, 0.10, fill_mode="reflect"),
            layers.RandomZoom(0.15, 0.15, fill_mode="reflect"),
            layers.RandomContrast(0.20),
            layers.RandomBrightness(0.15, value_range=(0.0, 255.0)),
        ],
        name="augmentation",
    )


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------


def build_model(tf, num_classes: int, dropout: float = 0.35):
    """A small VGG-style CNN: four conv blocks then a global-pooled head."""
    layers = tf.keras.layers

    def block(x, filters: int):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        return layers.MaxPooling2D()(x)

    inputs = layers.Input(shape=(CROP_SIZE, CROP_SIZE, 3), name="crop")
    x = layers.Rescaling(1.0 / 255.0)(inputs)
    x = block(x, 32)
    x = block(x, 64)
    x = block(x, 96)
    x = block(x, 128)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="stain_class")(x)

    return tf.keras.Model(inputs, outputs, name="cc1_stain_cnn")


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------


def evaluate(tf, model, test_ds) -> Dict[str, Any]:
    """Full test-set report: accuracy, precision, recall, F1, confusion matrix."""
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        precision_recall_fscore_support,
    )

    y_true: List[int] = []
    y_prob: List[np.ndarray] = []
    for batch_images, batch_labels in test_ds:
        y_prob.append(model.predict(batch_images, verbose=0))
        y_true.extend(batch_labels.numpy().tolist())

    probabilities = np.concatenate(y_prob, axis=0)
    y_pred = probabilities.argmax(axis=1)
    y_true_array = np.array(y_true, dtype=int)

    accuracy = float(accuracy_score(y_true_array, y_pred))
    macro = precision_recall_fscore_support(y_true_array, y_pred, average="macro",
                                            zero_division=0)
    weighted = precision_recall_fscore_support(y_true_array, y_pred, average="weighted",
                                               zero_division=0)
    report = classification_report(
        y_true_array, y_pred, target_names=list(CLASS_NAMES),
        output_dict=True, zero_division=0,
    )
    matrix = confusion_matrix(y_true_array, y_pred,
                             labels=list(range(len(CLASS_NAMES)))).tolist()

    # A separate, more operationally meaningful score: did the network get the
    # stain / not-stain decision right, regardless of which stain type it chose?
    clean_index = CLASS_NAMES.index("clean")
    binary_true = (y_true_array != clean_index).astype(int)
    binary_pred = (y_pred != clean_index).astype(int)
    binary = precision_recall_fscore_support(binary_true, binary_pred, average="binary",
                                             zero_division=0)

    return {
        "test_samples": int(len(y_true_array)),
        "test_accuracy": accuracy,
        "macro_precision": float(macro[0]),
        "macro_recall": float(macro[1]),
        "macro_f1": float(macro[2]),
        "weighted_precision": float(weighted[0]),
        "weighted_recall": float(weighted[1]),
        "weighted_f1": float(weighted[2]),
        "stain_vs_clean": {
            "precision": float(binary[0]),
            "recall": float(binary[1]),
            "f1": float(binary[2]),
            "accuracy": float(accuracy_score(binary_true, binary_pred)),
        },
        "per_class": {
            name: {
                "precision": float(report[name]["precision"]),
                "recall": float(report[name]["recall"]),
                "f1": float(report[name]["f1-score"]),
                "support": int(report[name]["support"]),
            }
            for name in CLASS_NAMES
        },
        "confusion_matrix": matrix,
        "class_names": list(CLASS_NAMES),
    }


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the CC1 Vision CNN stain classifier locally."
    )
    parser.add_argument("--dataset", type=str, default=str(DATASET_DIR))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    tf = require_tensorflow()
    tf.keras.utils.set_random_seed(int(args.seed))

    print("=" * 66)
    print(" CC1 Vision - training the CNN stain classifier")
    print("=" * 66)

    dataset_dir = Path(args.dataset)
    train_ds, val_ds, test_ds = load_splits(tf, dataset_dir, args.batch_size, args.seed)
    print(f"classes: {list(CLASS_NAMES)}")

    augmentation = build_augmentation(tf)
    autotune = tf.data.AUTOTUNE
    train_prepared = (
        train_ds
        .map(lambda x, y: (augmentation(x, training=True), y), num_parallel_calls=autotune)
        .prefetch(autotune)
    )
    val_prepared = val_ds.cache().prefetch(autotune)

    model = build_model(tf, num_classes=len(CLASS_NAMES))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=float(args.learning_rate)),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "stain_classifier.keras"

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=int(args.patience),
            restore_best_weights=True, mode="max", verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_path), monitor="val_accuracy",
            save_best_only=True, mode="max", verbose=0,
        ),
    ]

    history = model.fit(
        train_prepared,
        validation_data=val_prepared,
        epochs=int(args.epochs),
        callbacks=callbacks,
        verbose=2,
    )

    print("\nEvaluating on the held-out test split...")
    metrics = evaluate(tf, model, test_ds)

    # The saved checkpoint is the best-validation model; the in-memory model has
    # those weights restored too, so re-saving keeps the two consistent.
    model.save(model_path)

    history_data = {key: [float(v) for v in values] for key, values in history.history.items()}
    history_data["epochs_run"] = len(history.history.get("loss", []))

    config: Dict[str, Any] = {
        "model_file": model_path.name,
        "input_size": int(CROP_SIZE),
        "color_order": "RGB",
        "classes": list(CLASS_NAMES),
        "stain_classes": list(STAIN_CLASSES),
        "display_names": DISPLAY_NAMES,
        "min_stain_probability": DEFAULT_MIN_STAIN_PROBABILITY,
        "architecture": "CC1 lightweight CNN (4 conv blocks, GAP head)",
        "parameters": int(model.count_params()),
        "framework": f"TensorFlow {tf.__version__} / Keras {tf.keras.__version__}",
        "trained_epochs": history_data["epochs_run"],
        "test_accuracy": metrics["test_accuracy"],
        "dataset": str(dataset_dir.name),
    }

    for name, payload in (
        ("labels.json", list(CLASS_NAMES)),
        ("model_config.json", config),
        ("training_history.json", history_data),
        ("metrics.json", metrics),
    ):
        with open(MODELS_DIR / name, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    print("=" * 66)
    print(f" test accuracy      : {metrics['test_accuracy'] * 100:.2f}%")
    print(f" macro precision    : {metrics['macro_precision'] * 100:.2f}%")
    print(f" macro recall       : {metrics['macro_recall'] * 100:.2f}%")
    print(f" macro F1           : {metrics['macro_f1'] * 100:.2f}%")
    print(f" stain vs clean F1  : {metrics['stain_vs_clean']['f1'] * 100:.2f}%")
    print(" per class F1       :")
    for name in CLASS_NAMES:
        print(f"    {name:<16} {metrics['per_class'][name]['f1'] * 100:6.2f}%"
              f"  (n={metrics['per_class'][name]['support']})")
    print("=" * 66)
    print(f" saved: {model_path}")
    print(" run the dashboard with:  python -m streamlit run app.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
