"""
CC1 Vision - CNN stain classifier runtime.

Thin, dependency-tolerant wrapper around the locally trained Keras model in
``models/stain_classifier.keras``. This module is the only place in the project
that touches TensorFlow at run time, and it never imports it at module import
time, so the dashboard still starts (in heuristic fallback mode) on a machine
where TensorFlow is not installed.

Everything is local: the model file was produced by ``train_classifier.py`` on
this machine from crops made by ``dataset_generator.py``. No network call, no
API key, no cloud inference.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from utils import MODELS_DIR

#: Fallback class order, only used when ``models/labels.json`` is unreadable.
FALLBACK_CLASSES: Tuple[str, ...] = (
    "clean", "mud_dirt", "liquid_spill", "grease_oil", "dark_stain", "scattered_dirt",
)

#: Fallback folder-name -> dashboard label mapping.
FALLBACK_DISPLAY_NAMES: Dict[str, str] = {
    "clean": "Clean Floor",
    "mud_dirt": "Mud / Dirt",
    "liquid_spill": "Liquid Spill",
    "grease_oil": "Grease / Oil",
    "dark_stain": "Dark Stain",
    "scattered_dirt": "Scattered Dirt",
}


@dataclass
class Prediction:
    """One CNN verdict on one candidate region crop."""

    class_name: str                     # dataset folder name, e.g. "mud_dirt"
    display_name: str                   # dashboard label, e.g. "Mud / Dirt"
    confidence: float                   # softmax probability of ``class_name``
    is_stain: bool                      # False when the winner is "clean"
    clean_probability: float            # probability assigned to "clean"
    probabilities: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialisable view for the JSON report."""
        return {
            "class": self.class_name,
            "label": self.display_name,
            "confidence": round(float(self.confidence), 4),
            "is_stain": bool(self.is_stain),
            "clean_probability": round(float(self.clean_probability), 4),
            "probabilities": {k: round(float(v), 4) for k, v in self.probabilities.items()},
        }


# --------------------------------------------------------------------------
# Crop extraction
# --------------------------------------------------------------------------


def extract_crop(
    image_bgr: np.ndarray,
    bbox: Tuple[int, int, int, int],
    out_size: int,
    context: float = 1.6,
) -> Optional[np.ndarray]:
    """Cut a square, context-padded crop around a candidate region.

    The crop is deliberately larger than the region itself: the CNN needs to
    see some surrounding floor to judge whether the region really differs from
    it. Out-of-frame areas are filled by reflection, which is exactly how the
    training crops were built, so regions near an image edge are not fed a
    black border the network has never seen.
    """
    if image_bgr is None or image_bgr.size == 0:
        return None

    height, width = image_bgr.shape[:2]
    x, y, w, h = bbox
    centre_x = x + w / 2.0
    centre_y = y + h / 2.0
    side = max(float(max(w, h)) * float(context), 12.0)
    half = side / 2.0

    x0, y0 = int(round(centre_x - half)), int(round(centre_y - half))
    x1, y1 = int(round(centre_x + half)), int(round(centre_y + half))

    pad_left, pad_top = max(0, -x0), max(0, -y0)
    pad_right, pad_bottom = max(0, x1 - width), max(0, y1 - height)

    sub = image_bgr[max(0, y0):min(height, y1), max(0, x0):min(width, x1)]
    if sub.size == 0:
        return None

    if pad_left or pad_top or pad_right or pad_bottom:
        # BORDER_REFLECT101 needs the pad to be smaller than the source side.
        pad_left = min(pad_left, sub.shape[1] - 1)
        pad_right = min(pad_right, sub.shape[1] - 1)
        pad_top = min(pad_top, sub.shape[0] - 1)
        pad_bottom = min(pad_bottom, sub.shape[0] - 1)
        sub = cv2.copyMakeBorder(sub, pad_top, pad_bottom, pad_left, pad_right,
                                 cv2.BORDER_REFLECT101)

    interpolation = cv2.INTER_AREA if sub.shape[0] > out_size else cv2.INTER_LINEAR
    return cv2.resize(sub, (out_size, out_size), interpolation=interpolation)


# --------------------------------------------------------------------------
# Classifier
# --------------------------------------------------------------------------


class StainClassifier:
    """Loads and runs the trained CNN, degrading gracefully when it cannot.

    Every failure path (no model file, no TensorFlow, corrupt weights) leaves
    :attr:`is_ready` False and records a human readable reason, so the caller
    can fall back to the classical heuristic labels instead of crashing.
    """

    def __init__(self, models_dir: Path = MODELS_DIR) -> None:
        self.models_dir = Path(models_dir)
        self.model_path = self.models_dir / "stain_classifier.keras"
        self.labels_path = self.models_dir / "labels.json"
        self.config_path = self.models_dir / "model_config.json"
        self.metrics_path = self.models_dir / "metrics.json"
        self.history_path = self.models_dir / "training_history.json"

        self._model: Any = None
        self._lock = threading.Lock()
        self._loaded = False
        self._error: Optional[str] = None

        self._config: Dict[str, Any] = self._read_json(self.config_path) or {}
        labels = self._read_json(self.labels_path)
        self.classes: List[str] = list(
            labels if isinstance(labels, list) and labels
            else self._config.get("classes") or FALLBACK_CLASSES
        )
        self.display_names: Dict[str, str] = dict(
            self._config.get("display_names") or FALLBACK_DISPLAY_NAMES
        )
        self.input_size: int = int(self._config.get("input_size", 96))
        self.min_stain_probability: float = float(
            self._config.get("min_stain_probability", 0.60)
        )
        self.clean_class: str = "clean" if "clean" in self.classes else self.classes[0]

    # -- small helpers ----------------------------------------------------

    @staticmethod
    def _read_json(path: Path) -> Any:
        """Read a JSON file, returning ``None`` when it is missing or broken."""
        try:
            with open(path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except Exception:
            return None

    def label_for(self, class_name: str) -> str:
        """Dashboard label for a dataset class name."""
        return self.display_names.get(class_name, class_name.replace("_", " ").title())

    # -- availability -----------------------------------------------------

    def model_exists(self) -> bool:
        """True when a trained model file is present on disk."""
        return self.model_path.is_file()

    def load(self) -> bool:
        """Load the Keras model once. Returns True when it is usable."""
        if self._loaded:
            return self._model is not None

        with self._lock:
            if self._loaded:
                return self._model is not None
            self._loaded = True

            if not self.model_exists():
                self._error = "No trained model found in models/."
                return False
            try:
                import tensorflow as tf
            except Exception:
                self._error = "TensorFlow is not installed (see requirements-ml.txt)."
                return False
            try:
                self._model = tf.keras.models.load_model(self.model_path)
                # One warm-up pass so the first real prediction is not slow.
                blank = np.zeros((1, self.input_size, self.input_size, 3), dtype=np.float32)
                self._model.predict(blank, verbose=0)
            except Exception as exc:
                self._model = None
                self._error = f"Model could not be loaded: {exc}"
                return False
            return True

    @property
    def is_ready(self) -> bool:
        """True when the CNN is loaded and inference can run."""
        return self.load()

    @property
    def error(self) -> Optional[str]:
        """Why the model is unavailable, or ``None`` when it is fine."""
        return self._error

    @property
    def status(self) -> str:
        """Short sidebar status string."""
        if self.is_ready:
            return "Loaded"
        return self._error or "Not available"

    @property
    def model_name(self) -> str:
        """File name of the model, whether or not it loaded."""
        return self.model_path.name

    def framework_version(self) -> str:
        """Reported training framework, or the live one when loaded."""
        return str(self._config.get("framework", "TensorFlow / Keras"))

    def metrics(self) -> Optional[Dict[str, Any]]:
        """Saved test-set metrics, or ``None`` when the model was not trained."""
        return self._read_json(self.metrics_path)

    def history(self) -> Optional[Dict[str, Any]]:
        """Saved per-epoch training history."""
        return self._read_json(self.history_path)

    def config(self) -> Dict[str, Any]:
        """Saved model configuration."""
        return dict(self._config)

    # -- inference --------------------------------------------------------

    def predict_batch(self, crops: Sequence[np.ndarray]) -> List[Optional[Prediction]]:
        """Classify a list of BGR crops in one batched forward pass.

        Returns one entry per input crop; an entry is ``None`` when that crop
        was unusable or when the model is unavailable.
        """
        results: List[Optional[Prediction]] = [None] * len(crops)
        if not crops or not self.is_ready:
            return results

        usable: List[int] = []
        batch: List[np.ndarray] = []
        for index, crop in enumerate(crops):
            if crop is None or crop.size == 0 or crop.ndim != 3:
                continue
            if crop.shape[0] != self.input_size or crop.shape[1] != self.input_size:
                crop = cv2.resize(crop, (self.input_size, self.input_size),
                                  interpolation=cv2.INTER_AREA)
            # The model was trained on RGB images loaded by Keras.
            batch.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            usable.append(index)

        if not batch:
            return results

        try:
            tensor = np.asarray(batch, dtype=np.float32)
            probabilities = self._model.predict(tensor, verbose=0)
        except Exception as exc:  # pragma: no cover - defensive
            self._error = f"Prediction failed: {exc}"
            return results

        clean_index = self.classes.index(self.clean_class)
        for slot, row in zip(usable, probabilities):
            row = np.asarray(row, dtype=np.float64).ravel()
            if row.size != len(self.classes):
                continue
            winner = int(row.argmax())
            class_name = self.classes[winner]
            results[slot] = Prediction(
                class_name=class_name,
                display_name=self.label_for(class_name),
                confidence=float(row[winner]),
                is_stain=bool(winner != clean_index),
                clean_probability=float(row[clean_index]),
                probabilities={name: float(value) for name, value in zip(self.classes, row)},
            )
        return results

    def predict(self, crop: np.ndarray) -> Optional[Prediction]:
        """Classify a single BGR crop."""
        return self.predict_batch([crop])[0]


# --------------------------------------------------------------------------
# Process-wide singleton
# --------------------------------------------------------------------------

_SHARED: Optional[StainClassifier] = None
_SHARED_LOCK = threading.Lock()


def get_classifier(models_dir: Path = MODELS_DIR) -> StainClassifier:
    """Return the shared classifier, loading the model at most once per process.

    Streamlit reruns the script on every interaction, so caching the loaded
    Keras model here keeps the dashboard responsive.
    """
    global _SHARED
    if _SHARED is None:
        with _SHARED_LOCK:
            if _SHARED is None:
                _SHARED = StainClassifier(models_dir)
    return _SHARED
