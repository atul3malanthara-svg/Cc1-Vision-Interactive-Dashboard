"""
CC1 Vision - utility helpers.

This module contains everything that is NOT detection logic:

* safe image loading / validation / decoding
* resizing helpers
* colour-space conversion helpers for display
* encoding images to PNG / JPEG bytes for download
* building JSON and CSV detection reports
* small formatting helpers used by the dashboard

The module deliberately has no Streamlit dependency so that it can be used
from plain Python scripts (see ``selftest.py`` and ``sample_generator.py``).
"""

from __future__ import annotations

import io
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

PROJECT_NAME: str = "CC1 Vision"
PROJECT_TITLE: str = (
    "Vision-Based Floor Stain Detection and Cleaning Decision Support "
    "for Autonomous Cleaning Robots"
)
PROJECT_VERSION: str = "1.0.0"

SUPPORTED_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp")

#: Uploads larger than this are rejected before decoding (protects memory).
MAX_IMAGE_BYTES: int = 40 * 1024 * 1024  # 40 MB

#: Images smaller than this in either dimension are not usable.
MIN_IMAGE_DIMENSION: int = 32

#: Absolute safety limit on the number of pixels we will ever decode.
MAX_IMAGE_PIXELS: int = 80_000_000  # ~80 MP

PROJECT_ROOT: Path = Path(__file__).resolve().parent
SAMPLES_DIR: Path = PROJECT_ROOT / "samples"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
MODELS_DIR: Path = PROJECT_ROOT / "models"


class ImageLoadError(Exception):
    """Raised when an uploaded file cannot be decoded into a usable image."""


# --------------------------------------------------------------------------
# Image loading and validation
# --------------------------------------------------------------------------


def load_image_from_bytes(data: bytes, max_bytes: int = MAX_IMAGE_BYTES) -> np.ndarray:
    """Decode raw file bytes into a 3-channel BGR ``uint8`` image.

    Parameters
    ----------
    data:
        Raw bytes of a JPG / JPEG / PNG / WEBP file.
    max_bytes:
        Reject the file before decoding if it is larger than this.

    Returns
    -------
    np.ndarray
        Image of shape ``(H, W, 3)``, dtype ``uint8``, channel order BGR.

    Raises
    ------
    ImageLoadError
        If the data is empty, too large, corrupted or not an image.
    """
    if data is None or len(data) == 0:
        raise ImageLoadError("The uploaded file is empty.")

    if len(data) > max_bytes:
        raise ImageLoadError(
            f"File is {len(data) / 1024 / 1024:.1f} MB. "
            f"The limit is {max_bytes / 1024 / 1024:.0f} MB."
        )

    image: Optional[np.ndarray] = None

    # Primary path: OpenCV decoder (fast, handles jpg/png/webp).
    try:
        buffer = np.frombuffer(data, dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
    except Exception:  # pragma: no cover - defensive
        image = None

    # Fallback path: Pillow (handles a few exotic encodings OpenCV refuses).
    if image is None:
        try:
            with Image.open(io.BytesIO(data)) as pil_image:
                pil_image.load()
                rgb = np.array(pil_image.convert("RGB"))
            image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except Exception as exc:
            raise ImageLoadError(
                "The file could not be decoded. It may be corrupted or not a "
                "supported image format (JPG, JPEG, PNG, WEBP)."
            ) from exc

    if image is None or getattr(image, "size", 0) == 0:
        raise ImageLoadError("The file could not be decoded into an image.")

    image = _normalise_channels(image)

    height, width = image.shape[:2]
    if height < MIN_IMAGE_DIMENSION or width < MIN_IMAGE_DIMENSION:
        raise ImageLoadError(
            f"Image is only {width}x{height} px. "
            f"Minimum supported size is {MIN_IMAGE_DIMENSION}x{MIN_IMAGE_DIMENSION} px."
        )

    if height * width > MAX_IMAGE_PIXELS:
        raise ImageLoadError(
            f"Image has {height * width / 1e6:.1f} megapixels, which is above the "
            f"{MAX_IMAGE_PIXELS / 1e6:.0f} MP safety limit."
        )

    return np.ascontiguousarray(image)


def load_image_from_path(path: os.PathLike | str) -> np.ndarray:
    """Load an image from disk using the same validation as uploads."""
    file_path = Path(path)
    if not file_path.exists():
        raise ImageLoadError(f"File not found: {file_path}")
    if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ImageLoadError(
            f"Unsupported extension '{file_path.suffix}'. "
            f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )
    return load_image_from_bytes(file_path.read_bytes())


def _normalise_channels(image: np.ndarray) -> np.ndarray:
    """Force any decoded array into 8-bit 3-channel BGR."""
    if image.dtype != np.uint8:
        # 16-bit PNGs and float TIFF-style data are rescaled to 8-bit.
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        image = image.astype(np.uint8)

    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    if image.ndim == 3:
        channels = image.shape[2]
        if channels == 1:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if channels == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        if channels == 3:
            return image

    raise ImageLoadError(f"Unsupported image shape: {image.shape}")


# --------------------------------------------------------------------------
# Geometry / conversion helpers
# --------------------------------------------------------------------------


def resize_max_dimension(image: np.ndarray, max_dimension: int) -> Tuple[np.ndarray, float]:
    """Downscale so the longest side is ``max_dimension`` px, keeping aspect ratio.

    Returns the (possibly unchanged) image and the scale factor that was
    applied (``1.0`` means no resize happened).
    """
    if max_dimension <= 0:
        return image, 1.0

    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_dimension:
        return image, 1.0

    scale = max_dimension / float(longest)
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    resized = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return resized, scale


def to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    """Convert a BGR image to RGB (Streamlit / Pillow expect RGB)."""
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def encode_png(image_bgr: np.ndarray) -> bytes:
    """Encode a BGR image as PNG bytes (used for download buttons)."""
    success, buffer = cv2.imencode(".png", image_bgr)
    if not success:
        raise ValueError("Failed to encode image as PNG.")
    return buffer.tobytes()


def encode_jpeg(image_bgr: np.ndarray, quality: int = 92) -> bytes:
    """Encode a BGR image as JPEG bytes."""
    quality = int(np.clip(quality, 10, 100))
    success, buffer = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not success:
        raise ValueError("Failed to encode image as JPEG.")
    return buffer.tobytes()


def blend_images(base_bgr: np.ndarray, overlay_bgr: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Alpha-blend two same-sized BGR images."""
    if base_bgr.shape != overlay_bgr.shape:
        overlay_bgr = cv2.resize(
            overlay_bgr, (base_bgr.shape[1], base_bgr.shape[0]), interpolation=cv2.INTER_NEAREST
        )
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return cv2.addWeighted(overlay_bgr, alpha, base_bgr, 1.0 - alpha, 0.0)


# --------------------------------------------------------------------------
# Small numeric / formatting helpers
# --------------------------------------------------------------------------


def clamp(value: float, low: float, high: float) -> float:
    """Clamp ``value`` into the inclusive range ``[low, high]``."""
    return float(max(low, min(high, value)))


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Division that never raises on a zero denominator."""
    if denominator == 0:
        return default
    return float(numerator) / float(denominator)


def format_percent(value: float, decimals: int = 1) -> str:
    """Format a 0..1 ratio as a percentage string."""
    return f"{value * 100:.{decimals}f}%"


def format_duration(seconds: float) -> str:
    """Format seconds as ``M min S s`` (or just seconds when short)."""
    seconds = max(0, int(round(seconds)))
    if seconds < 60:
        return f"{seconds} s"
    minutes, remainder = divmod(seconds, 60)
    return f"{minutes} min {remainder:02d} s"


def timestamp_slug() -> str:
    """Filesystem-safe timestamp, e.g. ``20260905_143012``."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def timestamp_iso() -> str:
    """Human/machine readable timestamp for reports."""
    return datetime.now().isoformat(timespec="seconds")


# --------------------------------------------------------------------------
# Reports
# --------------------------------------------------------------------------

#: Column order used by the on-screen detection table and the CSV report.
DETECTION_TABLE_COLUMNS: List[str] = [
    "ID",
    "Stain Type",
    "Confidence",
    "Severity",
    "Classifier",
    "Area (px)",
    "X",
    "Y",
    "Width",
    "Height",
    "Recommended Cleaning Action",
]


def build_detection_table(
    detections: Sequence[Any],
    plans_by_id: Optional[Dict[int, Any]] = None,
) -> pd.DataFrame:
    """Build the main detection table shown under the images.

    ``detections`` items must expose the attributes produced by
    :class:`detector.Detection`. ``plans_by_id`` maps a detection id to a
    :class:`cleaning_engine.CleaningPlan` so the recommended action can be
    joined in. Missing plans degrade gracefully to ``"-"``.
    """
    plans_by_id = plans_by_id or {}
    rows: List[Dict[str, Any]] = []

    for det in detections:
        plan = plans_by_id.get(det.id)
        rows.append(
            {
                "ID": f"{det.id:02d}",
                "Stain Type": det.stain_type,
                "Confidence": round(float(det.confidence), 3),
                "Severity": det.severity,
                "Classifier": (
                    "CNN" if getattr(det, "classifier", "heuristic") == "cnn"
                    else "Heuristic"
                ),
                "Area (px)": int(det.area_px),
                "X": int(det.x),
                "Y": int(det.y),
                "Width": int(det.width),
                "Height": int(det.height),
                "Recommended Cleaning Action": getattr(plan, "action", "-"),
            }
        )

    if not rows:
        return pd.DataFrame(columns=DETECTION_TABLE_COLUMNS)

    return pd.DataFrame(rows, columns=DETECTION_TABLE_COLUMNS)


def build_json_report(
    detections: Sequence[Any],
    plans: Sequence[Any],
    summary: Dict[str, Any],
    settings: Dict[str, Any],
    image_info: Dict[str, Any],
) -> Dict[str, Any]:
    """Assemble the full JSON detection report as a plain dictionary."""
    plans_by_id = {plan.detection_id: plan for plan in plans}

    detection_records: List[Dict[str, Any]] = []
    for det in detections:
        plan = plans_by_id.get(det.id)
        record: Dict[str, Any] = {
            "id": int(det.id),
            "type": det.stain_type,
            "confidence": round(float(det.confidence), 3),
            "severity": det.severity,
            "area_px": int(det.area_px),
            "area_ratio": round(float(det.area_ratio), 5),
            "bbox": {
                "x": int(det.x),
                "y": int(det.y),
                "width": int(det.width),
                "height": int(det.height),
            },
            "centroid": {"x": int(det.center_x), "y": int(det.center_y)},
            "recommended_action": getattr(plan, "action", "Standard clean pass"),
            "cleaning_intensity": getattr(plan, "intensity", "Medium"),
            "priority": getattr(plan, "priority", "MEDIUM"),
            "estimated_clean_seconds": int(getattr(plan, "estimated_seconds", 0)),
            "features": {
                "anomaly_score": round(float(det.anomaly_score), 3),
                "darkness_delta": round(float(det.darkness_delta), 2),
                "color_delta": round(float(det.color_delta), 2),
                "saturation_delta": round(float(det.saturation_delta), 2),
                "texture_delta": round(float(det.texture_delta), 2),
                "edge_strength": round(float(det.edge_strength), 3),
                "solidity": round(float(det.solidity), 3),
                "circularity": round(float(det.circularity), 3),
            },
            "classification": {
                "source": getattr(det, "classifier", "heuristic"),
                "cnn_class": getattr(det, "ml_class", ""),
                "cnn_confidence": round(float(getattr(det, "ml_confidence", 0.0)), 4),
                "clean_probability": round(float(getattr(det, "clean_probability", 0.0)), 4),
                "probabilities": {
                    key: round(float(value), 4)
                    for key, value in getattr(det, "ml_probabilities", {}).items()
                },
                "merged_from_fragments": int(getattr(det, "merged_from", 1)),
            },
            "notes": list(det.notes),
        }
        detection_records.append(record)

    uses_cnn = summary.get("classifier") == "cnn"

    return {
        "project": PROJECT_NAME,
        "project_title": PROJECT_TITLE,
        "version": PROJECT_VERSION,
        "generated_at": timestamp_iso(),
        "detector": {
            "type": (
                "OpenCV candidate generation + locally trained CNN classification"
                if uses_cnn else
                "OpenCV local-background anomaly detector (heuristic fallback)"
            ),
            "trained_model_used": bool(uses_cnn),
            "classifier": summary.get("classifier", "heuristic"),
            "confidence_source": summary.get(
                "confidence_source",
                "CNN prediction confidence" if uses_cnn else "heuristic estimate",
            ),
            "note": (
                "OpenCV proposes candidate regions; the locally trained CNN in "
                "models/stain_classifier.keras decides whether each candidate is a "
                "stain and which class it is. Confidence is that class's softmax "
                "probability. The model was trained on synthetic floor crops, so "
                "these figures describe synthetic-domain performance, not validated "
                "real-world accuracy."
                if uses_cnn else
                "No trained model was used for this run. Stain type, confidence and "
                "severity are heuristic estimates derived from colour, brightness, "
                "saturation and texture statistics."
            ),
        },
        "image": image_info,
        "settings": settings,
        "summary": summary,
        "detections": detection_records,
    }


def report_to_json_bytes(report: Dict[str, Any]) -> bytes:
    """Serialise a report dictionary to pretty-printed UTF-8 JSON bytes."""
    return json.dumps(report, indent=2, ensure_ascii=False, default=_json_default).encode("utf-8")


def _json_default(value: Any) -> Any:
    """Fallback serialiser for NumPy scalar types."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    return str(value)


def dataframe_to_csv_bytes(frame: pd.DataFrame) -> bytes:
    """Serialise a DataFrame to UTF-8 CSV bytes (no index column)."""
    return frame.to_csv(index=False).encode("utf-8")


def ensure_directory(path: Path) -> Path:
    """Create ``path`` (and parents) if needed and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_bytes(path: Path, data: bytes) -> Path:
    """Write bytes to ``path``, creating parent folders as required."""
    ensure_directory(path.parent)
    path.write_bytes(data)
    return path


def save_report_bundle(
    annotated_png: bytes,
    json_bytes: bytes,
    csv_bytes: bytes,
    reports_dir: Path = REPORTS_DIR,
    prefix: str = "cc1_report",
) -> List[Path]:
    """Write the annotated image, JSON and CSV reports into ``reports/``.

    Returns the list of written paths.
    """
    stamp = timestamp_slug()
    written: List[Path] = [
        save_bytes(reports_dir / f"{prefix}_{stamp}_annotated.png", annotated_png),
        save_bytes(reports_dir / f"{prefix}_{stamp}.json", json_bytes),
        save_bytes(reports_dir / f"{prefix}_{stamp}.csv", csv_bytes),
    ]
    return written


def describe_image(image: np.ndarray, source_name: str, scale: float = 1.0) -> Dict[str, Any]:
    """Build the ``image`` block used inside the JSON report."""
    height, width = image.shape[:2]
    return {
        "source": source_name,
        "analysis_width": int(width),
        "analysis_height": int(height),
        "analysis_pixels": int(width * height),
        "resize_scale_applied": round(float(scale), 4),
    }


def iter_chunks(items: Iterable[Any], size: int) -> Iterable[List[Any]]:
    """Yield ``items`` in lists of at most ``size`` elements (used for layout)."""
    chunk: List[Any] = []
    for item in items:
        chunk.append(item)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk
