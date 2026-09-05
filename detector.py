"""
CC1 Vision - stain detection engine (OpenCV + CNN hybrid).

OpenCV *proposes* candidate regions; the locally trained CNN in
``models/stain_classifier.keras`` *decides* whether each candidate is really a
stain and which kind it is. The pipeline is:

    1.  validate + optionally downscale the image (aspect ratio preserved)
    2.  edge-preserving noise reduction
    3.  convert to LAB / HSV / grayscale
    4.  estimate the *local floor appearance* with a large-kernel median
        (this is the "what should this area look like" background model)
    5.  compare every pixel with its local background:
            - darkness difference   (LAB L channel)
            - brightness difference (LAB L channel, other direction)
            - colour difference     (LAB a/b channels)
            - saturation difference (HSV S channel)
            - texture difference    (local standard deviation of grayscale)
    6.  scale each difference map by a robust noise estimate (median + MAD)
        so a clean floor produces near-zero scores
    7.  fuse them into a single 0..1 anomaly map
    8.  threshold (sensitivity slider) with a coverage guard
    9.  morphological opening + closing
    10. remove thin line structures (tile grout joints)
    11. contour extraction + minimum-area filtering  -> CANDIDATE REGIONS
    12. merge fragments that belong to one physical stain
    13. geometric false-positive filters: thin straight lines, grout joints,
        glare, large uniform shadows, tile-texture noise
    14. CNN classification of every surviving candidate crop
    15. image-border / corner rule (edges need strong CNN evidence)
    16. severity from the CNN confidence and the affected area

Where the numbers come from
---------------------------
When the trained model is present, ``stain_type`` is the CNN's predicted class
and ``confidence`` is that class's softmax probability - not a heuristic score.
``Detection.classifier`` records which path produced each result.

The classical rule set is still here as a *fallback*. If TensorFlow is missing
or ``models/stain_classifier.keras`` has not been trained yet, the detector
degrades to heuristic labels and says so, instead of failing.

The CNN is trained locally by ``train_classifier.py`` on crops built by
``dataset_generator.py``. No pretrained weights, no cloud service, no API key.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from utils import clamp, resize_max_dimension, safe_div

# --------------------------------------------------------------------------
# Public constants
# --------------------------------------------------------------------------

#: Labels the classical heuristic fallback can produce. The CNN produces the
#: labels in ``CNN_CATEGORIES`` instead; both sets are understood by
#: ``cleaning_engine.CLEANING_ACTIONS``.
STAIN_CATEGORIES: Tuple[str, ...] = (
    "Liquid Spill",
    "Mud / Dirt",
    "Dust / Scuff",
    "Dark Stain",
    "Colored Stain",
    "Grime / Dirty Patch",
    "Generic Stain",
)

#: Labels produced by the trained CNN (``models/model_config.json``).
CNN_CATEGORIES: Tuple[str, ...] = (
    "Mud / Dirt",
    "Liquid Spill",
    "Grease / Oil",
    "Dark Stain",
    "Scattered Dirt",
)

#: Every label the dashboard may display.
ALL_CATEGORIES: Tuple[str, ...] = tuple(
    dict.fromkeys(STAIN_CATEGORIES + CNN_CATEGORIES)
)

SEVERITY_LEVELS: Tuple[str, ...] = ("Low", "Medium", "High")

#: BGR colours used when drawing annotations.
SEVERITY_COLORS: Dict[str, Tuple[int, int, int]] = {
    "High": (48, 48, 214),      # red
    "Medium": (30, 150, 235),   # amber
    "Low": (110, 160, 60),      # green
}

DEFAULT_COLOR: Tuple[int, int, int] = (200, 120, 40)


# --------------------------------------------------------------------------
# Configuration + result containers
# --------------------------------------------------------------------------


@dataclass
class DetectorConfig:
    """Tunable parameters for :func:`detect_stains`."""

    sensitivity: float = 50.0          # 0 (strict) .. 100 (aggressive)
    min_area_px: int = 400             # minimum region area on the analysis image
    max_dimension: int = 1280          # longest side used for analysis
    min_confidence: float = 0.35       # regions below this are discarded
    max_detections: int = 60           # hard cap, keeps the UI responsive
    max_coverage: float = 0.55         # guard: never mark more than 55% of the floor
    denoise: bool = True
    suppress_grout: bool = True        # remove long thin tile joints
    suppress_shadows: bool = True      # remove soft-edged colourless dark areas
    suppress_glare: bool = True        # remove bright desaturated reflections

    # --- CNN stage -------------------------------------------------------
    use_cnn: bool = True               # let the trained CNN decide accept/reject
    min_stain_probability: float = 0.60  # softmax floor for accepting a stain
    crop_context: float = 1.6          # crop side = context x region side

    # --- false-positive geometry ----------------------------------------
    #: Fragments whose bounding boxes are closer than this (fraction of the
    #: shorter image side) are merged into one physical stain.
    merge_gap_ratio: float = 0.030
    #: Width of the image-border band, as a fraction of the shorter side.
    border_margin_ratio: float = 0.020
    #: A border/corner region needs at least this CNN probability to survive.
    border_strong_probability: float = 0.85
    #: Regions smaller than this fraction of the image AND weakly anomalous
    #: are treated as tile texture noise rather than stains.
    texture_noise_area_ratio: float = 0.00035
    #: Fraction of a region that may sit on a detected grout joint.
    max_joint_overlap: float = 0.50

    def as_dict(self) -> Dict[str, Any]:
        """Serialise the configuration for the JSON report."""
        return {
            "sensitivity": float(self.sensitivity),
            "min_area_px": int(self.min_area_px),
            "max_dimension": int(self.max_dimension),
            "min_confidence": float(self.min_confidence),
            "max_detections": int(self.max_detections),
            "denoise": bool(self.denoise),
            "suppress_grout": bool(self.suppress_grout),
            "suppress_shadows": bool(self.suppress_shadows),
            "suppress_glare": bool(self.suppress_glare),
            "use_cnn": bool(self.use_cnn),
            "min_stain_probability": float(self.min_stain_probability),
            "merge_gap_ratio": float(self.merge_gap_ratio),
            "border_margin_ratio": float(self.border_margin_ratio),
        }


@dataclass
class Detection:
    """One detected stain-like region."""

    id: int
    stain_type: str
    confidence: float
    severity: str
    area_px: int
    area_ratio: float
    x: int
    y: int
    width: int
    height: int
    center_x: int
    center_y: int
    anomaly_score: float
    darkness_delta: float
    brightness_delta: float
    color_delta: float
    saturation_delta: float
    texture_delta: float
    hue: float
    edge_strength: float
    solidity: float
    circularity: float
    elongation: float
    contour: np.ndarray = field(repr=False, default_factory=lambda: np.empty((0, 1, 2), np.int32))
    notes: List[str] = field(default_factory=list)

    # --- classifier provenance ------------------------------------------
    #: ``"cnn"`` when the label and confidence came from the trained network,
    #: ``"heuristic"`` when the classical fallback was used.
    classifier: str = "heuristic"
    #: Dataset class name chosen by the CNN, e.g. ``"mud_dirt"``.
    ml_class: str = ""
    #: Softmax probability of :attr:`ml_class`. This is what ``confidence``
    #: carries whenever ``classifier == "cnn"``.
    ml_confidence: float = 0.0
    #: Probability the CNN assigned to the ``clean`` class.
    clean_probability: float = 0.0
    #: Full probability vector, class name -> probability.
    ml_probabilities: Dict[str, float] = field(default_factory=dict)
    #: How many raw fragments were merged into this detection.
    merged_from: int = 1
    #: Fraction of the region sitting inside the image-border band.
    border_fraction: float = 0.0
    #: Fraction of the region overlapping a detected grout joint.
    joint_overlap: float = 0.0

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        """Bounding box as ``(x, y, w, h)``."""
        return self.x, self.y, self.width, self.height

    def to_dict(self) -> Dict[str, Any]:
        """Plain-dict view (contour excluded) for tables and reports."""
        return {
            "id": int(self.id),
            "stain_type": self.stain_type,
            "confidence": round(float(self.confidence), 3),
            "severity": self.severity,
            "area_px": int(self.area_px),
            "area_ratio": round(float(self.area_ratio), 5),
            "x": int(self.x),
            "y": int(self.y),
            "width": int(self.width),
            "height": int(self.height),
            "center_x": int(self.center_x),
            "center_y": int(self.center_y),
            "anomaly_score": round(float(self.anomaly_score), 3),
            "classifier": self.classifier,
            "ml_class": self.ml_class,
            "ml_confidence": round(float(self.ml_confidence), 4),
            "clean_probability": round(float(self.clean_probability), 4),
            "ml_probabilities": {
                key: round(float(value), 4) for key, value in self.ml_probabilities.items()
            },
            "merged_from": int(self.merged_from),
            "notes": list(self.notes),
        }


@dataclass
class DetectionResult:
    """Everything the dashboard needs after one analysis run."""

    image_bgr: np.ndarray                       # analysis-resolution input image
    detections: List[Detection]
    mask: np.ndarray                            # uint8 0/255 binary detection mask
    anomaly_map: np.ndarray                     # float32 0..1 anomaly score map
    scale: float                                # resize factor applied to the input
    original_size: Tuple[int, int]              # (width, height) before resizing
    threshold_used: float
    coverage_ratio: float                       # dirty pixels / total pixels
    elapsed_seconds: float
    rejected: Dict[str, int] = field(default_factory=dict)
    #: ``"cnn"`` when the trained network judged the candidates, otherwise
    #: ``"heuristic"``. Drives the wording the dashboard uses for confidence.
    classifier_used: str = "heuristic"
    #: Why the CNN was not used, when it was not.
    classifier_note: str = ""
    #: How many candidate regions the OpenCV stage proposed before the CNN.
    candidates_considered: int = 0

    @property
    def uses_cnn(self) -> bool:
        """True when the displayed confidences are CNN probabilities."""
        return self.classifier_used == "cnn"

    @property
    def confidence_label(self) -> str:
        """Wording for the confidence figure shown in the dashboard."""
        return "CNN prediction confidence" if self.uses_cnn else "heuristic estimate"

    @property
    def count(self) -> int:
        """Number of accepted detections."""
        return len(self.detections)

    @property
    def average_confidence(self) -> float:
        """Mean confidence over accepted detections (0.0 when none)."""
        if not self.detections:
            return 0.0
        return float(np.mean([d.confidence for d in self.detections]))

    @property
    def high_severity_count(self) -> int:
        """How many detections were rated ``High``."""
        return sum(1 for d in self.detections if d.severity == "High")

    def summary(self) -> Dict[str, Any]:
        """Summary block used by the JSON report and the metric strip."""
        by_type: Dict[str, int] = {}
        for det in self.detections:
            by_type[det.stain_type] = by_type.get(det.stain_type, 0) + 1
        return {
            "detected_stains": self.count,
            "average_confidence": round(self.average_confidence, 3),
            "high_severity_areas": self.high_severity_count,
            "dirty_coverage_ratio": round(float(self.coverage_ratio), 5),
            "threshold_used": round(float(self.threshold_used), 4),
            "analysis_seconds": round(float(self.elapsed_seconds), 3),
            "counts_by_type": by_type,
            "rejected_regions": dict(self.rejected),
            "classifier": self.classifier_used,
            "classifier_note": self.classifier_note,
            "confidence_source": self.confidence_label,
            "candidates_considered": int(self.candidates_considered),
        }


# --------------------------------------------------------------------------
# Low level image helpers
# --------------------------------------------------------------------------


def _odd(value: int, minimum: int = 3) -> int:
    """Return the nearest odd integer >= ``minimum``."""
    value = int(max(minimum, value))
    return value if value % 2 == 1 else value + 1


def _local_background(channel: np.ndarray, kernel: int) -> np.ndarray:
    """Estimate the local floor appearance of a single 8-bit channel.

    A large median filter removes stains (which are small relative to the
    kernel) while keeping the underlying floor colour and lighting gradient.
    For speed the median is computed on a quarter-scale copy and then
    upsampled - this is visually equivalent for background estimation.
    """
    height, width = channel.shape[:2]
    small_w = max(16, width // 4)
    small_h = max(16, height // 4)
    small = cv2.resize(channel, (small_w, small_h), interpolation=cv2.INTER_AREA)

    small_kernel = _odd(kernel // 4, 3)
    small_kernel = min(small_kernel, _odd(min(small_w, small_h) // 2 - 1, 3), 81)
    blurred = cv2.medianBlur(small, small_kernel)
    # A second, softer pass suppresses residual structure from large stains.
    blurred = cv2.blur(blurred, (5, 5))
    return cv2.resize(blurred, (width, height), interpolation=cv2.INTER_LINEAR)


def _local_std(gray: np.ndarray, kernel: int) -> np.ndarray:
    """Local standard deviation of the grayscale image (texture energy)."""
    kernel = _odd(kernel, 3)
    values = gray.astype(np.float32)
    mean = cv2.boxFilter(values, -1, (kernel, kernel), normalize=True,
                         borderType=cv2.BORDER_REFLECT)
    mean_sq = cv2.boxFilter(values * values, -1, (kernel, kernel), normalize=True,
                            borderType=cv2.BORDER_REFLECT)
    variance = np.maximum(mean_sq - mean * mean, 0.0)
    return np.sqrt(variance, dtype=np.float32)


def _robust_scale(difference: np.ndarray, floor: float) -> float:
    """Noise-aware scale for a one-sided difference map.

    Uses median + 3 * (MAD-based sigma). ``floor`` is an absolute minimum in
    pixel units so that a perfectly uniform floor cannot produce large scores
    just because its own noise is tiny.
    """
    sample = difference
    if sample.size > 400_000:  # subsample large images for speed
        step = int(np.sqrt(sample.size / 400_000)) + 1
        sample = sample[::step, ::step]
    median = float(np.median(sample))
    mad = float(np.median(np.abs(sample - median)))
    sigma = 1.4826 * mad
    return float(max(floor, median + 3.0 * sigma))


def _normalise(difference: np.ndarray, floor: float) -> np.ndarray:
    """Scale a difference map into 0..1 using :func:`_robust_scale`."""
    scale = _robust_scale(difference, floor)
    return np.clip(difference / scale, 0.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------
# Stage 1-7: anomaly map
# --------------------------------------------------------------------------


@dataclass
class _FeatureMaps:
    """Intermediate per-pixel maps shared by the later pipeline stages."""

    anomaly: np.ndarray
    darkness: np.ndarray
    brightness: np.ndarray
    color: np.ndarray
    saturation: np.ndarray
    texture: np.ndarray
    gradient: np.ndarray
    lightness: np.ndarray
    background_lightness: np.ndarray
    hue: np.ndarray
    raw_saturation: np.ndarray
    joint_mask: np.ndarray


def _line_kernel(length: int, angle_degrees: float) -> np.ndarray:
    """A 1-pixel-wide straight-line structuring element at ``angle_degrees``."""
    length = max(3, int(length) | 1)
    kernel = np.zeros((length, length), dtype=np.uint8)
    centre = length // 2
    radians = np.deg2rad(angle_degrees)
    dx, dy = np.cos(radians), -np.sin(radians)
    end_a = (int(round(centre - dx * centre)), int(round(centre - dy * centre)))
    end_b = (int(round(centre + dx * centre)), int(round(centre + dy * centre)))
    cv2.line(kernel, end_a, end_b, 1, 1)
    return kernel


def estimate_joint_mask(darkness: np.ndarray, min_delta: float = 6.0) -> np.ndarray:
    """Locate tile grout joints so they can be excluded from the anomaly map.

    Grout lines are *long, continuous and thin*. We keep only the structures
    that survive a long 1-pixel-wide opening but are destroyed by a thick
    square opening (which a real stain would survive).

    Joints are probed at four orientations, not just horizontal and vertical,
    because a floor photographed at an angle has tilted joints. The result is
    also closed along each line direction so that a joint interrupted by a
    stain, a highlight or a chip is still recovered as one continuous line.

    Returns a uint8 mask where 255 marks joint pixels.
    """
    height, width = darkness.shape[:2]
    binary = ((darkness > float(min_delta)).astype(np.uint8)) * 255

    line_length = max(31, int(0.12 * max(height, width)))
    thickness = _odd(max(9, line_length // 8), 9)

    joints = np.zeros_like(binary)
    for angle in (0.0, 45.0, 90.0, 135.0):
        kernel = _line_kernel(line_length, angle)
        # Close first so small interruptions do not break a long joint apart.
        bridged = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,
                                   _line_kernel(max(9, line_length // 4), angle))
        opened = cv2.morphologyEx(bridged, cv2.MORPH_OPEN, kernel)
        joints = cv2.bitwise_or(joints, opened)

    thick = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (thickness, thickness)),
    )
    joints = cv2.bitwise_and(joints, cv2.bitwise_not(thick))

    # Grow slightly so the soft shoulders of a joint are covered too.
    grow = _odd(max(5, thickness // 2), 5)
    joints = cv2.dilate(joints, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow, grow)))
    return joints


def build_feature_maps(image_bgr: np.ndarray, config: DetectorConfig) -> _FeatureMaps:
    """Run stages 1-7 of the pipeline and return all per-pixel maps."""
    height, width = image_bgr.shape[:2]

    # --- 3. noise reduction (edge preserving so stain borders survive) ---
    if config.denoise:
        working = cv2.bilateralFilter(image_bgr, d=7, sigmaColor=35, sigmaSpace=7)
    else:
        working = image_bgr.copy()

    # --- 4. colour spaces ---
    lab = cv2.cvtColor(working, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(working, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)

    lightness = lab[:, :, 0].astype(np.float32)
    a_channel = lab[:, :, 1].astype(np.float32)
    b_channel = lab[:, :, 2].astype(np.float32)
    hue = hsv[:, :, 0].astype(np.float32)
    saturation = hsv[:, :, 1].astype(np.float32)

    # --- 5. local background model ---
    # The kernel must be clearly larger than the stains we want to find,
    # otherwise a wide stain is absorbed into its own "background".
    background_kernel = _odd(max(41, int(0.30 * min(height, width))), 41)
    bg_lightness = _local_background(lab[:, :, 0], background_kernel).astype(np.float32)
    bg_a = _local_background(lab[:, :, 1], background_kernel).astype(np.float32)
    bg_b = _local_background(lab[:, :, 2], background_kernel).astype(np.float32)
    bg_saturation = _local_background(hsv[:, :, 1], background_kernel).astype(np.float32)

    texture_kernel = _odd(max(7, int(0.012 * min(height, width))), 7)
    texture = _local_std(gray, texture_kernel)
    texture_u8 = np.clip(texture, 0, 255).astype(np.uint8)
    bg_texture = _local_background(texture_u8, background_kernel).astype(np.float32)

    # --- 6/7. differences against the local floor ---
    darkness = np.clip(bg_lightness - lightness, 0, None)          # stain darker than floor
    brightness = np.clip(lightness - bg_lightness, 0, None)        # stain brighter than floor
    color = np.sqrt((a_channel - bg_a) ** 2 + (b_channel - bg_b) ** 2, dtype=np.float32)
    saturation_diff = np.clip(saturation - bg_saturation, 0, None)
    texture_diff = np.clip(texture - bg_texture, 0, None)

    # Absolute floors (in channel units) encode "how much difference is real".
    s_dark = _normalise(darkness, floor=9.0)
    s_bright = _normalise(brightness, floor=11.0)
    s_color = _normalise(color, floor=7.0)
    s_sat = _normalise(saturation_diff, floor=14.0)
    s_texture = _normalise(texture_diff, floor=7.0)

    # Weighted soft-OR fusion: a single strong cue can saturate the score.
    anomaly = (
        0.46 * s_dark
        + 0.28 * s_color
        + 0.18 * s_sat
        + 0.18 * s_texture
        + 0.20 * s_bright
    )
    anomaly = np.clip(anomaly, 0.0, 1.0).astype(np.float32)
    anomaly = cv2.GaussianBlur(anomaly, (5, 5), 0)

    # Tile joints are floor structure, not dirt: damp them before thresholding.
    if config.suppress_grout:
        joint_mask = estimate_joint_mask(darkness, min_delta=6.0)
        # Near-total suppression: a joint should not be able to reach the
        # threshold on its own, only a real stain lying across one should.
        attenuation = 1.0 - 0.97 * (joint_mask.astype(np.float32) / 255.0)
        anomaly = anomaly * attenuation
    else:
        joint_mask = np.zeros(anomaly.shape, dtype=np.uint8)

    # Gradient magnitude is used later to separate hard-edged stains from
    # soft-edged shadows.
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(grad_x, grad_y)
    gradient = np.clip(gradient / max(1.0, _robust_scale(gradient, 25.0)), 0.0, 1.0)

    return _FeatureMaps(
        anomaly=anomaly,
        darkness=darkness,
        brightness=brightness,
        color=color,
        saturation=saturation_diff,
        texture=texture_diff,
        gradient=gradient.astype(np.float32),
        lightness=lightness,
        background_lightness=bg_lightness,
        hue=hue,
        raw_saturation=saturation,
        joint_mask=joint_mask,
    )


# --------------------------------------------------------------------------
# Stage 8-10: binary mask
# --------------------------------------------------------------------------


def _sensitivity_to_threshold(sensitivity: float) -> float:
    """Map the 0..100 sensitivity slider onto an anomaly threshold."""
    sensitivity = clamp(sensitivity, 0.0, 100.0)
    return float(np.interp(sensitivity, [0.0, 50.0, 100.0], [0.62, 0.34, 0.16]))


def build_mask(anomaly: np.ndarray, config: DetectorConfig) -> Tuple[np.ndarray, float]:
    """Threshold the anomaly map and clean it morphologically.

    Returns the binary mask (0/255) and the threshold that was actually used.
    """
    threshold = _sensitivity_to_threshold(config.sensitivity)
    mask = (anomaly >= threshold).astype(np.uint8) * 255

    # Coverage guard: if a textured floor lights up everywhere, raise the
    # threshold so we report the worst areas instead of the whole image.
    coverage = float(np.count_nonzero(mask)) / float(mask.size)
    if coverage > config.max_coverage:
        percentile = 100.0 * (1.0 - config.max_coverage)
        threshold = float(max(threshold, np.percentile(anomaly, percentile)))
        mask = (anomaly >= threshold).astype(np.uint8) * 255

    height, width = anomaly.shape[:2]
    small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # The closing kernel doubles as the first fragment-merging stage: a stain
    # that broke into pieces across a grout line or a highlight is rejoined
    # here, before anything is counted as a separate detection.
    close_size = _odd(max(7, int(config.merge_gap_ratio * min(height, width))), 7)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, small_kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    return mask, threshold


def suppress_line_structures(mask: np.ndarray) -> np.ndarray:
    """Remove long, thin horizontal/vertical structures (tile grout joints).

    A structure is considered a joint when it survives a long 1-pixel-wide
    opening but is destroyed by a small square opening (i.e. it is long but
    not thick).
    """
    height, width = mask.shape[:2]
    line_length = max(25, int(0.10 * max(height, width)))
    thickness = _odd(max(5, line_length // 12), 5)

    horizontal = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (line_length, 1)),
    )
    vertical = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, line_length)),
    )
    thick = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (thickness, thickness)),
    )

    lines = cv2.bitwise_or(horizontal, vertical)
    lines = cv2.bitwise_and(lines, cv2.bitwise_not(thick))
    lines = cv2.dilate(lines, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    return cv2.bitwise_and(mask, cv2.bitwise_not(lines))


# --------------------------------------------------------------------------
# Stage 11-13: regions, classification, scoring
# --------------------------------------------------------------------------


def _box_gap(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    """Shortest distance in pixels between two ``(x, y, w, h)`` boxes.

    Returns 0.0 when the boxes touch or overlap.
    """
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ax1, ay1 = ax0 + aw, ay0 + ah
    bx1, by1 = bx0 + bw, by0 + bh
    dx = max(0, max(bx0 - ax1, ax0 - bx1))
    dy = max(0, max(by0 - ay1, ay0 - by1))
    return float(np.hypot(dx, dy))


def merge_nearby_contours(
    contours: Sequence[np.ndarray],
    shape: Tuple[int, int],
    gap: float,
) -> List[Tuple[np.ndarray, int]]:
    """Group contours that belong to one physical stain and fuse each group.

    Two fragments are considered the same stain when the shortest distance
    between their bounding boxes is at most ``gap`` pixels. Each group is
    rasterised and closed with a disc wide enough to bridge that distance, so
    the group comes back as a single outline rather than a cluster of boxes.

    Returns ``(contour, fragment_count)`` pairs.
    """
    count = len(contours)
    if count <= 1:
        return [(contour, 1) for contour in contours]

    boxes = [cv2.boundingRect(contour) for contour in contours]
    parent = list(range(count))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a: int, b: int) -> None:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    for i in range(count):
        for j in range(i + 1, count):
            if _box_gap(boxes[i], boxes[j]) <= gap:
                union(i, j)

    groups: Dict[int, List[int]] = {}
    for index in range(count):
        groups.setdefault(find(index), []).append(index)

    merged: List[Tuple[np.ndarray, int]] = []
    kernel_size = _odd(max(3, int(gap) * 2 + 1), 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    for members in groups.values():
        if len(members) == 1:
            merged.append((contours[members[0]], 1))
            continue

        canvas = np.zeros(shape, dtype=np.uint8)
        cv2.drawContours(canvas, [contours[k] for k in members], -1, 255, cv2.FILLED)
        canvas = cv2.morphologyEx(canvas, cv2.MORPH_CLOSE, kernel)
        found, _ = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(found) == 1:
            merged.append((found[0], len(members)))
        elif found:
            # Closing did not bridge every piece (possible for a very sparse
            # cluster); fall back to the hull of the whole group.
            merged.append((cv2.convexHull(np.vstack(found)), len(members)))
    return merged


def _border_fraction(region_mask: np.ndarray, offset: Tuple[int, int],
                     shape: Tuple[int, int], margin: int) -> float:
    """Fraction of a region's pixels lying inside the image-border band."""
    total = cv2.countNonZero(region_mask)
    if total == 0 or margin <= 0:
        return 0.0

    height, width = shape
    x0, y0 = offset
    band = np.zeros(region_mask.shape, dtype=np.uint8)
    # Paint the part of the global border band that falls inside this window.
    band[:] = 255
    inner_top = max(0, margin - y0)
    inner_left = max(0, margin - x0)
    inner_bottom = region_mask.shape[0] - max(0, (y0 + region_mask.shape[0]) - (height - margin))
    inner_right = region_mask.shape[1] - max(0, (x0 + region_mask.shape[1]) - (width - margin))
    if inner_bottom > inner_top and inner_right > inner_left:
        band[inner_top:inner_bottom, inner_left:inner_right] = 0

    inside = cv2.countNonZero(cv2.bitwise_and(region_mask, band))
    return float(inside) / float(total)


def classifier_crop(
    image_bgr: np.ndarray,
    bbox: Tuple[int, int, int, int],
    out_size: int,
    context: float,
) -> Optional[np.ndarray]:
    """Cut the square, context-padded crop the CNN expects for one region."""
    try:
        from ml_classifier import extract_crop
    except Exception:  # pragma: no cover - ml_classifier is always present
        return None
    return extract_crop(image_bgr, bbox, out_size, context)


def _mean_in_mask(source: np.ndarray, region_mask: np.ndarray) -> float:
    """Mean of ``source`` over the non-zero pixels of ``region_mask``."""
    return float(cv2.mean(source, mask=region_mask)[0])


def _circular_mean_hue(hue_roi: np.ndarray, region_mask: np.ndarray) -> float:
    """Circular mean of OpenCV hue values (0..179) inside a region."""
    selected = hue_roi[region_mask > 0]
    if selected.size == 0:
        return 0.0
    angles = selected.astype(np.float32) * (2.0 * np.pi / 180.0)
    mean_angle = np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))
    if mean_angle < 0:
        mean_angle += 2.0 * np.pi
    return float(mean_angle * 180.0 / (2.0 * np.pi))


def _estimate_stain_type(
    darkness: float,
    brightness: float,
    color: float,
    saturation_delta: float,
    saturation_abs: float,
    texture: float,
    hue: float,
    circularity: float,
    solidity: float,
    area_ratio: float,
    edge_strength: float,
) -> Tuple[str, str]:
    """Heuristic stain category from region statistics.

    Returns ``(category, reason)``. Every category gets a transparent score;
    the highest one wins. This is a rule set, not a trained classifier.
    """
    scores: Dict[str, float] = {name: 0.0 for name in STAIN_CATEGORIES}
    reasons: Dict[str, str] = {}

    # Hue is meaningless for near-grey pixels, so it is only trusted when the
    # region carries real colour. This stops grey grout from reading as "mud".
    has_color = saturation_delta >= 10.0 or color >= 12.0
    brown_hue = bool(has_color and saturation_abs >= 30.0 and 5.0 <= hue <= 32.0)
    strong_color = saturation_delta >= 18.0 or color >= 14.0

    # Mud / dirt: brown-ish, coloured, darker, textured.
    scores["Mud / Dirt"] = (
        1.4 * float(brown_hue)
        + 0.9 * min(saturation_delta / 30.0, 1.0)
        + 0.6 * min(darkness / 30.0, 1.0)
        + 0.5 * min(texture / 12.0, 1.0)
    )
    reasons["Mud / Dirt"] = "brown hue with raised saturation and texture"

    # Colored stain: strongly coloured but not brown.
    scores["Colored Stain"] = (
        1.3 * float(strong_color and not brown_hue)
        + 1.0 * min(color / 25.0, 1.0)
        + 0.6 * min(saturation_delta / 30.0, 1.0)
    )
    reasons["Colored Stain"] = "large colour deviation from the surrounding floor"

    # Liquid spill: clearly darker, smooth inside, rounded, crisp boundary.
    scores["Liquid Spill"] = (
        1.5 * min(darkness / 25.0, 1.0)
        + 0.45 * max(0.0, 1.0 - texture / 8.0)
        + 0.70 * min(circularity / 0.60, 1.0)
        + 0.60 * min(edge_strength / 0.35, 1.0)
        + 0.80 * float(darkness >= 14.0)   # a spill is definitely darker than the floor
        - 1.40 * float(strong_color)
    )
    reasons["Liquid Spill"] = (
        "clearly darker, smooth inside and sharply bounded, like a wet patch"
    )

    # Dark stain: very dark, low saturation.
    scores["Dark Stain"] = (
        1.6 * min(darkness / 45.0, 1.0)
        + 0.5 * max(0.0, 1.0 - saturation_abs / 70.0)
        - 0.6 * float(strong_color)
    )
    reasons["Dark Stain"] = "strong brightness drop with weak colour signal"

    # Dust / scuff: light deposit or dry mark, low darkness, small footprint.
    scores["Dust / Scuff"] = (
        1.3 * min(brightness / 14.0, 1.0)
        + 0.8 * max(0.0, 1.0 - darkness / 14.0)
        + 0.6 * min(texture / 9.0, 1.0)
        + 0.4 * max(0.0, 1.0 - area_ratio / 0.02)
        - 0.7 * float(strong_color)
    )
    reasons["Dust / Scuff"] = "light, low-contrast dry deposit rather than a wet mark"

    # Grime: wide, faint, soft-edged soiling - the opposite profile of a spill.
    scores["Grime / Dirty Patch"] = (
        0.9 * min(texture / 10.0, 1.0)
        + 1.0 * min(area_ratio / 0.02, 1.0)
        + 0.6 * max(0.0, 1.0 - solidity / 0.85)
        + 0.5 * min(darkness / 22.0, 1.0)
        + 0.7 * max(0.0, 1.0 - edge_strength / 0.30)
        + 1.0 * float(darkness < 12.0 and area_ratio >= 0.010)
    )
    reasons["Grime / Dirty Patch"] = (
        "wide, faint, soft-edged soiling spread over a large area"
    )

    # Generic is the honest fallback: it wins whenever no specific profile is
    # clearly supported by the measurements.
    scores["Generic Stain"] = 1.60

    category = max(scores, key=lambda key: scores[key])
    return category, reasons.get(category, "does not match a specific stain profile")


def _estimate_confidence(
    anomaly_mean: float,
    edge_strength: float,
    area_ratio: float,
    solidity: float,
) -> float:
    """Confidence from anomaly strength, edge crispness, size and shape."""
    contrast = clamp(anomaly_mean, 0.0, 1.0)
    edge_factor = clamp(edge_strength / 0.40, 0.0, 1.0)
    size_factor = clamp(area_ratio / 0.006, 0.0, 1.0)
    shape_factor = clamp(solidity, 0.0, 1.0)

    raw = 0.50 * contrast + 0.20 * edge_factor + 0.18 * size_factor + 0.12 * shape_factor
    return clamp(0.25 + 0.74 * raw, 0.0, 0.99)


def _estimate_severity(confidence: float, area_ratio: float) -> str:
    """Severity from confidence combined with how much floor is affected."""
    area_component = clamp(area_ratio / 0.05, 0.0, 1.0)
    score = 0.55 * confidence + 0.45 * area_component
    if score >= 0.62 or (area_ratio >= 0.06 and confidence >= 0.6):
        return "High"
    if score >= 0.40:
        return "Medium"
    return "Low"


def _extract_regions(
    mask: np.ndarray,
    maps: _FeatureMaps,
    config: DetectorConfig,
    analysis_image: np.ndarray,
    classifier: Optional[Any] = None,
) -> Tuple[List[Detection], Dict[str, int], np.ndarray, str, str, int]:
    """Turn the binary mask into scored :class:`Detection` objects.

    The pipeline here is:

    1. contours -> raw fragments,
    2. drop fragments that are too small to be anything,
    3. merge fragments belonging to one physical stain,
    4. measure each candidate and apply the geometric false-positive filters
       (thin line, grout joint, glare, shadow, tile-texture noise),
    5. hand every surviving candidate crop to the CNN, which makes the final
       stain / not-a-stain decision and supplies the label and confidence,
    6. apply the image-border rule, which only lets an edge or corner region
       through when the CNN is *strongly* convinced it is a stain.

    When the CNN is unavailable, step 5 falls back to the classical heuristic
    label and score, so the dashboard keeps working.

    Also returns a rejection counter, the *accepted* mask (filtered regions are
    erased so the displayed mask matches the displayed boxes), which classifier
    was used, why, and how many candidates were considered.
    """
    height, width = mask.shape[:2]
    image_area = float(height * width)
    accepted_mask = np.zeros_like(mask)
    rejected: Dict[str, int] = {
        "too_small": 0,
        "texture_noise": 0,
        "thin_line": 0,
        "grout_line": 0,
        "shadow": 0,
        "glare": 0,
        "image_border": 0,
        "cnn_not_a_stain": 0,
        "cnn_low_confidence": 0,
        "low_confidence": 0,
        "over_limit": 0,
    }

    # --- 1/2. raw fragments -------------------------------------------------
    # The area gate here is deliberately far below ``min_area_px``: a faint
    # stain often arrives as a handful of small pieces, and dropping them now
    # would destroy the very fragments that step 3 is meant to reassemble.
    # ``min_area_px`` is applied again after merging, to the whole region.
    fragment_floor = max(20.0, float(config.min_area_px) / 8.0)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    scored: List[Tuple[float, np.ndarray]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < fragment_floor:
            rejected["too_small"] += 1
            continue
        scored.append((area, contour))

    # Merging is quadratic in the fragment count, so keep it bounded on very
    # noisy images by considering only the largest pieces.
    max_fragments = max(120, config.max_detections * 4)
    scored.sort(key=lambda item: item[0], reverse=True)
    if len(scored) > max_fragments:
        rejected["too_small"] += len(scored) - max_fragments
        scored = scored[:max_fragments]
    fragments = [contour for _, contour in scored]

    # --- 3. merge the pieces of one stain ----------------------------------
    merge_gap = max(4.0, config.merge_gap_ratio * float(min(height, width)))
    merged = merge_nearby_contours(fragments, (height, width), merge_gap)

    candidates: List[Tuple[float, np.ndarray, int]] = []
    for contour, fragment_count in merged:
        area = float(cv2.contourArea(contour))
        if area < max(20.0, float(config.min_area_px)):
            rejected["too_small"] += 1
            continue
        candidates.append((area, contour, fragment_count))

    # Keep the largest regions first so the cap removes noise, not real stains.
    candidates.sort(key=lambda item: item[0], reverse=True)
    if len(candidates) > config.max_detections:
        rejected["over_limit"] = len(candidates) - config.max_detections
        candidates = candidates[: config.max_detections]

    candidates_considered = len(candidates)
    border_margin = int(round(config.border_margin_ratio * min(height, width)))

    # --- 4. measure and filter ---------------------------------------------
    #: Candidates that survived geometry, awaiting the CNN's verdict.
    survivors: List[Dict[str, Any]] = []

    for area, contour, fragment_count in candidates:
        x, y, w, h = cv2.boundingRect(contour)

        # Pad the ROI so the edge-strength band has context around the region.
        pad = 8
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(width, x + w + pad)
        y1 = min(height, y + h + pad)

        region_mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        cv2.drawContours(region_mask, [contour], -1, 255, thickness=cv2.FILLED,
                         offset=(-x0, -y0))
        if cv2.countNonZero(region_mask) == 0:
            rejected["too_small"] += 1
            continue

        def roi(source: np.ndarray) -> np.ndarray:
            """Crop a full-size map to the padded region window."""
            return source[y0:y1, x0:x1]

        # --- region statistics -------------------------------------------
        darkness = _mean_in_mask(roi(maps.darkness), region_mask)
        brightness = _mean_in_mask(roi(maps.brightness), region_mask)
        color = _mean_in_mask(roi(maps.color), region_mask)
        saturation_delta = _mean_in_mask(roi(maps.saturation), region_mask)
        texture = _mean_in_mask(roi(maps.texture), region_mask)
        anomaly_mean = _mean_in_mask(roi(maps.anomaly), region_mask)
        saturation_abs = _mean_in_mask(roi(maps.raw_saturation), region_mask)
        hue = _circular_mean_hue(roi(maps.hue), region_mask)

        # Edge strength: mean gradient in a thin band along the contour.
        band_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilated = cv2.dilate(region_mask, band_kernel)
        eroded = cv2.erode(region_mask, band_kernel)
        border_band = cv2.subtract(dilated, eroded)
        if cv2.countNonZero(border_band) > 0:
            edge_strength = _mean_in_mask(roi(maps.gradient), border_band)
        else:
            edge_strength = 0.0

        # --- shape descriptors -------------------------------------------
        perimeter = float(cv2.arcLength(contour, True))
        circularity = clamp(safe_div(4.0 * np.pi * area, perimeter ** 2), 0.0, 1.0)
        hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
        solidity = clamp(safe_div(area, hull_area, default=1.0), 0.0, 1.0)

        (_, _), (rect_w, rect_h), _ = cv2.minAreaRect(contour)
        long_side = max(rect_w, rect_h)
        short_side = max(1.0, min(rect_w, rect_h))
        elongation = float(long_side / short_side)
        rect_fill = clamp(safe_div(area, max(1.0, rect_w * rect_h), default=0.0), 0.0, 1.0)

        area_ratio = area / image_area
        notes: List[str] = []

        # --- false positive filters --------------------------------------
        # (a) Thin straight structures: long, narrow, and filling their own
        #     rotated bounding box - the signature of a grout joint or a seam.
        is_thin_line = (
            (elongation >= 7.0 and short_side <= 0.02 * max(height, width))
            or (elongation >= 4.5 and short_side <= 0.014 * max(height, width)
                and rect_fill >= 0.55)
        )
        if config.suppress_grout and is_thin_line:
            rejected["thin_line"] += 1
            continue

        # (b) Regions sitting mostly on a detected grout joint.
        joint_overlap = 0.0
        if config.suppress_grout and np.any(maps.joint_mask):
            joint_roi = cv2.bitwise_and(roi(maps.joint_mask), region_mask)
            joint_overlap = safe_div(float(cv2.countNonZero(joint_roi)),
                                     float(cv2.countNonZero(region_mask)))
            if joint_overlap > config.max_joint_overlap:
                rejected["grout_line"] += 1
                continue

        # (c) A reflection is bright, colourless, smooth and soft-edged. A
        #     light dust deposit is bright too, but grainy and sharply bounded.
        is_glare = (
            brightness >= 18.0
            and saturation_abs <= 45.0
            and darkness <= 4.0
            and texture <= 3.0
            and edge_strength <= 0.35
        )
        if config.suppress_glare and is_glare:
            rejected["glare"] += 1
            continue

        # (d) Shadows: colourless, textureless, soft-edged darkening. A large
        #     uniform one is the classic false positive, so the wide case is
        #     caught with looser thresholds than the small case.
        looks_like_shadow = (
            darkness >= 6.0
            and color <= 5.0
            and saturation_delta <= 6.0
            and texture <= 5.0
            and edge_strength <= 0.18
        )
        big_uniform_shadow = (
            area_ratio >= 0.030
            and darkness >= 5.0          # a shadow darkens; a pale deposit does not
            and brightness <= 3.0
            and color <= 7.0
            and saturation_delta <= 8.0
            and texture <= 6.0
            and edge_strength <= 0.24
        )
        if looks_like_shadow or big_uniform_shadow:
            if config.suppress_shadows:
                rejected["shadow"] += 1
                continue
            notes.append("Soft edged and colourless - may be a shadow.")

        # (e) Tile texture: tiny *and* weakly anomalous. Small but strong marks
        #     are kept, so a genuine small stain still gets through.
        if area_ratio < config.texture_noise_area_ratio and anomaly_mean < 0.45:
            rejected["texture_noise"] += 1
            continue

        border_ratio = _border_fraction(region_mask, (x0, y0), (height, width),
                                        border_margin)
        touches_corner = (
            (x <= border_margin or x + w >= width - border_margin)
            and (y <= border_margin or y + h >= height - border_margin)
        )

        moments = cv2.moments(contour)
        if moments["m00"] > 0:
            center_x = int(moments["m10"] / moments["m00"])
            center_y = int(moments["m01"] / moments["m00"])
        else:  # pragma: no cover - degenerate contour
            center_x, center_y = x + w // 2, y + h // 2

        survivors.append({
            "area": area, "contour": contour, "fragments": fragment_count,
            "bbox": (x, y, w, h), "center": (center_x, center_y),
            "darkness": darkness, "brightness": brightness, "color": color,
            "saturation_delta": saturation_delta, "saturation_abs": saturation_abs,
            "texture": texture, "anomaly_mean": anomaly_mean, "hue": hue,
            "edge_strength": edge_strength, "circularity": circularity,
            "solidity": solidity, "elongation": elongation, "area_ratio": area_ratio,
            "shadowish": bool(looks_like_shadow or big_uniform_shadow),
            "border_ratio": border_ratio, "touches_corner": touches_corner,
            "joint_overlap": joint_overlap, "notes": notes,
        })

    # --- 5. CNN verdict -----------------------------------------------------
    classifier_used = "heuristic"
    classifier_note = ""
    predictions: List[Optional[Any]] = [None] * len(survivors)

    if not config.use_cnn:
        classifier_note = "Trained CNN disabled in the detector settings."
    else:
        if classifier is None:
            try:
                from ml_classifier import get_classifier
                classifier = get_classifier()
            except Exception as exc:  # pragma: no cover - defensive
                classifier = None
                classifier_note = f"Classifier unavailable: {exc}"

        if classifier is not None:
            if classifier.is_ready:
                # The CNN is in charge of this run even when there happened to
                # be nothing for it to judge, e.g. on a perfectly clean floor.
                classifier_used = "cnn"
                if survivors:
                    crops = [
                        classifier_crop(analysis_image, item["bbox"],
                                        classifier.input_size, config.crop_context)
                        for item in survivors
                    ]
                    predictions = classifier.predict_batch(crops)
                    if not any(p is not None for p in predictions):  # pragma: no cover
                        classifier_used = "heuristic"
                        classifier_note = (
                            classifier.error or "CNN returned no predictions."
                        )
            else:
                classifier_note = classifier.error or "Trained model not available."

    # --- 6. build the accepted detections ----------------------------------
    detections: List[Detection] = []

    for item, prediction in zip(survivors, predictions):
        notes: List[str] = list(item["notes"])
        contour = item["contour"]
        x, y, w, h = item["bbox"]

        if prediction is not None:
            # The CNN decides. Confidence is its softmax probability - no
            # heuristic scoring is mixed in.
            if not prediction.is_stain:
                rejected["cnn_not_a_stain"] += 1
                continue
            if prediction.confidence < config.min_stain_probability:
                rejected["cnn_low_confidence"] += 1
                continue

            confidence = clamp(float(prediction.confidence), 0.0, 1.0)
            stain_type = prediction.display_name
            source = "cnn"
            ml_class = prediction.class_name
            clean_probability = float(prediction.clean_probability)
            probabilities = dict(prediction.probabilities)
            notes.append(
                f"CNN predicted {prediction.class_name} at "
                f"{confidence * 100:.0f}% (clean {clean_probability * 100:.0f}%)."
            )
        else:
            # Classical fallback: the original heuristic score and label.
            confidence = _estimate_confidence(
                item["anomaly_mean"], item["edge_strength"],
                item["area_ratio"], item["solidity"],
            )
            if item["elongation"] >= 5.0:
                confidence *= 0.88
                notes.append("Elongated region - could follow a floor joint.")
            if item["shadowish"]:
                confidence *= 0.70
            confidence = clamp(confidence, 0.0, 0.99)
            if confidence < config.min_confidence:
                rejected["low_confidence"] += 1
                continue

            stain_type, reason = _estimate_stain_type(
                darkness=item["darkness"], brightness=item["brightness"],
                color=item["color"], saturation_delta=item["saturation_delta"],
                saturation_abs=item["saturation_abs"], texture=item["texture"],
                hue=item["hue"], circularity=item["circularity"],
                solidity=item["solidity"], area_ratio=item["area_ratio"],
                edge_strength=item["edge_strength"],
            )
            notes.append(f"Estimated type: {reason} (heuristic).")
            source = "heuristic"
            ml_class = ""
            clean_probability = 0.0
            probabilities = {}

        # --- image-border and corner rule ----------------------------------
        # Edges and corners of a photograph carry vignetting, the frame itself
        # and whatever the camera clipped. Only strong evidence gets through.
        on_border = item["border_ratio"] >= 0.55 or item["touches_corner"]
        if on_border and confidence < config.border_strong_probability:
            rejected["image_border"] += 1
            continue
        if on_border:
            notes.append("At the image edge, kept on strong classifier evidence.")

        if item["fragments"] > 1:
            notes.append(f"Merged from {item['fragments']} nearby fragments.")

        severity = _estimate_severity(confidence, item["area_ratio"])

        detections.append(
            Detection(
                id=0,  # assigned after sorting
                stain_type=stain_type,
                confidence=round(confidence, 3),
                severity=severity,
                area_px=int(item["area"]),
                area_ratio=item["area_ratio"],
                x=int(x),
                y=int(y),
                width=int(w),
                height=int(h),
                center_x=int(item["center"][0]),
                center_y=int(item["center"][1]),
                anomaly_score=round(item["anomaly_mean"], 3),
                darkness_delta=item["darkness"],
                brightness_delta=item["brightness"],
                color_delta=item["color"],
                saturation_delta=item["saturation_delta"],
                texture_delta=item["texture"],
                hue=item["hue"],
                edge_strength=item["edge_strength"],
                solidity=item["solidity"],
                circularity=item["circularity"],
                elongation=item["elongation"],
                contour=contour,
                notes=notes,
                classifier=source,
                ml_class=ml_class,
                ml_confidence=round(confidence, 4) if source == "cnn" else 0.0,
                clean_probability=round(clean_probability, 4),
                ml_probabilities=probabilities,
                merged_from=int(item["fragments"]),
                border_fraction=round(float(item["border_ratio"]), 3),
                joint_overlap=round(float(item["joint_overlap"]), 3),
            )
        )
        cv2.drawContours(accepted_mask, [contour], -1, 255, thickness=cv2.FILLED)

    # Stable, meaningful ordering: strongest and largest first.
    detections.sort(key=lambda d: (d.confidence * (0.4 + d.area_ratio)), reverse=True)
    for index, detection in enumerate(detections, start=1):
        detection.id = index

    if detections and all(d.classifier == "heuristic" for d in detections):
        classifier_used = "heuristic"

    return (detections, rejected, accepted_mask, classifier_used,
            classifier_note, candidates_considered)



# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------


def detect_stains(
    image_bgr: np.ndarray,
    config: Optional[DetectorConfig] = None,
    classifier: Optional[Any] = None,
) -> DetectionResult:
    """Run the full detection pipeline on a BGR image.

    Parameters
    ----------
    image_bgr:
        3-channel uint8 image (as returned by :func:`utils.load_image_from_bytes`).
    config:
        Detector parameters. Defaults are used when omitted.

    Returns
    -------
    DetectionResult
        Detections plus the mask, anomaly map and run statistics. An empty
        detection list is a valid, non-exceptional result.
    """
    if config is None:
        config = DetectorConfig()

    if image_bgr is None or not isinstance(image_bgr, np.ndarray):
        raise ValueError("detect_stains expects a NumPy image array.")
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError(f"Expected a 3-channel BGR image, got shape {image_bgr.shape}.")
    if image_bgr.dtype != np.uint8:
        image_bgr = np.clip(image_bgr, 0, 255).astype(np.uint8)

    started = time.perf_counter()
    original_height, original_width = image_bgr.shape[:2]

    # --- 1/2. resize only when the image is larger than needed -------------
    analysis_image, scale = resize_max_dimension(image_bgr, config.max_dimension)

    # --- 3-7. anomaly map --------------------------------------------------
    maps = build_feature_maps(analysis_image, config)

    # --- 8-10. binary mask -------------------------------------------------
    mask, threshold = build_mask(maps.anomaly, config)
    if config.suppress_grout:
        mask = suppress_line_structures(mask)

    # --- 11-16. candidates, filtering, CNN classification ------------------
    (detections, rejected, accepted_mask, classifier_used,
     classifier_note, candidates) = _extract_regions(
        mask, maps, config, analysis_image, classifier,
    )

    coverage = float(np.count_nonzero(accepted_mask)) / float(accepted_mask.size)
    elapsed = time.perf_counter() - started

    return DetectionResult(
        image_bgr=analysis_image,
        detections=detections,
        mask=accepted_mask,
        anomaly_map=maps.anomaly,
        scale=scale,
        original_size=(original_width, original_height),
        threshold_used=threshold,
        coverage_ratio=coverage,
        elapsed_seconds=elapsed,
        rejected=rejected,
        classifier_used=classifier_used,
        classifier_note=classifier_note,
        candidates_considered=candidates,
    )


# --------------------------------------------------------------------------
# Drawing helpers
# --------------------------------------------------------------------------


def _color_for(detection: Detection) -> Tuple[int, int, int]:
    """BGR colour for a detection, based on severity."""
    return SEVERITY_COLORS.get(detection.severity, DEFAULT_COLOR)


def draw_detections(
    image_bgr: np.ndarray,
    detections: Sequence[Detection],
    show_boxes: bool = True,
    show_contours: bool = True,
    show_labels: bool = True,
    show_confidence: bool = True,
) -> np.ndarray:
    """Return a copy of the image annotated with detection overlays."""
    canvas = image_bgr.copy()
    if not detections:
        return canvas

    height, width = canvas.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = clamp(min(width, height) / 1900.0, 0.36, 0.62)
    thickness = max(1, int(round(font_scale * 2)))
    line_thickness = max(1, int(round(min(width, height) / 500.0)))
    placed_boxes: List[Tuple[int, int, int, int]] = []

    if show_contours:
        # Translucent fill makes the affected area readable without hiding it.
        fill = canvas.copy()
        for detection in detections:
            cv2.drawContours(fill, [detection.contour], -1, _color_for(detection),
                             thickness=cv2.FILLED)
        canvas = cv2.addWeighted(fill, 0.16, canvas, 0.84, 0.0)
        for detection in detections:
            cv2.drawContours(canvas, [detection.contour], -1, _color_for(detection),
                             thickness=line_thickness)

    for detection in detections:
        color = _color_for(detection)
        x, y, w, h = detection.bbox

        if show_boxes:
            cv2.rectangle(canvas, (x, y), (x + w, y + h), color, line_thickness + 1)

        if not show_labels:
            continue

        lines = [f"{detection.id:02d}  {detection.stain_type}"]
        if show_confidence:
            lines.append(f"Confidence: {detection.confidence * 100:.0f}%")
        lines.append(f"Severity: {detection.severity}")

        _draw_label(canvas, lines, (x, y, w, h), color, font, font_scale, thickness,
                    placed_boxes)

    return canvas


def _boxes_overlap(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> bool:
    """True when two ``(x1, y1, x2, y2)`` rectangles intersect."""
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _draw_label(
    canvas: np.ndarray,
    lines: Sequence[str],
    box: Tuple[int, int, int, int],
    color: Tuple[int, int, int],
    font: int,
    font_scale: float,
    thickness: int,
    placed_boxes: Optional[List[Tuple[int, int, int, int]]] = None,
) -> None:
    """Draw a filled multi-line label near a bounding box, avoiding overlaps."""
    height, width = canvas.shape[:2]
    padding = 5
    sizes = [cv2.getTextSize(line, font, font_scale, thickness)[0] for line in lines]
    box_w = max(size[0] for size in sizes) + 2 * padding
    line_h = max(size[1] for size in sizes) + 5
    box_h = line_h * len(lines) + padding

    x, y, w, h = box
    # Candidate anchors: above, below, inside-top, right of the detection.
    candidates = [
        (x, y - box_h - 4),
        (x, y + h + 4),
        (x + 2, y + 2),
        (x + w + 4, y),
        (x - box_w - 4, y),
    ]

    chosen = None
    for cand_x, cand_y in candidates:
        left = int(min(max(0, cand_x), max(0, width - box_w - 1)))
        top = int(min(max(0, cand_y), max(0, height - box_h - 1)))
        rect = (left, top, left + box_w, top + box_h)
        if placed_boxes is None or not any(_boxes_overlap(rect, other) for other in placed_boxes):
            chosen = rect
            break

    if chosen is None:  # everything collides - fall back to just above the box
        left = int(min(max(0, x), max(0, width - box_w - 1)))
        top = int(min(max(0, y - box_h - 4), max(0, height - box_h - 1)))
        top = max(0, top)
        chosen = (left, top, left + box_w, top + box_h)

    left, top = chosen[0], chosen[1]
    if placed_boxes is not None:
        placed_boxes.append(chosen)

    # Leader line from the label to the detection box when they are apart.
    cv2.line(canvas, (left + box_w // 2, top + box_h), (x + w // 2, y + h // 2),
             color, 1, cv2.LINE_AA)

    overlay = canvas.copy()
    cv2.rectangle(overlay, (left, top), (left + box_w, top + box_h), color, cv2.FILLED)
    cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0.0, dst=canvas)
    cv2.rectangle(canvas, (left, top), (left + box_w, top + box_h), (255, 255, 255), 1)

    text_y = top + line_h - 4
    for line in lines:
        cv2.putText(canvas, line, (left + padding, text_y), font, font_scale,
                    (255, 255, 255), thickness, cv2.LINE_AA)
        text_y += line_h


def render_mask_view(result: DetectionResult, colorise: bool = True) -> np.ndarray:
    """Render the detection mask as a viewable BGR image.

    When ``colorise`` is enabled the anomaly map is shown as a heat map with
    the accepted detection mask outlined on top.
    """
    if not colorise:
        return cv2.cvtColor(result.mask, cv2.COLOR_GRAY2BGR)

    heat = np.clip(result.anomaly_map * 255.0, 0, 255).astype(np.uint8)
    heat_bgr = cv2.applyColorMap(heat, cv2.COLORMAP_INFERNO)

    # Dim everything that was not accepted as a detection.
    accepted = result.mask > 0
    dimmed = (heat_bgr.astype(np.float32) * 0.45).astype(np.uint8)
    view = np.where(accepted[:, :, None], heat_bgr, dimmed)

    contours, _ = cv2.findContours(result.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(view, contours, -1, (255, 255, 255), 1)
    return view


# --------------------------------------------------------------------------
# Backwards compatibility
# --------------------------------------------------------------------------
# The trained classifier is no longer optional plumbing bolted onto the
# heuristic detector: it is part of the pipeline. Its implementation now lives
# in ml_classifier.StainClassifier, which this alias points at so that any
# older script importing the previous name keeps working.

from ml_classifier import StainClassifier as OptionalStainClassifier  # noqa: E402,F401
