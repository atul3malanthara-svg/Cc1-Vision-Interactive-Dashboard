"""
CC1 Vision - synthetic labelled dataset builder.

Creates the training material for the CNN stain classifier by reusing the same
floor and stain primitives that ``sample_generator.py`` uses for the demo
image. Everything is generated locally with NumPy and OpenCV; nothing is
downloaded and no external dataset is required.

    python dataset_generator.py --per-class 900

Output layout (image folders, one sub-folder per class)::

    dataset/
        train/ clean/ mud_dirt/ liquid_spill/ grease_oil/ dark_stain/ scattered_dirt/
        val/   ... same six classes ...
        test/  ... same six classes ...
        dataset_summary.json

Why the ``clean`` class matters
-------------------------------
The detector's OpenCV stage proposes *candidate* regions. Many candidates are
not stains at all: tile grout lines, grout junctions, lamp reflections, soft
shadows, plain speckled floor and image-border artifacts. Those are exactly the
sub-types generated for the ``clean`` class, so the CNN learns to veto them.
That veto is what removes the false positives, rather than a hand-tuned rule.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import cv2
import numpy as np

from sample_generator import apply_lighting, apply_stain, blob_alpha, build_tiled_floor
from utils import ensure_directory

# --------------------------------------------------------------------------
# Dataset definition
# --------------------------------------------------------------------------

#: Class folder names, in the order the model reports them.
CLASS_NAMES: Tuple[str, ...] = (
    "clean",
    "mud_dirt",
    "liquid_spill",
    "grease_oil",
    "dark_stain",
    "scattered_dirt",
)

#: Which classes count as real soiling (everything except ``clean``).
STAIN_CLASSES: Tuple[str, ...] = tuple(name for name in CLASS_NAMES if name != "clean")

#: Sub-types generated inside the ``clean`` class. These are the distractors
#: that used to become false detections.
CLEAN_SUBTYPES: Tuple[str, ...] = (
    "plain",
    "grout_line",
    "grout_junction",
    "glare",
    "shadow",
    "speckle",
    "image_border",
)

#: Side length of a stored crop, in pixels. Also the CNN input size.
CROP_SIZE: int = 96

#: Working resolution before the final downscale (keeps edges anti-aliased).
BUILD_SIZE: int = 192

PROJECT_ROOT: Path = Path(__file__).resolve().parent
DATASET_DIR: Path = PROJECT_ROOT / "dataset"

#: train / val / test proportions.
SPLITS: Tuple[Tuple[str, float], ...] = (("train", 0.70), ("val", 0.15), ("test", 0.15))


# --------------------------------------------------------------------------
# Background floor patches
# --------------------------------------------------------------------------

#: Plausible tile base colours in BGR (grey, warm grey, cream, beige, blue-grey,
#: light brown, near-white). Variety here stops the CNN from keying on one hue.
TILE_COLORS: Tuple[Tuple[float, float, float], ...] = (
    (176.0, 178.0, 181.0),
    (168.0, 172.0, 178.0),
    (162.0, 174.0, 186.0),
    (150.0, 162.0, 176.0),
    (186.0, 186.0, 184.0),
    (196.0, 198.0, 198.0),
    (142.0, 150.0, 158.0),
    (158.0, 166.0, 172.0),
    (172.0, 182.0, 192.0),
    (130.0, 138.0, 146.0),
)

#: The base colour ``sample_generator.build_tiled_floor`` paints with.
_SOURCE_TILE_COLOR = np.array([176.0, 178.0, 181.0], dtype=np.float32)


def _tinted_floor(rng: np.random.Generator, size: int) -> np.ndarray:
    """Build a tiled floor canvas that is larger than one crop.

    The canvas is oversized so a later random crop can land on plain tile, on a
    grout line or on a grout junction, which gives natural positional variety.
    """
    canvas = int(size * 2.4)
    tile = int(rng.integers(int(size * 0.55), int(size * 1.9)))
    floor = build_tiled_floor(canvas, canvas, rng, tile_size=max(24, tile))

    # ``build_tiled_floor`` works around one fixed grey. Re-tint it to a random
    # tile colour while keeping its grout lines, speckle and per-tile shading.
    target = np.array(TILE_COLORS[int(rng.integers(0, len(TILE_COLORS)))], dtype=np.float32)
    target = target * float(rng.uniform(0.88, 1.12))
    floor = floor * (target / _SOURCE_TILE_COLOR)

    # Extra floor texture: some tiles are near-uniform, some are granite-like.
    grain = float(rng.uniform(0.4, 3.4))
    floor = floor + rng.normal(0.0, grain, size=(canvas, canvas, 1)).astype(np.float32)
    return floor


def _crop_window(
    floor: np.ndarray,
    rng: np.random.Generator,
    size: int,
    mode: str = "any",
) -> np.ndarray:
    """Take a ``size`` x ``size`` window out of an oversized floor canvas.

    ``mode`` steers where the window lands so the ``clean`` class can request a
    grout line or a grout junction on purpose.
    """
    canvas = floor.shape[0]
    limit = max(1, canvas - size)

    if mode in ("grout_line", "grout_junction"):
        # Grout pixels are markedly darker than their tile; find them and centre
        # the window on one so the joint runs through the middle of the crop.
        grey = floor.mean(axis=2)
        dark = grey < (float(np.median(grey)) - 12.0)
        interior = dark[size // 2:limit + size // 2, size // 2:limit + size // 2]
        ys, xs = np.nonzero(interior)
        if ys.size:
            pick = int(rng.integers(0, ys.size))
            top = int(np.clip(ys[pick], 0, limit))
            left = int(np.clip(xs[pick], 0, limit))
            if mode == "grout_junction":
                # Nudge towards a corner where a horizontal and a vertical joint
                # meet; there, both row and column coverage are high at once.
                best, best_score = (top, left), -1.0
                for _ in range(24):
                    ty = int(np.clip(top + rng.integers(-size, size), 0, limit))
                    tx = int(np.clip(left + rng.integers(-size, size), 0, limit))
                    patch = dark[ty:ty + size, tx:tx + size]
                    rows = float(np.any(patch, axis=1).mean())
                    cols = float(np.any(patch, axis=0).mean())
                    score = min(rows, cols)
                    if score > best_score:
                        best_score, best = score, (ty, tx)
                top, left = best
            return floor[top:top + size, left:left + size].copy()

    top = int(rng.integers(0, limit))
    left = int(rng.integers(0, limit))
    return floor[top:top + size, left:left + size].copy()


# --------------------------------------------------------------------------
# Stain painters - one per class
# --------------------------------------------------------------------------


def _centre(rng: np.random.Generator, size: int, spread: float = 0.13) -> Tuple[int, int]:
    """A slightly off-centre position, mimicking imperfect region cropping."""
    jitter = size * spread
    return (
        int(np.clip(size / 2 + rng.normal(0, jitter), size * 0.25, size * 0.75)),
        int(np.clip(size / 2 + rng.normal(0, jitter), size * 0.25, size * 0.75)),
    )


def _paint_mud_dirt(patch: np.ndarray, rng: np.random.Generator, size: int) -> np.ndarray:
    """Brown, granular, irregular soiling with satellite splashes.

    One crop in three is a *faint, wide* patch rather than a fresh dark one:
    built-up grime covering most of the crop at low contrast. Without those,
    the network learns that soiling must be strong and small, and then rejects
    exactly the dirty-floor areas a cleaning robot most needs to see.
    """
    faint = bool(rng.random() < 0.33)
    centre = _centre(rng, size)
    if faint:
        radius = int(size * rng.uniform(0.32, 0.52))
        strength = float(rng.uniform(0.18, 0.42))
        softness = float(rng.uniform(0.40, 0.62))
    else:
        radius = int(size * rng.uniform(0.20, 0.36))
        strength = float(rng.uniform(0.52, 0.88))
        softness = float(rng.uniform(0.30, 0.48))

    alpha = blob_alpha((size, size), centre, radius, rng,
                       lobes=int(rng.integers(8, 14)), softness=softness)
    if faint:
        # Built-up soiling is blotchy, not an even wash. The mottling is what
        # separates it from a shadow, which has no texture of its own.
        mottle = rng.normal(0.0, 1.0, size=(max(4, size // 20),) * 2).astype(np.float32)
        mottle = cv2.resize(mottle, (size, size), interpolation=cv2.INTER_CUBIC)
        alpha = np.clip(alpha * (1.0 + 0.45 * mottle), 0.0, 1.0)

    color = (float(rng.uniform(34, 74)), float(rng.uniform(64, 104)), float(rng.uniform(104, 156)))
    patch = apply_stain(patch, alpha, color, strength, rng,
                        texture=float(rng.uniform(6.0, 13.0)))
    if faint:
        return patch

    # Splashes around the main patch: mud never lands as one clean disc.
    for _ in range(int(rng.integers(2, 7))):
        angle = float(rng.uniform(0, 2 * np.pi))
        distance = radius * float(rng.uniform(0.9, 1.9))
        spot = (int(centre[0] + np.cos(angle) * distance),
                int(centre[1] + np.sin(angle) * distance))
        spot_alpha = blob_alpha((size, size), spot,
                                max(3, int(radius * rng.uniform(0.12, 0.32))),
                                rng, lobes=4, softness=0.34)
        patch = apply_stain(patch, spot_alpha, color, float(rng.uniform(0.45, 0.80)), rng,
                            texture=float(rng.uniform(4.0, 10.0)))
    return patch


def _paint_liquid_spill(patch: np.ndarray, rng: np.random.Generator, size: int) -> np.ndarray:
    """Darker, smooth, rounded wet patch with a bright wet rim."""
    centre = _centre(rng, size)
    radius = int(size * rng.uniform(0.20, 0.36))
    alpha = blob_alpha((size, size), centre, radius, rng,
                       lobes=int(rng.integers(3, 7)), softness=float(rng.uniform(0.16, 0.28)))
    base = float(rng.uniform(52, 96))
    color = (base + float(rng.uniform(-6, 14)),
             base + float(rng.uniform(-6, 10)),
             base + float(rng.uniform(-8, 8)))
    patch = apply_stain(patch, alpha, color, float(rng.uniform(0.45, 0.78)), rng,
                        texture=float(rng.uniform(0.0, 2.2)))

    # A wet edge catches the light: thin bright ring just inside the boundary.
    ring = cv2.morphologyEx((alpha > 0.45).astype(np.uint8) * 255, cv2.MORPH_GRADIENT,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    ring = cv2.GaussianBlur(ring.astype(np.float32) / 255.0, (0, 0), 1.6)
    return patch + (ring * float(rng.uniform(14.0, 34.0)))[:, :, None]


def _paint_grease_oil(patch: np.ndarray, rng: np.random.Generator, size: int) -> np.ndarray:
    """Dark, glossy, smeared film with broad specular highlights inside it."""
    centre = _centre(rng, size)
    radius = int(size * rng.uniform(0.20, 0.34))
    alpha = blob_alpha((size, size), centre, radius, rng,
                       lobes=int(rng.integers(5, 10)), softness=float(rng.uniform(0.28, 0.46)))

    # Smear the blob along one axis: grease is wiped, not poured.
    angle = float(rng.uniform(0, 180))
    stretch = cv2.getRotationMatrix2D((size / 2.0, size / 2.0), angle, 1.0)
    alpha = cv2.warpAffine(alpha, stretch, (size, size), borderMode=cv2.BORDER_REFLECT101)
    alpha = cv2.GaussianBlur(alpha, (0, 0), float(rng.uniform(1.5, 4.0)), 0.8)
    if alpha.max() > 0:
        alpha = alpha / alpha.max()

    tint = float(rng.uniform(0.0, 26.0))          # slightly warm/yellow film
    base = float(rng.uniform(34, 70))
    color = (base, base + tint * 0.55, base + tint)
    patch = apply_stain(patch, alpha, color, float(rng.uniform(0.58, 0.86)), rng,
                        texture=float(rng.uniform(0.0, 1.8)))

    # Gloss: several soft bright lobes sitting on top of the dark film.
    gloss = np.zeros((size, size), dtype=np.float32)
    for _ in range(int(rng.integers(2, 5))):
        gx = int(np.clip(centre[0] + rng.normal(0, radius * 0.55), 0, size - 1))
        gy = int(np.clip(centre[1] + rng.normal(0, radius * 0.55), 0, size - 1))
        cv2.circle(gloss, (gx, gy), max(3, int(radius * rng.uniform(0.18, 0.42))),
                   1.0, cv2.FILLED, cv2.LINE_AA)
    gloss = cv2.GaussianBlur(gloss, (0, 0), max(0.6, radius * 0.20)) * alpha
    if gloss.max() > 0:
        gloss = gloss / gloss.max()
    return patch + (gloss * float(rng.uniform(30.0, 62.0)))[:, :, None]


def _paint_dark_stain(patch: np.ndarray, rng: np.random.Generator, size: int) -> np.ndarray:
    """Deep, matte, set-in discolouration - no gloss, often with a dry halo."""
    centre = _centre(rng, size)
    radius = int(size * rng.uniform(0.18, 0.33))
    alpha = blob_alpha((size, size), centre, radius, rng,
                       lobes=int(rng.integers(4, 9)), softness=float(rng.uniform(0.18, 0.34)))
    level = float(rng.uniform(24, 62))
    color = (level + float(rng.uniform(-6, 6)),
             level + float(rng.uniform(-6, 6)),
             level + float(rng.uniform(-6, 6)))
    # Older, partly cleaned stains are much fainter than fresh ones.
    patch = apply_stain(patch, alpha, color, float(rng.uniform(0.34, 0.90)), rng,
                        texture=float(rng.uniform(1.0, 4.0)))

    # Set-in stains often leave a slightly darker dried ring around the core.
    if rng.random() < 0.55:
        halo = cv2.dilate((alpha > 0.35).astype(np.uint8) * 255,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        halo = cv2.GaussianBlur(halo.astype(np.float32) / 255.0, (0, 0), 3.0)
        halo = np.clip(halo - alpha, 0.0, 1.0)
        patch = patch - (halo * float(rng.uniform(6.0, 20.0)))[:, :, None]
    return patch


def _paint_scattered_dirt(patch: np.ndarray, rng: np.random.Generator, size: int) -> np.ndarray:
    """Many small dry particles spread across the crop - crumbs, grit, dust.

    One crop in four is instead a dry *streak*: the scuff or dust smear a shoe
    or a trolley wheel leaves behind. It is low contrast and often lighter than
    the floor, so it needs to be represented explicitly or it reads as clean.
    """
    if rng.random() < 0.25:
        return _paint_dust_streak(patch, rng, size)

    count = int(rng.integers(28, 130))
    dark = bool(rng.random() < 0.78)
    speck = np.zeros((size, size), dtype=np.float32)
    for _ in range(count):
        px = int(rng.integers(0, size))
        py = int(rng.integers(0, size))
        rx = max(1, int(rng.uniform(1.0, size * 0.035)))
        ry = max(1, int(rx * rng.uniform(0.5, 1.6)))
        speck[:] = 0.0
        cv2.ellipse(speck, (px, py), (rx, ry), float(rng.uniform(0, 180)),
                    0, 360, 1.0, cv2.FILLED, cv2.LINE_AA)
        speck = cv2.GaussianBlur(speck, (0, 0), 0.8)
        if speck.max() > 0:
            speck = speck / speck.max()
        if dark:
            tone = float(rng.uniform(38, 108))
            color = (tone, tone + float(rng.uniform(0, 22)), tone + float(rng.uniform(0, 34)))
        else:                                    # pale dust and lint
            tone = float(rng.uniform(196, 236))
            color = (tone, tone, tone)
        patch = apply_stain(patch, speck, color, float(rng.uniform(0.42, 0.92)), rng)

    # A faint soiling wash often accompanies scattered debris.
    if rng.random() < 0.5:
        wash = blob_alpha((size, size), _centre(rng, size, 0.20),
                          int(size * rng.uniform(0.30, 0.48)), rng, lobes=10, softness=0.60)
        tone = float(rng.uniform(120, 160))
        patch = apply_stain(patch, wash, (tone, tone + 4, tone + 10),
                            float(rng.uniform(0.10, 0.26)), rng, texture=3.0)
    return patch


def _paint_dust_streak(patch: np.ndarray, rng: np.random.Generator, size: int) -> np.ndarray:
    """A dry, grainy, elongated deposit - a dust smear or a shoe scuff."""
    centre = _centre(rng, size, 0.16)
    streak = np.zeros((size, size), dtype=np.float32)
    cv2.ellipse(streak, centre,
                (int(size * rng.uniform(0.28, 0.48)), int(size * rng.uniform(0.05, 0.13))),
                float(rng.uniform(0, 180)), 0, 360, 1.0, cv2.FILLED, cv2.LINE_AA)
    streak = cv2.GaussianBlur(streak, (0, 0), size * float(rng.uniform(0.03, 0.07)))
    if streak.max() > 0:
        streak = streak / streak.max()

    # Grainy and uneven, unlike the smooth ramp of a shadow.
    mottle = rng.normal(0.0, 1.0, size=(max(4, size // 18),) * 2).astype(np.float32)
    mottle = cv2.resize(mottle, (size, size), interpolation=cv2.INTER_CUBIC)
    streak = np.clip(streak * (1.0 + 0.50 * mottle), 0.0, 1.0)

    if rng.random() < 0.6:                        # pale dust
        tone = float(rng.uniform(198, 240))
        color = (tone, tone, tone)
    else:                                         # dark rubber scuff
        tone = float(rng.uniform(80, 130))
        color = (tone, tone + float(rng.uniform(0, 12)), tone + float(rng.uniform(0, 16)))

    return apply_stain(patch, streak, color, float(rng.uniform(0.35, 0.85)), rng,
                       texture=float(rng.uniform(2.5, 6.0)))


def _paint_clean(patch: np.ndarray, rng: np.random.Generator, size: int,
                 subtype: str) -> np.ndarray:
    """Add a *distractor* rather than a stain: the things we must not detect."""
    if subtype == "glare":
        gx, gy = _centre(rng, size, 0.18)
        glare = np.zeros((size, size), dtype=np.float32)
        cv2.circle(glare, (gx, gy), int(size * rng.uniform(0.16, 0.40)),
                   1.0, cv2.FILLED, cv2.LINE_AA)
        glare = cv2.GaussianBlur(glare, (0, 0), size * float(rng.uniform(0.08, 0.16)))
        patch = patch + (glare * float(rng.uniform(26.0, 70.0)))[:, :, None]

    elif subtype == "shadow":
        # A large, uniform, soft-edged darkening - no colour and no texture.
        shadow = np.zeros((size, size), dtype=np.float32)
        if rng.random() < 0.5:
            cut = int(size * rng.uniform(0.25, 0.75))
            if rng.random() < 0.5:
                shadow[:, :cut] = 1.0
            else:
                shadow[:, cut:] = 1.0
        else:
            cv2.circle(shadow, _centre(rng, size, 0.3),
                       int(size * rng.uniform(0.40, 0.80)), 1.0, cv2.FILLED, cv2.LINE_AA)
        shadow = cv2.GaussianBlur(shadow, (0, 0), size * float(rng.uniform(0.10, 0.22)))
        patch = patch * (1.0 - float(rng.uniform(0.10, 0.30)) * shadow[:, :, None])

    elif subtype == "speckle":
        heavy = rng.normal(0.0, float(rng.uniform(5.0, 13.0)),
                           size=(size, size)).astype(np.float32)
        patch = patch + cv2.GaussianBlur(heavy, (0, 0), 0.9)[:, :, None]

    elif subtype == "image_border":
        # Photo edges, vignettes and letterboxing look like a huge dark region.
        band = int(size * rng.uniform(0.10, 0.30))
        side = int(rng.integers(0, 4))
        tone = float(rng.uniform(0, 70))
        if side == 0:
            patch[:band, :] = tone
        elif side == 1:
            patch[-band:, :] = tone
        elif side == 2:
            patch[:, :band] = tone
        else:
            patch[:, -band:] = tone
        patch = cv2.GaussianBlur(patch, (0, 0), 1.2)

    # "plain", "grout_line" and "grout_junction" need nothing extra: the crop
    # window itself was already chosen to contain that structure.
    return patch


PAINTERS = {
    "mud_dirt": _paint_mud_dirt,
    "liquid_spill": _paint_liquid_spill,
    "grease_oil": _paint_grease_oil,
    "dark_stain": _paint_dark_stain,
    "scattered_dirt": _paint_scattered_dirt,
}


# --------------------------------------------------------------------------
# Capture conditions
# --------------------------------------------------------------------------


def _apply_capture_conditions(patch: np.ndarray, rng: np.random.Generator,
                              size: int) -> np.ndarray:
    """Simulate the camera: rotation, lighting, blur, contrast, noise, JPEG."""
    # Rotation, with reflected borders so no black corners are introduced.
    angle = float(rng.uniform(0, 360))
    matrix = cv2.getRotationMatrix2D((size / 2.0, size / 2.0), angle,
                                     float(rng.uniform(1.0, 1.18)))
    patch = cv2.warpAffine(patch, matrix, (size, size), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REFLECT101)

    if rng.random() < 0.5:
        patch = np.ascontiguousarray(patch[:, ::-1])

    patch = apply_lighting(patch, rng)

    # Global exposure and contrast drift between photographs.
    patch = patch * float(rng.uniform(0.74, 1.24)) + float(rng.uniform(-16.0, 16.0))
    mean = float(patch.mean())
    patch = (patch - mean) * float(rng.uniform(0.78, 1.26)) + mean

    blur = float(rng.uniform(0.0, 1.7))
    if blur > 0.25:
        patch = cv2.GaussianBlur(patch, (0, 0), blur)

    patch = patch + rng.normal(0.0, float(rng.uniform(0.5, 5.5)),
                               size=patch.shape).astype(np.float32)

    image = np.clip(patch, 0, 255).astype(np.uint8)
    image = cv2.resize(image, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_AREA)

    # Compression artifacts, as in a real uploaded photo.
    if rng.random() < 0.55:
        quality = int(rng.integers(45, 96))
        ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if ok:
            image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    return image


def make_sample(class_name: str, rng: np.random.Generator) -> np.ndarray:
    """Generate one labelled ``CROP_SIZE`` x ``CROP_SIZE`` BGR crop."""
    size = BUILD_SIZE
    floor = _tinted_floor(rng, size)

    if class_name == "clean":
        subtype = CLEAN_SUBTYPES[int(rng.integers(0, len(CLEAN_SUBTYPES)))]
        window = subtype if subtype in ("grout_line", "grout_junction") else "any"
        patch = _crop_window(floor, rng, size, mode=window)
        patch = _paint_clean(patch, rng, size, subtype)
    else:
        # A stain crop may also contain a grout line; the CNN must not care.
        window = "grout_line" if rng.random() < 0.30 else "any"
        patch = _crop_window(floor, rng, size, mode=window)
        patch = PAINTERS[class_name](patch, rng, size)

    return _apply_capture_conditions(patch, rng, size)


# --------------------------------------------------------------------------
# Dataset assembly
# --------------------------------------------------------------------------


def build_dataset(
    output_dir: Path = DATASET_DIR,
    per_class: int = 900,
    seed: int = 17,
    clean_multiplier: float = 1.5,
    overwrite: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Generate the full train/val/test dataset on disk.

    ``per_class`` is the *total* number of crops per stain class across all
    three splits. The ``clean`` class is over-sampled by ``clean_multiplier``
    because it has to cover seven visually different distractor sub-types.
    """
    output_dir = Path(output_dir)
    if overwrite and output_dir.exists():
        shutil.rmtree(output_dir)

    rng = np.random.default_rng(int(seed))
    counts: Dict[str, Dict[str, int]] = {split: {} for split, _ in SPLITS}

    for class_name in CLASS_NAMES:
        total = int(per_class * (clean_multiplier if class_name == "clean" else 1.0))

        # Deterministic split sizes; the remainder goes to train.
        sizes: Dict[str, int] = {}
        assigned = 0
        for split, fraction in SPLITS[1:]:
            sizes[split] = max(1, int(round(total * fraction)))
            assigned += sizes[split]
        sizes[SPLITS[0][0]] = max(1, total - assigned)

        for split, _ in SPLITS:
            count = sizes[split]
            folder = ensure_directory(output_dir / split / class_name)
            for index in range(count):
                image = make_sample(class_name, rng)
                path = folder / f"{class_name}_{index:05d}.jpg"
                cv2.imwrite(str(path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
            counts[split][class_name] = count

        if verbose:
            written = ", ".join(f"{split}={sizes[split]}" for split, _ in SPLITS)
            print(f"  {class_name:<16} {written}")

    summary: Dict[str, Any] = {
        "classes": list(CLASS_NAMES),
        "stain_classes": list(STAIN_CLASSES),
        "clean_subtypes": list(CLEAN_SUBTYPES),
        "crop_size": CROP_SIZE,
        "seed": int(seed),
        "counts": counts,
        "totals": {split: sum(values.values()) for split, values in counts.items()},
    }
    ensure_directory(output_dir)
    with open(output_dir / "dataset_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    if verbose:
        print(f"  total: {sum(summary['totals'].values())} crops in {output_dir}")
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the CC1 Vision synthetic stain dataset."
    )
    parser.add_argument("--output", type=str, default=str(DATASET_DIR),
                        help="Dataset root folder (default: dataset/)")
    parser.add_argument("--per-class", type=int, default=900,
                        help="Total crops per stain class across all splits")
    parser.add_argument("--seed", type=int, default=17, help="Random seed")
    parser.add_argument("--keep", action="store_true",
                        help="Add to an existing dataset instead of replacing it")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    print("CC1 Vision - building synthetic stain dataset")
    build_dataset(
        output_dir=Path(args.output),
        per_class=int(args.per_class),
        seed=int(args.seed),
        overwrite=not args.keep,
    )
    print("Done. Train with:  python train_classifier.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
