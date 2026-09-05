"""
CC1 Vision - synthetic test floor generator.

Creates a tiled floor image containing several visually different stain-like
areas so the detector can be tested immediately, without hunting for photos.

The generated floor deliberately also contains *distractors* that a naive
detector would flag:

    * grout joints between tiles          -> tests line suppression
    * a soft shadow band across a corner  -> tests shadow suppression
    * a specular highlight                -> tests glare suppression
    * per-tile colour variation and noise -> tests the local background model

Run it directly:

    python sample_generator.py
    python sample_generator.py --output samples/floor_02.jpg --seed 21
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from utils import SAMPLES_DIR, ensure_directory

DEFAULT_WIDTH: int = 1280
DEFAULT_HEIGHT: int = 860
DEFAULT_SEED: int = 7


# --------------------------------------------------------------------------
# Floor background
# --------------------------------------------------------------------------


def _build_tiled_floor(
    width: int,
    height: int,
    rng: np.random.Generator,
    tile_size: int = 220,
) -> np.ndarray:
    """Build a tiled floor base in BGR float32 (0..255)."""
    base_color = np.array([176.0, 178.0, 181.0], dtype=np.float32)  # cool light grey
    floor = np.zeros((height, width, 3), dtype=np.float32)
    floor[:, :] = base_color

    grout = 6
    for top in range(0, height + tile_size, tile_size):
        for left in range(0, width + tile_size, tile_size):
            y0, y1 = top, min(height, top + tile_size - grout)
            x0, x1 = left, min(width, left + tile_size - grout)
            if y0 >= y1 or x0 >= x1:
                continue
            # Each tile has a slightly different shade and a faint gradient.
            shade = rng.normal(0.0, 4.5)
            tile = floor[y0:y1, x0:x1]
            gradient = np.linspace(-2.5, 2.5, tile.shape[0], dtype=np.float32)[:, None, None]
            floor[y0:y1, x0:x1] = tile + shade + gradient

            # Darker grout joint on the right/bottom edge of the tile.
            if x1 + grout <= width:
                floor[y0:y1, x1:x1 + grout] = base_color - 34.0
            if y1 + grout <= height:
                floor[y1:y1 + grout, x0:min(width, x1 + grout)] = base_color - 34.0

    # Fine speckle texture, as found on real vinyl / granite tiles.
    speckle = rng.normal(0.0, 3.2, size=(height, width, 1)).astype(np.float32)
    floor += speckle

    coarse = rng.normal(0.0, 9.0, size=(height // 8, width // 8, 1)).astype(np.float32)
    coarse = cv2.resize(coarse, (width, height), interpolation=cv2.INTER_CUBIC)
    floor += coarse[:, :, None] * 0.35

    return floor


def _apply_lighting(floor: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply an uneven lighting gradient plus one specular highlight."""
    height, width = floor.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)

    # Broad diagonal illumination gradient.
    gradient = 1.0 + 0.10 * ((xx / width) - 0.5) - 0.12 * ((yy / height) - 0.5)
    floor = floor * gradient[:, :, None]

    # Specular highlight (a lamp reflection) - a glare distractor.
    cx, cy = int(width * 0.78), int(height * 0.22)
    radius = int(min(width, height) * 0.13)
    highlight = np.zeros((height, width), dtype=np.float32)
    cv2.circle(highlight, (cx, cy), radius, 1.0, cv2.FILLED, cv2.LINE_AA)
    highlight = cv2.GaussianBlur(highlight, (0, 0), radius * 0.45)
    floor = floor + highlight[:, :, None] * 34.0

    # Soft shadow band in the lower-left corner - a shadow distractor.
    shadow = np.zeros((height, width), dtype=np.float32)
    points = np.array(
        [[0, int(height * 0.55)], [int(width * 0.30), height], [0, height]], dtype=np.int32
    )
    cv2.fillPoly(shadow, [points], 1.0, cv2.LINE_AA)
    shadow = cv2.GaussianBlur(shadow, (0, 0), min(width, height) * 0.05)
    floor = floor * (1.0 - 0.16 * shadow[:, :, None])

    return floor


# --------------------------------------------------------------------------
# Stain shapes
# --------------------------------------------------------------------------


def _blob_alpha(
    shape: Tuple[int, int],
    center: Tuple[int, int],
    radius: int,
    rng: np.random.Generator,
    lobes: int = 7,
    softness: float = 0.35,
) -> np.ndarray:
    """Create a soft, irregular blob alpha mask in the range 0..1."""
    height, width = shape
    alpha = np.zeros((height, width), dtype=np.float32)

    axes = (max(4, int(radius * rng.uniform(0.8, 1.2))),
            max(4, int(radius * rng.uniform(0.6, 1.0))))
    angle = float(rng.uniform(0, 180))
    cv2.ellipse(alpha, center, axes, angle, 0, 360, 1.0, cv2.FILLED, cv2.LINE_AA)

    # Extra lobes make the outline irregular, like a real spill.
    for _ in range(lobes):
        offset_angle = rng.uniform(0, 2 * np.pi)
        offset_radius = radius * rng.uniform(0.35, 1.0)
        lobe_center = (
            int(center[0] + np.cos(offset_angle) * offset_radius),
            int(center[1] + np.sin(offset_angle) * offset_radius),
        )
        lobe_radius = int(radius * rng.uniform(0.25, 0.6))
        cv2.circle(alpha, lobe_center, max(3, lobe_radius), 1.0, cv2.FILLED, cv2.LINE_AA)

    blur = max(3, int(radius * softness))
    alpha = cv2.GaussianBlur(alpha, (0, 0), blur)
    peak = float(alpha.max())
    if peak > 0:
        alpha /= peak
    return np.clip(alpha, 0.0, 1.0)


def _apply_stain(
    floor: np.ndarray,
    alpha: np.ndarray,
    color_bgr: Tuple[float, float, float],
    strength: float,
    rng: np.random.Generator,
    texture: float = 0.0,
) -> np.ndarray:
    """Blend a coloured stain into the floor using an alpha mask."""
    weight = np.clip(alpha * strength, 0.0, 1.0)[:, :, None]
    color = np.array(color_bgr, dtype=np.float32).reshape(1, 1, 3)
    blended = floor * (1.0 - weight) + color * weight

    if texture > 0:
        grain = rng.normal(0.0, texture, size=floor.shape[:2]).astype(np.float32)
        grain = cv2.GaussianBlur(grain, (0, 0), 1.5)
        blended = blended + (grain * np.clip(alpha, 0, 1))[:, :, None]

    return blended


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def generate_floor_image(
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    seed: int = DEFAULT_SEED,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Generate a synthetic floor image and its ground-truth stain list.

    Returns
    -------
    (image, ground_truth)
        ``image`` is a BGR uint8 array; ``ground_truth`` describes what was
        painted onto the floor (useful when demonstrating the detector).
    """
    width = int(max(320, width))
    height = int(max(320, height))
    rng = np.random.default_rng(int(seed))

    floor = _build_tiled_floor(width, height, rng, tile_size=max(140, width // 6))
    truth: List[Dict[str, Any]] = []

    def place(name: str, cx: float, cy: float, radius_ratio: float,
              color: Tuple[float, float, float], strength: float,
              texture: float = 0.0, lobes: int = 7, softness: float = 0.35) -> None:
        """Place one stain and record it as ground truth."""
        nonlocal floor
        center = (int(width * cx), int(height * cy))
        radius = int(min(width, height) * radius_ratio)
        alpha = _blob_alpha((height, width), center, radius, rng, lobes=lobes, softness=softness)
        floor = _apply_stain(floor, alpha, color, strength, rng, texture=texture)
        truth.append(
            {
                "name": name,
                "center": {"x": center[0], "y": center[1]},
                "approx_radius_px": radius,
            }
        )

    # 1. Dark liquid spill - smooth, rounded, clearly darker.
    place("Liquid spill (dark, smooth)", 0.24, 0.30, 0.085,
          color=(58.0, 60.0, 66.0), strength=0.80, texture=1.5, lobes=5, softness=0.22)

    # 2. Mud / dirt - brown, textured, irregular.
    place("Mud patch (brown, textured)", 0.55, 0.62, 0.075,
          color=(52.0, 86.0, 126.0), strength=0.78, texture=9.0, lobes=10, softness=0.40)

    # 3. Small mud splashes near the main patch.
    for index, (dx, dy, radius_ratio) in enumerate(
        [(0.06, -0.07, 0.020), (0.10, 0.04, 0.016), (-0.07, 0.06, 0.014)], start=1
    ):
        place(f"Mud splash {index}", 0.55 + dx, 0.62 + dy, radius_ratio,
              color=(48.0, 82.0, 122.0), strength=0.72, texture=7.0, lobes=5, softness=0.35)

    # 4. Coloured stain - saturated, non-brown (e.g. spilled drink).
    place("Colored stain (blue)", 0.80, 0.68, 0.055,
          color=(168.0, 88.0, 52.0), strength=0.70, texture=3.0, lobes=6, softness=0.30)

    # 5. Dust / scuff - light, low contrast, streaky.
    dust_center = (int(width * 0.36), int(height * 0.80))
    dust_alpha = np.zeros((height, width), dtype=np.float32)
    cv2.ellipse(dust_alpha, dust_center,
                (int(width * 0.11), int(height * 0.022)), 18, 0, 360, 1.0, cv2.FILLED, cv2.LINE_AA)
    dust_alpha = cv2.GaussianBlur(dust_alpha, (0, 0), 9)
    if dust_alpha.max() > 0:
        dust_alpha /= dust_alpha.max()
    # Grainy, uneven deposit rather than a flat airbrush.
    dust_mottle = rng.normal(0.0, 1.0, size=(height // 20, width // 20)).astype(np.float32)
    dust_mottle = cv2.resize(dust_mottle, (width, height), interpolation=cv2.INTER_CUBIC)
    dust_alpha = np.clip(dust_alpha * (1.0 + 0.45 * dust_mottle), 0.0, 1.0)
    floor = _apply_stain(floor, dust_alpha, (222.0, 223.0, 224.0), 1.00, rng, texture=3.0)
    truth.append(
        {
            "name": "Dust / scuff streak (light)",
            "center": {"x": dust_center[0], "y": dust_center[1]},
            "approx_radius_px": int(width * 0.11),
        }
    )

    # 6. Grime - large, low contrast, heavily textured dirty area.
    grime_center = (int(width * 0.76), int(height * 0.38))
    grime_alpha = _blob_alpha((height, width), grime_center,
                              int(min(width, height) * 0.13), rng, lobes=12, softness=0.55)
    # Coarse mottling makes the grime patch look like built-up soiling rather
    # than a flat wash, and gives the detector a real texture signal.
    mottle = rng.normal(0.0, 1.0, size=(height // 24, width // 24)).astype(np.float32)
    mottle = cv2.resize(mottle, (width, height), interpolation=cv2.INTER_CUBIC)
    grime_alpha = np.clip(grime_alpha * (1.0 + 0.30 * mottle), 0.0, 1.0)
    floor = _apply_stain(floor, grime_alpha, (126.0, 130.0, 134.0), 0.72, rng, texture=6.0)
    truth.append(
        {
            "name": "Grime / dirty patch (large, textured)",
            "center": {"x": grime_center[0], "y": grime_center[1]},
            "approx_radius_px": int(min(width, height) * 0.13),
        }
    )

    floor = _apply_lighting(floor, rng)

    image = np.clip(floor, 0, 255).astype(np.uint8)
    return image, truth


def save_sample(
    output_path: Path,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    seed: int = DEFAULT_SEED,
    quality: int = 94,
) -> Tuple[Path, List[Dict[str, Any]]]:
    """Generate a floor image and write it to ``output_path``."""
    image, truth = generate_floor_image(width=width, height=height, seed=seed)
    ensure_directory(output_path.parent)

    suffix = output_path.suffix.lower()
    if suffix in (".jpg", ".jpeg"):
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    else:
        params = []

    if not cv2.imwrite(str(output_path), image, params):
        raise IOError(f"Could not write the sample image to {output_path}")
    return output_path, truth


def _parse_args() -> argparse.Namespace:
    """Command-line interface for the generator."""
    parser = argparse.ArgumentParser(
        description="Generate a synthetic floor image with stain-like areas for CC1 Vision."
    )
    parser.add_argument("--output", type=str, default=str(SAMPLES_DIR / "sample_floor.jpg"),
                        help="Output image path (default: samples/sample_floor.jpg)")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH, help="Image width in pixels")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT, help="Image height in pixels")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    parser.add_argument("--extra", type=int, default=0,
                        help="Also generate N extra floors with incremented seeds")
    return parser.parse_args()


def main() -> int:
    """Entry point: write the sample image(s) and print a short summary."""
    args = _parse_args()
    try:
        path, truth = save_sample(Path(args.output), args.width, args.height, args.seed)
    except Exception as exc:
        print(f"[CC1 Vision] Sample generation failed: {exc}")
        return 1

    print(f"[CC1 Vision] Sample floor written to: {path.resolve()}")
    print(f"[CC1 Vision] Painted {len(truth)} stain-like areas:")
    for item in truth:
        print(f"   - {item['name']} at ({item['center']['x']}, {item['center']['y']})")

    for index in range(int(args.extra)):
        extra_path = Path(args.output).with_name(f"sample_floor_{index + 2:02d}.jpg")
        try:
            save_sample(extra_path, args.width, args.height, args.seed + index + 1)
            print(f"[CC1 Vision] Extra sample written to: {extra_path.resolve()}")
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[CC1 Vision] Could not write {extra_path}: {exc}")

    print("[CC1 Vision] Now run:  python -m streamlit run app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# --------------------------------------------------------------------------
# Public aliases used by dataset_generator.py
# --------------------------------------------------------------------------
# The dataset builder reuses exactly the same floor/stain primitives that
# produce the demo image, so the CNN is trained on the same visual world the
# detector is demonstrated on. These names are the supported entry points.

build_tiled_floor = _build_tiled_floor
apply_lighting = _apply_lighting
blob_alpha = _blob_alpha
apply_stain = _apply_stain


def generate_single_stain_floor(
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    seed: int = 11,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Generate a tiled floor carrying exactly ONE central brown stain.

    This is the regression image used by ``selftest.py`` to verify that the
    detector collapses a single physical stain into a single detection instead
    of reporting grout lines, corners and fragments alongside it.
    """
    width = int(max(320, width))
    height = int(max(320, height))
    rng = np.random.default_rng(int(seed))

    floor = _build_tiled_floor(width, height, rng, tile_size=max(140, width // 5))

    center = (int(width * 0.5), int(height * 0.5))
    radius = int(min(width, height) * 0.11)
    alpha = _blob_alpha((height, width), center, radius, rng, lobes=9, softness=0.34)
    floor = _apply_stain(floor, alpha, (46.0, 84.0, 128.0), 0.82, rng, texture=8.0)

    floor = _apply_lighting(floor, rng)
    image = np.clip(floor, 0, 255).astype(np.uint8)

    truth = [{
        "name": "Central brown stain (mud / dirt)",
        "center": {"x": center[0], "y": center[1]},
        "approx_radius_px": radius,
    }]
    return image, truth
