"""
CC1 Vision - headless self test.

Runs the entire pipeline without Streamlit, so you can confirm the install
works before starting the dashboard:

    python selftest.py

It verifies the trained CNN (file present, loads, runs, returns a valid
probability distribution), generates a sample floor, runs the OpenCV + CNN
hybrid detector, checks that a single central stain yields a single detection
with no border or grout artifacts, builds cleaning decisions, plans a route,
renders every image, writes JSON/CSV reports into reports/, and checks the
awkward edge cases (empty bytes, corrupt file, tiny image, blank floor,
degenerate crops).

Exit code 0 means everything passed.
"""

from __future__ import annotations

import sys
import traceback
from typing import Callable, List, Tuple

import cv2
import numpy as np

import utils
from cleaning_engine import build_cleaning_plans, plans_by_id, plans_to_records, summarize_mission
from detector import DetectorConfig, detect_stains, draw_detections, render_mask_view
from ml_classifier import extract_crop, get_classifier
from route_simulator import plan_route, render_floor_map, route_to_records
from sample_generator import generate_floor_image, generate_single_stain_floor, save_sample
from utils import ImageLoadError

PASS = "[ OK ]"
FAIL = "[FAIL]"


# --------------------------------------------------------------------------
# Machine-learning checks
# --------------------------------------------------------------------------


def check_model_file() -> None:
    """The trained model and its metadata must be present on disk."""
    classifier = get_classifier()
    assert classifier.model_exists(), (
        f"{classifier.model_path} not found. Train it first:\n"
        "    python dataset_generator.py\n"
        "    python train_classifier.py"
    )
    size_mb = classifier.model_path.stat().st_size / (1024 * 1024)
    assert size_mb > 0.05, "the model file looks empty"

    for path in (classifier.labels_path, classifier.config_path, classifier.metrics_path):
        assert path.is_file(), f"{path.name} is missing from models/"

    assert len(classifier.classes) >= 2, "the model must have at least two classes"
    assert "clean" in classifier.classes, "a 'clean' class is required to reject regions"
    print(f"       {classifier.model_path.name}, {size_mb:.1f} MB, "
          f"{len(classifier.classes)} classes: {', '.join(classifier.classes)}")


def check_model_loads() -> None:
    """The model must load through Keras without raising."""
    classifier = get_classifier()
    assert classifier.is_ready, f"model did not load: {classifier.status}"
    assert classifier.status == "Loaded", f"unexpected status: {classifier.status}"
    assert classifier.input_size >= 32, "implausible model input size"
    print(f"       loaded, input {classifier.input_size}x{classifier.input_size}, "
          f"{classifier.framework_version()}")


def check_model_inference() -> None:
    """Inference must return one valid probability distribution per crop."""
    classifier = get_classifier()
    size = classifier.input_size

    image, _ = generate_single_stain_floor(width=640, height=480, seed=4)
    stain_crop = extract_crop(image, (250, 180, 140, 120), size)
    assert stain_crop is not None and stain_crop.shape == (size, size, 3), "bad crop shape"

    clean_crop = extract_crop(image, (10, 10, 90, 90), size)
    blank_crop = np.full((size, size, 3), 180, np.uint8)

    predictions = classifier.predict_batch([stain_crop, clean_crop, blank_crop])
    assert len(predictions) == 3, "wrong number of predictions"

    for index, prediction in enumerate(predictions):
        assert prediction is not None, f"crop {index} returned no prediction"
        assert 0.0 <= prediction.confidence <= 1.0, "confidence outside 0..1"
        assert prediction.class_name in classifier.classes, "unknown class name"
        assert len(prediction.probabilities) == len(classifier.classes), \
            "probability vector has the wrong length"
        total = sum(prediction.probabilities.values())
        assert abs(total - 1.0) < 1e-3, f"probabilities sum to {total}, not 1"
        assert all(0.0 <= value <= 1.0 for value in prediction.probabilities.values()), \
            "a probability fell outside 0..1"
        best = max(prediction.probabilities.values())
        assert abs(best - prediction.confidence) < 1e-5, \
            "confidence is not the maximum probability"

    # A flat grey patch carries no stain evidence at all.
    flat = predictions[2]
    assert not flat.is_stain or flat.confidence < 0.9, \
        "a blank grey patch was confidently called a stain"

    print(f"       stain crop -> {predictions[0].class_name} "
          f"({predictions[0].confidence * 100:.0f}%), "
          f"blank patch -> {flat.class_name} ({flat.confidence * 100:.0f}%)")


def check_hybrid_pipeline() -> None:
    """The detector must actually route its decisions through the CNN."""
    image = utils.load_image_from_path(utils.SAMPLES_DIR / "sample_floor.jpg")
    result = detect_stains(image, DetectorConfig())

    assert result.classifier_used == "cnn", \
        f"detector did not use the CNN: {result.classifier_note}"
    assert result.candidates_considered >= result.count, \
        "fewer candidates than accepted detections"
    assert result.uses_cnn and result.confidence_label == "CNN prediction confidence"

    for det in result.detections:
        assert det.classifier == "cnn", "a detection bypassed the CNN"
        assert det.ml_class, "no CNN class recorded"
        assert abs(det.confidence - det.ml_confidence) < 1e-3, \
            "displayed confidence is not the CNN probability"
        assert det.ml_probabilities, "no probability vector stored"

    print(f"       {result.candidates_considered} candidates -> {result.count} "
          f"CNN-confirmed stains, avg confidence {result.average_confidence:.2f}")


def check_single_stain_regression() -> None:
    """One central stain must produce one detection, with no edge artifacts."""
    image, truth = generate_single_stain_floor()
    result = detect_stains(image, DetectorConfig())

    assert result.count >= 1, "the central stain was missed entirely"
    assert result.count <= 2, f"one stain fragmented into {result.count} detections"

    height, width = result.image_bgr.shape[:2]
    margin_x, margin_y = 0.06 * width, 0.06 * height
    for det in result.detections:
        assert det.center_x > margin_x and det.center_x < width - margin_x, \
            f"detection {det.id} sits on the left/right image border"
        assert det.center_y > margin_y and det.center_y < height - margin_y, \
            f"detection {det.id} sits on the top/bottom image border"
        assert det.elongation < 7.0, \
            f"detection {det.id} is a thin line, probably a grout joint"

    # The strongest detection must land on the painted stain.
    best = result.detections[0]
    expected = truth[0]["center"]
    scale = result.scale
    distance = float(np.hypot(best.center_x - expected["x"] * scale,
                              best.center_y - expected["y"] * scale))
    tolerance = truth[0]["approx_radius_px"] * scale * 1.5
    assert distance <= tolerance, (
        f"top detection is {distance:.0f}px from the real stain (tolerance {tolerance:.0f}px)"
    )

    print(f"       one central stain -> {result.count} detection(s): "
          f"{best.stain_type} at {best.confidence * 100:.0f}% "
          f"({best.severity}), {distance:.0f}px from ground truth")


def check_sample_generation() -> None:
    """The synthetic floor must be generated and written to samples/."""
    path, truth = save_sample(utils.SAMPLES_DIR / "sample_floor.jpg")
    assert path.exists(), "sample image was not written"
    assert path.stat().st_size > 5000, "sample image looks empty"
    assert len(truth) >= 5, "sample should contain several stain areas"
    print(f"       sample: {path.name}, {path.stat().st_size // 1024} KB, "
          f"{len(truth)} painted areas")


def check_detection() -> None:
    """Detection must find the planted stains and stay reasonably fast."""
    image = utils.load_image_from_path(utils.SAMPLES_DIR / "sample_floor.jpg")
    result = detect_stains(image, DetectorConfig())

    assert result.count > 0, "no stains detected on the sample floor"
    assert result.elapsed_seconds < 15, "detection is unexpectedly slow"
    assert 0.0 < result.coverage_ratio < 0.5, f"odd coverage: {result.coverage_ratio}"

    for detection in result.detections:
        assert 0.0 <= detection.confidence <= 1.0, "confidence out of range"
        assert detection.severity in ("Low", "Medium", "High"), "bad severity value"
        assert detection.area_px > 0 and detection.width > 0 and detection.height > 0
        assert detection.contour is not None and len(detection.contour) >= 3

    types = sorted({d.stain_type for d in result.detections})
    print(f"       {result.count} detections in {result.elapsed_seconds:.2f}s, "
          f"avg confidence {result.average_confidence:.2f}")
    print(f"       categories: {', '.join(types)}")


def check_sensitivity_monotonic() -> None:
    """Raising sensitivity must not reduce the number of detections."""
    image = utils.load_image_from_path(utils.SAMPLES_DIR / "sample_floor.jpg")
    counts = []
    for sensitivity in (20, 50, 80):
        result = detect_stains(image, DetectorConfig(sensitivity=sensitivity))
        counts.append(result.count)
    assert counts[0] <= counts[2], f"sensitivity behaved backwards: {counts}"
    print(f"       detections at sensitivity 20/50/80: {counts}")


def check_cleaning_engine() -> None:
    """Every detection must produce a valid, priority-sorted cleaning plan."""
    image = utils.load_image_from_path(utils.SAMPLES_DIR / "sample_floor.jpg")
    result = detect_stains(image, DetectorConfig())
    plans = build_cleaning_plans(result.detections)

    assert len(plans) == result.count, "plan count does not match detection count"
    scores = [plan.priority_score for plan in plans]
    assert scores == sorted(scores, reverse=True), "plans are not sorted by priority"
    for plan in plans:
        assert plan.action and plan.intensity and plan.tool, "incomplete cleaning plan"
        assert plan.priority in ("HIGH", "MEDIUM", "LOW"), "bad priority level"
        assert plan.estimated_seconds > 0, "zero cleaning time"

    mission = summarize_mission(plans, result.coverage_ratio)
    assert mission["tasks"] == len(plans)
    assert len(plans_to_records(plans)) == len(plans)
    print(f"       {mission['tasks']} tasks, {mission['high_priority']} high priority, "
          f"total {mission['total_clean_time_text']}, mode: {mission['recommended_mode']}")


def check_route_and_rendering() -> None:
    """Route planning and all three renderers must produce valid images."""
    image = utils.load_image_from_path(utils.SAMPLES_DIR / "sample_floor.jpg")
    result = detect_stains(image, DetectorConfig())
    plans = build_cleaning_plans(result.detections)
    route = plan_route(plans, result.image_bgr.shape[:2])

    assert len(route.stops) == len(plans), "route missed some targets"
    visited = [stop.detection_id for stop in route.stops]
    assert len(set(visited)) == len(visited), "a target was visited twice"
    priorities = [stop.priority for stop in route.stops]
    rank = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    assert [rank[p] for p in priorities] == sorted(rank[p] for p in priorities), \
        "route does not respect priority order"
    assert route.total_distance_px > 0 and route.total_seconds > 0

    annotated = draw_detections(result.image_bgr, result.detections)
    mask_view = render_mask_view(result)
    floor_map = render_floor_map(result.image_bgr, route)
    for name, rendered in (("annotated", annotated), ("mask", mask_view), ("map", floor_map)):
        assert rendered.shape == result.image_bgr.shape, f"{name} has the wrong shape"
        assert rendered.dtype == np.uint8, f"{name} has the wrong dtype"

    assert len(route_to_records(route)) == len(route.stops)
    print(f"       route: {len(route.stops)} stops, {route.total_distance_m:.2f} m, "
          f"mission {utils.format_duration(route.total_seconds)}")


def check_reports() -> None:
    """JSON, CSV and PNG reports must be produced and written to disk."""
    image = utils.load_image_from_path(utils.SAMPLES_DIR / "sample_floor.jpg")
    result = detect_stains(image, DetectorConfig())
    plans = build_cleaning_plans(result.detections)
    table = utils.build_detection_table(result.detections, plans_by_id(plans))

    assert list(table.columns) == utils.DETECTION_TABLE_COLUMNS, "table columns changed"
    assert len(table) == result.count, "table row count mismatch"

    report = utils.build_json_report(
        detections=result.detections,
        plans=plans,
        summary={**result.summary(), "mission": summarize_mission(plans, result.coverage_ratio)},
        settings=DetectorConfig().as_dict(),
        image_info=utils.describe_image(result.image_bgr, "sample_floor.jpg", result.scale),
    )
    assert report["project"] == "CC1 Vision"
    assert report["detector"]["trained_model_used"] is result.uses_cnn
    assert len(report["detections"]) == result.count

    json_bytes = utils.report_to_json_bytes(report)
    csv_bytes = utils.dataframe_to_csv_bytes(table)
    png_bytes = utils.encode_png(draw_detections(result.image_bgr, result.detections))
    assert len(json_bytes) > 100 and len(csv_bytes) > 20 and len(png_bytes) > 1000

    import json as json_module
    reloaded = json_module.loads(json_bytes.decode("utf-8"))
    assert reloaded["summary"]["detected_stains"] == result.count, "JSON round trip failed"

    written = utils.save_report_bundle(png_bytes, json_bytes, csv_bytes)
    for path in written:
        assert path.exists() and path.stat().st_size > 0, f"{path} was not written"
    print(f"       wrote {len(written)} files into reports/ "
          f"({', '.join(p.suffix for p in written)})")


def check_edge_cases() -> None:
    """Bad and unusual inputs must fail cleanly instead of crashing."""
    for label, payload in (
        ("empty bytes", b""),
        ("corrupt data", b"not an image at all, just text"),
        ("truncated jpeg", b"\xff\xd8\xff\xe0" + b"\x00" * 40),
    ):
        try:
            utils.load_image_from_bytes(payload)
        except ImageLoadError:
            pass
        else:
            raise AssertionError(f"{label} should have raised ImageLoadError")

    tiny = cv2.imencode(".png", np.full((16, 16, 3), 200, np.uint8))[1].tobytes()
    try:
        utils.load_image_from_bytes(tiny)
    except ImageLoadError:
        pass
    else:
        raise AssertionError("undersized image should have been rejected")

    # A blank floor must return zero detections rather than raising.
    blank = np.full((600, 800, 3), 185, np.uint8)
    blank_result = detect_stains(blank, DetectorConfig())
    assert blank_result.count == 0, f"blank floor produced {blank_result.count} detections"
    assert build_cleaning_plans(blank_result.detections) == []
    empty_route = plan_route([], blank.shape[:2])
    assert empty_route.stops == [] and empty_route.total_seconds == 0
    assert utils.build_detection_table([], {}).empty
    render_floor_map(blank, empty_route)  # must not raise with zero targets

    # Grayscale and RGBA inputs must be normalised to 3-channel BGR.
    gray_bytes = cv2.imencode(".png", np.full((200, 200), 180, np.uint8))[1].tobytes()
    assert utils.load_image_from_bytes(gray_bytes).shape[2] == 3, "grayscale not converted"
    rgba = np.dstack([np.full((200, 200, 3), 180, np.uint8),
                      np.full((200, 200), 255, np.uint8)])
    # The classifier must survive degenerate crops rather than raising.
    classifier = get_classifier()
    degenerate = classifier.predict_batch([None, np.zeros((0, 0, 3), np.uint8),
                                           np.zeros((4, 4, 3), np.uint8)])
    assert len(degenerate) == 3, "predict_batch changed the result length"
    assert degenerate[0] is None and degenerate[1] is None, \
        "an unusable crop should return None, not a prediction"
    assert degenerate[2] is not None, "a tiny but valid crop should still be classified"
    assert classifier.predict_batch([]) == [], "empty batch should return an empty list"

    # An out-of-frame region must still yield a correctly shaped crop.
    edge_crop = extract_crop(np.full((120, 120, 3), 180, np.uint8), (-40, -40, 60, 60),
                             classifier.input_size)
    assert edge_crop is not None, "an off-image region produced no crop"
    assert edge_crop.shape == (classifier.input_size, classifier.input_size, 3), \
        "off-image crop has the wrong shape"

    # The detector must also run with the CNN explicitly switched off.
    fallback = detect_stains(blank, DetectorConfig(use_cnn=False))
    assert fallback.classifier_used == "heuristic", "CNN opt-out was ignored"

    rgba_bytes = cv2.imencode(".png", rgba)[1].tobytes()
    assert utils.load_image_from_bytes(rgba_bytes).shape[2] == 3, "alpha channel not dropped"

    # A very large image must be downscaled, not rejected.
    large, _ = generate_floor_image(width=3000, height=2000, seed=3)
    large_result = detect_stains(large, DetectorConfig())
    assert max(large_result.image_bgr.shape[:2]) <= 1280, "large image was not downscaled"
    assert large_result.scale < 1.0, "resize scale was not recorded"
    print(f"       3000x2000 input analysed at "
          f"{large_result.image_bgr.shape[1]}x{large_result.image_bgr.shape[0]} "
          f"(scale {large_result.scale:.3f})")


def check_many_regions() -> None:
    """A very noisy floor must stay bounded by max_detections."""
    rng = np.random.default_rng(5)
    noisy = np.full((700, 900, 3), 175, np.uint8)
    for _ in range(400):
        centre = (int(rng.integers(20, 880)), int(rng.integers(20, 680)))
        cv2.circle(noisy, centre, int(rng.integers(6, 16)),
                   (int(rng.integers(60, 140)),) * 3, cv2.FILLED)
    result = detect_stains(noisy, DetectorConfig(sensitivity=85, min_area_px=100))
    assert result.count <= DetectorConfig().max_detections, "detection cap was exceeded"
    plans = build_cleaning_plans(result.detections)
    assert len(plans) == result.count
    print(f"       400 synthetic spots -> {result.count} detections "
          f"(cap {DetectorConfig().max_detections})")


CHECKS: List[Tuple[str, Callable[[], None]]] = [
    ("Trained model present", check_model_file),
    ("Model loads", check_model_loads),
    ("CNN inference and probabilities", check_model_inference),
    ("Sample generation", check_sample_generation),
    ("OpenCV + CNN hybrid pipeline", check_hybrid_pipeline),
    ("Single central stain regression", check_single_stain_regression),
    ("Detection pipeline", check_detection),
    ("Sensitivity control", check_sensitivity_monotonic),
    ("Cleaning decision engine", check_cleaning_engine),
    ("Route simulation and rendering", check_route_and_rendering),
    ("Report generation", check_reports),
    ("Edge cases and bad input", check_edge_cases),
    ("Many detected regions", check_many_regions),
]


def main() -> int:
    """Run every check and report a summary."""
    print("=" * 66)
    print(" CC1 Vision - self test")
    print("=" * 66)

    # Load the CNN up front so its one-off load time is not charged to the
    # first detection check, which asserts on analysis speed.
    get_classifier().load()

    failures = 0
    for name, check in CHECKS:
        try:
            check()
            print(f"{PASS} {name}")
        except Exception as exc:
            failures += 1
            print(f"{FAIL} {name}: {exc}")
            traceback.print_exc()

    print("=" * 66)
    if failures:
        print(f" {failures} of {len(CHECKS)} checks FAILED")
        return 1
    print(f" All {len(CHECKS)} checks passed. Start the dashboard with:")
    print("     python -m streamlit run app.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
