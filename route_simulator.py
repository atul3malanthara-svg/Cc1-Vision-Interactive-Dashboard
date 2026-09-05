"""
CC1 Vision - virtual floor map and route simulation.

Treats the uploaded image as a top-down cleaning zone, places the detected
stains on it, and plans the order in which a simulated robot would visit them.

Routing rule (deliberately simple and predictable):

    1. group targets by priority tier (HIGH -> MEDIUM -> LOW)
    2. inside each tier, walk a nearest-neighbour path from the robot's
       current position
    3. accumulate travel distance and add each target's cleaning time

Nothing here drives a real robot, and nothing here is required by the
detection pipeline - the module is fully self-contained so that a failure in
the map cannot break the core application.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from cleaning_engine import PRIORITY_LEVELS
from utils import clamp, format_duration

#: Default assumption: the pictured floor area is 4 metres wide.
DEFAULT_ZONE_WIDTH_METERS: float = 4.0

#: Simulated travel speed of the cleaning robot in metres per second.
DEFAULT_ROBOT_SPEED_MPS: float = 0.6

#: BGR marker colours per priority tier.
PRIORITY_MARKER_COLORS: Dict[str, Tuple[int, int, int]] = {
    "HIGH": (48, 48, 214),
    "MEDIUM": (30, 150, 235),
    "LOW": (110, 160, 60),
}

ROBOT_COLOR: Tuple[int, int, int] = (190, 130, 40)
ROUTE_COLOR: Tuple[int, int, int] = (235, 235, 235)


@dataclass
class RouteStop:
    """One cleaning target on the simulated route."""

    order: int
    detection_id: int
    x: int
    y: int
    stain_type: str
    severity: str
    priority: str
    action: str
    leg_distance_px: float
    cumulative_distance_px: float
    clean_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        """Plain-dict view for reports."""
        return {
            "order": int(self.order),
            "detection_id": int(self.detection_id),
            "x": int(self.x),
            "y": int(self.y),
            "stain_type": self.stain_type,
            "priority": self.priority,
            "action": self.action,
            "leg_distance_px": round(float(self.leg_distance_px), 1),
            "clean_seconds": round(float(self.clean_seconds), 1),
        }


@dataclass
class RoutePlan:
    """A complete simulated cleaning mission."""

    stops: List[RouteStop] = field(default_factory=list)
    start_point: Tuple[int, int] = (0, 0)
    total_distance_px: float = 0.0
    pixels_per_meter: float = 100.0
    zone_width_meters: float = DEFAULT_ZONE_WIDTH_METERS
    robot_speed_mps: float = DEFAULT_ROBOT_SPEED_MPS
    cleaning_seconds: float = 0.0

    @property
    def total_distance_m(self) -> float:
        """Route length converted to metres using the assumed zone scale."""
        if self.pixels_per_meter <= 0:
            return 0.0
        return self.total_distance_px / self.pixels_per_meter

    @property
    def travel_seconds(self) -> float:
        """Time spent driving between targets."""
        if self.robot_speed_mps <= 0:
            return 0.0
        return self.total_distance_m / self.robot_speed_mps

    @property
    def total_seconds(self) -> float:
        """Travel time plus cleaning time."""
        return self.travel_seconds + self.cleaning_seconds

    def summary(self) -> Dict[str, Any]:
        """Mission figures for the dashboard and the JSON report."""
        return {
            "targets": len(self.stops),
            "start_point": {"x": int(self.start_point[0]), "y": int(self.start_point[1])},
            "route_length_px": round(self.total_distance_px, 1),
            "route_length_m": round(self.total_distance_m, 2),
            "assumed_zone_width_m": round(float(self.zone_width_meters), 2),
            "travel_time": format_duration(self.travel_seconds),
            "cleaning_time": format_duration(self.cleaning_seconds),
            "total_time": format_duration(self.total_seconds),
            "order": [stop.detection_id for stop in self.stops],
        }


def _distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Euclidean distance between two points."""
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def plan_route(
    plans: Sequence[Any],
    image_shape: Tuple[int, int],
    start_point: Optional[Tuple[int, int]] = None,
    zone_width_meters: float = DEFAULT_ZONE_WIDTH_METERS,
    robot_speed_mps: float = DEFAULT_ROBOT_SPEED_MPS,
) -> RoutePlan:
    """Order cleaning targets into a simulated robot route.

    Parameters
    ----------
    plans:
        Objects exposing ``detection_id``, ``center_x``, ``center_y``,
        ``priority``, ``stain_type``, ``severity``, ``action`` and
        ``estimated_seconds`` (i.e. :class:`cleaning_engine.CleaningPlan`).
    image_shape:
        ``(height, width)`` of the map image.
    start_point:
        Robot dock position in pixels. Defaults to the bottom-left corner.
    """
    height, width = int(image_shape[0]), int(image_shape[1])
    if start_point is None:
        start_point = (int(0.04 * width), int(0.94 * height))

    pixels_per_meter = width / max(0.1, float(zone_width_meters))

    route = RoutePlan(
        start_point=(int(start_point[0]), int(start_point[1])),
        pixels_per_meter=pixels_per_meter,
        zone_width_meters=float(zone_width_meters),
        robot_speed_mps=robot_speed_mps,
    )
    if not plans:
        return route

    # Group by priority tier, keeping any unknown tier at the end.
    tiers: Dict[str, List[Any]] = {level: [] for level in PRIORITY_LEVELS}
    for plan in plans:
        tiers.setdefault(getattr(plan, "priority", "LOW"), []).append(plan)

    current = (float(start_point[0]), float(start_point[1]))
    cumulative = 0.0
    order = 0

    for level in list(PRIORITY_LEVELS) + [key for key in tiers if key not in PRIORITY_LEVELS]:
        remaining = list(tiers.get(level, []))
        while remaining:
            # Nearest-neighbour step inside the current priority tier.
            nearest_index = min(
                range(len(remaining)),
                key=lambda i: _distance(
                    current,
                    (float(remaining[i].center_x), float(remaining[i].center_y)),
                ),
            )
            plan = remaining.pop(nearest_index)
            target = (float(plan.center_x), float(plan.center_y))
            leg = _distance(current, target)
            cumulative += leg
            order += 1

            route.stops.append(
                RouteStop(
                    order=order,
                    detection_id=int(getattr(plan, "detection_id", order)),
                    x=int(target[0]),
                    y=int(target[1]),
                    stain_type=str(getattr(plan, "stain_type", "Generic Stain")),
                    severity=str(getattr(plan, "severity", "Medium")),
                    priority=str(getattr(plan, "priority", "LOW")),
                    action=str(getattr(plan, "action", "Standard Clean Pass")),
                    leg_distance_px=leg,
                    cumulative_distance_px=cumulative,
                    clean_seconds=float(getattr(plan, "estimated_seconds", 0.0)),
                )
            )
            current = target

    route.total_distance_px = cumulative
    route.cleaning_seconds = float(sum(stop.clean_seconds for stop in route.stops))
    return route


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def _draw_dashed_line(
    canvas: np.ndarray,
    start: Tuple[int, int],
    end: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int = 2,
    dash_length: int = 12,
) -> None:
    """Draw a dashed straight line between two points."""
    distance = _distance(start, end)
    if distance < 1:
        return
    steps = max(1, int(distance // dash_length))
    for step in range(steps):
        if step % 2 == 1:
            continue
        t0 = step / steps
        t1 = min(1.0, (step + 1) / steps)
        p0 = (int(start[0] + (end[0] - start[0]) * t0),
              int(start[1] + (end[1] - start[1]) * t0))
        p1 = (int(start[0] + (end[0] - start[0]) * t1),
              int(start[1] + (end[1] - start[1]) * t1))
        cv2.line(canvas, p0, p1, color, thickness, cv2.LINE_AA)


def _draw_grid(canvas: np.ndarray, spacing: int, color: Tuple[int, int, int]) -> None:
    """Draw a light reference grid over the floor map."""
    height, width = canvas.shape[:2]
    for x in range(spacing, width, spacing):
        cv2.line(canvas, (x, 0), (x, height), color, 1, cv2.LINE_AA)
    for y in range(spacing, height, spacing):
        cv2.line(canvas, (0, y), (width, y), color, 1, cv2.LINE_AA)


def _draw_robot(canvas: np.ndarray, position: Tuple[int, int], radius: int) -> None:
    """Draw the simulated robot at its dock position."""
    cv2.circle(canvas, position, radius, ROBOT_COLOR, cv2.FILLED, cv2.LINE_AA)
    cv2.circle(canvas, position, radius, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.circle(canvas, position, max(2, radius // 3), (255, 255, 255), cv2.FILLED, cv2.LINE_AA)


def render_floor_map(
    image_bgr: np.ndarray,
    route: RoutePlan,
    show_route: bool = True,
    show_labels: bool = True,
    dim_factor: float = 0.55,
) -> np.ndarray:
    """Render the 2D virtual floor map with targets and the planned route."""
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("render_floor_map needs a valid image.")

    height, width = image_bgr.shape[:2]

    # Desaturated, dimmed floor so overlays stay readable.
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    canvas = (canvas.astype(np.float32) * clamp(dim_factor, 0.1, 1.0)).astype(np.uint8)
    canvas = cv2.addWeighted(canvas, 0.85, np.full_like(canvas, (28, 24, 20)), 0.15, 0)

    _draw_grid(canvas, spacing=max(40, int(min(height, width) / 10)), color=(70, 70, 70))
    cv2.rectangle(canvas, (0, 0), (width - 1, height - 1), (120, 120, 120), 2)

    scale = clamp(min(width, height) / 900.0, 0.45, 1.2)
    marker_radius = int(clamp(min(width, height) * 0.022, 10, 34))
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Route legs first so markers sit on top of the lines.
    if show_route and route.stops:
        previous = route.start_point
        for stop in route.stops:
            _draw_dashed_line(canvas, previous, (stop.x, stop.y), ROUTE_COLOR,
                              thickness=max(1, int(2 * scale)))
            mid = ((previous[0] + stop.x) // 2, (previous[1] + stop.y) // 2)
            cv2.circle(canvas, mid, max(2, int(3 * scale)), ROUTE_COLOR, cv2.FILLED, cv2.LINE_AA)
            previous = (stop.x, stop.y)

    _draw_robot(canvas, route.start_point, marker_radius)
    if show_labels:
        cv2.putText(canvas, "DOCK",
                    (route.start_point[0] - marker_radius,
                     route.start_point[1] + marker_radius + int(18 * scale)),
                    font, 0.5 * scale, (255, 255, 255), max(1, int(scale)), cv2.LINE_AA)

    for stop in route.stops:
        color = PRIORITY_MARKER_COLORS.get(stop.priority, (200, 200, 200))
        cv2.circle(canvas, (stop.x, stop.y), marker_radius, color, cv2.FILLED, cv2.LINE_AA)
        cv2.circle(canvas, (stop.x, stop.y), marker_radius, (255, 255, 255), 2, cv2.LINE_AA)

        label = f"{stop.order}"
        (text_w, text_h), _ = cv2.getTextSize(label, font, 0.62 * scale, 2)
        cv2.putText(canvas, label,
                    (stop.x - text_w // 2, stop.y + text_h // 2),
                    font, 0.62 * scale, (255, 255, 255), 2, cv2.LINE_AA)

        if show_labels:
            caption = f"#{stop.detection_id:02d} {stop.stain_type}"
            cv2.putText(canvas, caption,
                        (stop.x - marker_radius, stop.y - marker_radius - int(8 * scale)),
                        font, 0.46 * scale, (255, 255, 255), max(1, int(scale)), cv2.LINE_AA)

    _draw_legend(canvas, route, scale)
    return canvas


def _draw_legend(canvas: np.ndarray, route: RoutePlan, scale: float) -> None:
    """Draw the legend / mission readout in the top-left corner."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    lines = [
        "CC1 VIRTUAL FLOOR MAP",
        f"Targets: {len(route.stops)}",
        f"Route: {route.total_distance_m:.2f} m",
        f"Mission: {format_duration(route.total_seconds)}",
    ]
    padding = int(12 * scale)
    line_height = int(22 * scale)
    box_w = int(max(cv2.getTextSize(line, font, 0.5 * scale, 1)[0][0] for line in lines)) + 2 * padding
    box_h = line_height * len(lines) + padding

    overlay = canvas.copy()
    cv2.rectangle(overlay, (padding, padding), (padding + box_w, padding + box_h),
                  (18, 18, 18), cv2.FILLED)
    cv2.addWeighted(overlay, 0.72, canvas, 0.28, 0, dst=canvas)
    cv2.rectangle(canvas, (padding, padding), (padding + box_w, padding + box_h),
                  (150, 150, 150), 1)

    y = padding + line_height - int(6 * scale)
    for index, line in enumerate(lines):
        weight = 2 if index == 0 else 1
        cv2.putText(canvas, line, (padding * 2, y), font, 0.5 * scale,
                    (255, 255, 255), weight, cv2.LINE_AA)
        y += line_height


def route_to_records(route: RoutePlan) -> List[Dict[str, Any]]:
    """Row dictionaries describing the route, for display in a table."""
    records: List[Dict[str, Any]] = []
    for stop in route.stops:
        records.append(
            {
                "Stop": stop.order,
                "Stain": f"Stain {stop.detection_id:02d}",
                "Type": stop.stain_type,
                "Priority": stop.priority,
                "Action": stop.action,
                "X": stop.x,
                "Y": stop.y,
                "Leg (px)": round(stop.leg_distance_px, 1),
                "Clean Time": format_duration(stop.clean_seconds),
            }
        )
    return records
