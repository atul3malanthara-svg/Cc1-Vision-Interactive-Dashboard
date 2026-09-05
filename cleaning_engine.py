"""
CC1 Vision - cleaning decision engine.

Converts detected stain regions into *simulated* cleaning actions for an
autonomous floor-cleaning robot: which tool to use, how much water, how many
passes, how long it should take and how urgent the target is.

The rules below are an engineering rule table, not a learned policy. They are
kept in one place so a real robot's capability profile could replace them
without touching the detector or the dashboard.

This module has no OpenCV / Streamlit dependency and works on any object that
exposes the attributes of :class:`detector.Detection` (duck typing), which
keeps the import graph acyclic:

    app.py -> detector.py -> utils.py
    app.py -> cleaning_engine.py -> utils.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

from utils import clamp, format_duration

# --------------------------------------------------------------------------
# Rule table
# --------------------------------------------------------------------------

#: Per-category cleaning profile.
#:
#: ``urgency`` biases the priority score (a liquid spill is a slip hazard, so
#: it is escalated even when it is small).
CLEANING_ACTIONS: Dict[str, Dict[str, Any]] = {
    "Liquid Spill": {
        "action": "Scrub + Suction",
        "intensity": "High",
        "tool": "Roller brush + vacuum squeegee",
        "water_usage": "Medium",
        "passes": 2,
        "seconds_per_pass": 9.0,
        "urgency": 1.18,
        "hazard": "Slip hazard - clean before routine areas.",
    },
    "Mud / Dirt": {
        "action": "Scrub + Water",
        "intensity": "High",
        "tool": "Roller brush + water jet",
        "water_usage": "High",
        "passes": 3,
        "seconds_per_pass": 10.0,
        "urgency": 1.10,
        "hazard": "Tracked dirt spreads quickly across the floor.",
    },
    "Dust / Scuff": {
        "action": "Vacuum",
        "intensity": "Low",
        "tool": "Suction only",
        "water_usage": "None",
        "passes": 1,
        "seconds_per_pass": 5.0,
        "urgency": 0.85,
        "hazard": "Cosmetic only.",
    },
    "Dark Stain": {
        "action": "Scrub + Detergent",
        "intensity": "High",
        "tool": "Roller brush + detergent dosing",
        "water_usage": "Medium",
        "passes": 3,
        "seconds_per_pass": 11.0,
        "urgency": 1.05,
        "hazard": "May be set-in; re-inspect after the pass.",
    },
    "Colored Stain": {
        "action": "Scrub + Detergent",
        "intensity": "Medium/High",
        "tool": "Roller brush + detergent dosing",
        "water_usage": "Medium",
        "passes": 2,
        "seconds_per_pass": 10.0,
        "urgency": 1.02,
        "hazard": "Pigmented residue may need a second visit.",
    },
    "Grime / Dirty Patch": {
        "action": "Deep Scrub",
        "intensity": "Medium/High",
        "tool": "Dual brush deep-clean mode",
        "water_usage": "High",
        "passes": 3,
        "seconds_per_pass": 12.0,
        "urgency": 1.00,
        "hazard": "Built-up soiling over a wide area.",
    },
    # --- labels produced by the trained CNN ------------------------------
    "Grease / Oil": {
        "action": "Degrease + Scrub",
        "intensity": "High",
        "tool": "Roller brush + degreaser dosing",
        "water_usage": "Medium",
        "passes": 3,
        "seconds_per_pass": 12.0,
        "urgency": 1.20,
        "hazard": "Slippery film - degrease before any water-only pass.",
    },
    "Scattered Dirt": {
        "action": "Vacuum + Light Pass",
        "intensity": "Low/Medium",
        "tool": "Suction with side brush",
        "water_usage": "Low",
        "passes": 2,
        "seconds_per_pass": 6.0,
        "urgency": 0.90,
        "hazard": "Loose debris - vacuum first so it is not spread by water.",
    },
    "Generic Stain": {
        "action": "Standard Clean Pass",
        "intensity": "Medium",
        "tool": "Roller brush",
        "water_usage": "Low",
        "passes": 2,
        "seconds_per_pass": 8.0,
        "urgency": 0.95,
        "hazard": "Unclassified deposit - inspect after cleaning.",
    },
}

FALLBACK_PROFILE: Dict[str, Any] = CLEANING_ACTIONS["Generic Stain"]

SEVERITY_WEIGHT: Dict[str, float] = {"High": 1.0, "Medium": 0.6, "Low": 0.3}

PRIORITY_LEVELS: tuple = ("HIGH", "MEDIUM", "LOW")

PRIORITY_COLORS: Dict[str, str] = {
    "HIGH": "#C0392B",
    "MEDIUM": "#C97B12",
    "LOW": "#2C7A5E",
}


# --------------------------------------------------------------------------
# Data container
# --------------------------------------------------------------------------


@dataclass
class CleaningPlan:
    """One simulated robot task derived from one detection."""

    detection_id: int
    stain_type: str
    severity: str
    confidence: float
    area_px: int
    area_ratio: float
    center_x: int
    center_y: int
    action: str
    intensity: str
    tool: str
    water_usage: str
    passes: int
    estimated_seconds: float
    priority: str
    priority_score: float
    hazard_note: str

    @property
    def estimated_duration_text(self) -> str:
        """Human readable cleaning time, e.g. ``1 min 05 s``."""
        return format_duration(self.estimated_seconds)

    def to_dict(self) -> Dict[str, Any]:
        """Plain-dict view for tables and JSON reports."""
        return {
            "detection_id": int(self.detection_id),
            "stain_type": self.stain_type,
            "severity": self.severity,
            "confidence": round(float(self.confidence), 3),
            "area_px": int(self.area_px),
            "action": self.action,
            "intensity": self.intensity,
            "tool": self.tool,
            "water_usage": self.water_usage,
            "passes": int(self.passes),
            "estimated_seconds": round(float(self.estimated_seconds), 1),
            "priority": self.priority,
            "priority_score": round(float(self.priority_score), 3),
        }


# --------------------------------------------------------------------------
# Rule lookups
# --------------------------------------------------------------------------


def get_action_profile(stain_type: str) -> Dict[str, Any]:
    """Return the cleaning profile for a stain category (never raises)."""
    return CLEANING_ACTIONS.get(stain_type, FALLBACK_PROFILE)


def recommend_action(stain_type: str) -> str:
    """Short recommended action string, e.g. ``Scrub + Water``."""
    return str(get_action_profile(stain_type)["action"])


def compute_priority_score(
    severity: str,
    confidence: float,
    area_ratio: float,
    urgency: float = 1.0,
) -> float:
    """Blend severity, confidence and affected area into a 0..1 urgency score."""
    severity_component = SEVERITY_WEIGHT.get(severity, 0.5)
    area_component = clamp(area_ratio / 0.05, 0.0, 1.0)
    base = 0.45 * severity_component + 0.35 * clamp(confidence, 0.0, 1.0) + 0.20 * area_component
    return clamp(base * urgency, 0.0, 1.0)


def priority_label(score: float) -> str:
    """Convert a priority score into ``HIGH`` / ``MEDIUM`` / ``LOW``."""
    if score >= 0.65:
        return "HIGH"
    if score >= 0.42:
        return "MEDIUM"
    return "LOW"


# --------------------------------------------------------------------------
# Plan building
# --------------------------------------------------------------------------


def build_cleaning_plan(detection: Any) -> CleaningPlan:
    """Build a :class:`CleaningPlan` from a single detection object."""
    profile = get_action_profile(getattr(detection, "stain_type", "Generic Stain"))

    severity = getattr(detection, "severity", "Medium")
    confidence = float(getattr(detection, "confidence", 0.5))
    area_ratio = float(getattr(detection, "area_ratio", 0.0))
    passes = int(profile["passes"])

    # Bigger and more severe areas simply take longer to clean.
    area_factor = 1.0 + clamp(area_ratio / 0.02, 0.0, 4.0)
    severity_factor = 1.0 + 0.25 * SEVERITY_WEIGHT.get(severity, 0.5)
    estimated_seconds = passes * float(profile["seconds_per_pass"]) * area_factor * severity_factor

    score = compute_priority_score(
        severity=severity,
        confidence=confidence,
        area_ratio=area_ratio,
        urgency=float(profile["urgency"]),
    )

    return CleaningPlan(
        detection_id=int(getattr(detection, "id", 0)),
        stain_type=str(getattr(detection, "stain_type", "Generic Stain")),
        severity=severity,
        confidence=confidence,
        area_px=int(getattr(detection, "area_px", 0)),
        area_ratio=area_ratio,
        center_x=int(getattr(detection, "center_x", 0)),
        center_y=int(getattr(detection, "center_y", 0)),
        action=str(profile["action"]),
        intensity=str(profile["intensity"]),
        tool=str(profile["tool"]),
        water_usage=str(profile["water_usage"]),
        passes=passes,
        estimated_seconds=round(estimated_seconds, 1),
        priority=priority_label(score),
        priority_score=score,
        hazard_note=str(profile["hazard"]),
    )


def build_cleaning_plans(detections: Sequence[Any]) -> List[CleaningPlan]:
    """Build plans for every detection, sorted by descending priority."""
    plans = [build_cleaning_plan(detection) for detection in detections]
    plans.sort(key=lambda plan: plan.priority_score, reverse=True)
    return plans


def plans_by_id(plans: Sequence[CleaningPlan]) -> Dict[int, CleaningPlan]:
    """Index plans by their detection id (used to join into the table)."""
    return {plan.detection_id: plan for plan in plans}


# --------------------------------------------------------------------------
# Mission level summary
# --------------------------------------------------------------------------


def recommend_cleaning_mode(plans: Sequence[CleaningPlan], coverage_ratio: float) -> str:
    """Pick a whole-room cleaning mode from the task mix."""
    if not plans:
        return "Standby - no cleaning required"

    high_count = sum(1 for plan in plans if plan.priority == "HIGH")
    wet_tasks = sum(1 for plan in plans if plan.water_usage in ("Medium", "High"))

    if coverage_ratio >= 0.15 or high_count >= 4:
        return "Deep clean - full wet scrub of the mapped zone"
    if high_count >= 1 or wet_tasks >= 2:
        return "Spot clean - targeted wet scrubbing of flagged areas"
    if wet_tasks == 0:
        return "Dry pass - vacuum flagged areas only"
    return "Standard pass - light scrub over flagged areas"


def summarize_mission(plans: Sequence[CleaningPlan], coverage_ratio: float = 0.0) -> Dict[str, Any]:
    """Aggregate all plans into the mission summary shown in the dashboard."""
    counts = {level: 0 for level in PRIORITY_LEVELS}
    for plan in plans:
        counts[plan.priority] = counts.get(plan.priority, 0) + 1

    total_seconds = float(sum(plan.estimated_seconds for plan in plans))
    water_levels = [plan.water_usage for plan in plans]
    if "High" in water_levels:
        water_demand = "High"
    elif "Medium" in water_levels:
        water_demand = "Medium"
    elif "Low" in water_levels:
        water_demand = "Low"
    else:
        water_demand = "None"

    return {
        "tasks": len(plans),
        "high_priority": counts.get("HIGH", 0),
        "medium_priority": counts.get("MEDIUM", 0),
        "low_priority": counts.get("LOW", 0),
        "total_clean_seconds": round(total_seconds, 1),
        "total_clean_time_text": format_duration(total_seconds),
        "water_demand": water_demand,
        "recommended_mode": recommend_cleaning_mode(plans, coverage_ratio),
    }


def plans_to_records(plans: Sequence[CleaningPlan]) -> List[Dict[str, Any]]:
    """Row dictionaries for the cleaning-decision table."""
    records: List[Dict[str, Any]] = []
    for order, plan in enumerate(plans, start=1):
        records.append(
            {
                "Order": order,
                "Stain": f"Stain {plan.detection_id:02d}",
                "Type": plan.stain_type,
                "Severity": plan.severity,
                "Robot Action": plan.action,
                "Intensity": plan.intensity,
                "Water": plan.water_usage,
                "Passes": plan.passes,
                "Est. Time": plan.estimated_duration_text,
                "Priority": plan.priority,
            }
        )
    return records
