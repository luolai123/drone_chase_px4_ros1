#!/usr/bin/env python3
"""Pure-Python observation_v2 builder.

This module is intentionally not wired into GazeboChaseEnv. It builds a
candidate extended observation for offline analysis and future Phase 9 policy
training only.
"""

import json
import math

import numpy as np


MAX_DEPTH = 10.0
TARGET_LOST_WINDOW = 20.0
RADIUS_NORMALIZER_PX = 80.0


FIELD_SPECS = [
    {"name": "target_visible", "default": 0.0, "min": 0.0, "max": 1.0},
    {"name": "target_x_camera", "default": 0.0, "min": -10.0, "max": 10.0},
    {"name": "target_y_camera", "default": 0.0, "min": -10.0, "max": 10.0},
    {"name": "target_z_camera", "default": MAX_DEPTH, "min": 0.0, "max": MAX_DEPTH},
    {"name": "target_distance", "default": MAX_DEPTH, "min": 0.0, "max": MAX_DEPTH},
    {"name": "target_u", "default": 0.0, "min": -1.5, "max": 1.5},
    {"name": "target_v", "default": 0.0, "min": -1.5, "max": 1.5},
    {"name": "target_confidence", "default": 0.0, "min": 0.0, "max": 1.0},
    {"name": "target_visible_ratio_window", "default": 0.0, "min": 0.0, "max": 1.0},
    {"name": "target_lost_frames_normalized", "default": 1.0, "min": 0.0, "max": 1.0},
    {"name": "depth_valid_ratio", "default": 0.0, "min": 0.0, "max": 1.0},
    {"name": "radius_px_smooth_normalized", "default": 0.0, "min": 0.0, "max": 1.0},
    {"name": "detection_quality_score", "default": 0.0, "min": 0.0, "max": 1.0},
    {"name": "sector_far_left_q05", "default": MAX_DEPTH, "min": 0.0, "max": MAX_DEPTH},
    {"name": "sector_left_q05", "default": MAX_DEPTH, "min": 0.0, "max": MAX_DEPTH},
    {"name": "sector_front_q05", "default": MAX_DEPTH, "min": 0.0, "max": MAX_DEPTH},
    {"name": "sector_right_q05", "default": MAX_DEPTH, "min": 0.0, "max": MAX_DEPTH},
    {"name": "sector_far_right_q05", "default": MAX_DEPTH, "min": 0.0, "max": MAX_DEPTH},
    {"name": "front_q05_smoothed", "default": MAX_DEPTH, "min": 0.0, "max": MAX_DEPTH},
    {"name": "obstacle_area_ratio", "default": 0.0, "min": 0.0, "max": 1.0},
    {"name": "obstacle_danger", "default": 0.0, "min": 0.0, "max": 1.0},
    {"name": "drone_z", "default": 0.0, "min": 0.0, "max": 5.0},
    {"name": "drone_vx", "default": 0.0, "min": -3.0, "max": 3.0},
    {"name": "drone_vy", "default": 0.0, "min": -3.0, "max": 3.0},
    {"name": "drone_vz", "default": 0.0, "min": -2.0, "max": 2.0},
    {"name": "prev_vx", "default": 0.0, "min": -0.5, "max": 0.5},
    {"name": "prev_vz", "default": 0.0, "min": -0.25, "max": 0.25},
    {"name": "prev_yaw_rate", "default": 0.0, "min": -0.6, "max": 0.6},
]

FIELD_NAMES = [spec["name"] for spec in FIELD_SPECS]
FIELD_INDEX = {name: index for index, name in enumerate(FIELD_NAMES)}
OBSERVATION_V2_DIM = len(FIELD_SPECS)

QUALITY_SCORE = {
    "lost": 0.0,
    "unstable": 0.35,
    "weak_depth": 0.45,
    "small_target": 0.5,
    "good": 1.0,
}


def _finite(value, default=math.nan):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(default)
    return value if math.isfinite(value) else float(default)


def _get(source, name, default=None):
    if source is None:
        return default
    if isinstance(source, dict):
        return source.get(name, default)
    return getattr(source, name, default)


def _nested_get(source, names, default=None):
    current = source
    for name in names:
        current = _get(current, name, None)
        if current is None:
            return default
    return current


def _parse_json(value):
    if value is None or isinstance(value, dict):
        return value
    if hasattr(value, "data"):
        value = value.data
    if not isinstance(value, str):
        return None
    try:
        return json.loads(value)
    except Exception:
        return None


def _clip(name, value):
    spec = FIELD_SPECS[FIELD_INDEX[name]]
    value = _finite(value, spec["default"])
    value = float(np.clip(value, spec["min"], spec["max"]))
    return value


def _quality_score(quality):
    return QUALITY_SCORE.get(str(quality or "lost"), 0.0)


def _sector_q05(depth_debug, name, default=MAX_DEPTH):
    sectors = _get(depth_debug, "sectors", {}) or {}
    sector = _get(sectors, name, {}) or {}
    return _finite(_get(sector, "q05", default), default)


def _target_camera(target_state):
    position = _get(target_state, "position_camera", None)
    x = _finite(_get(position, "x", _get(target_state, "target_x_camera", 0.0)), 0.0)
    y = _finite(_get(position, "y", _get(target_state, "target_y_camera", 0.0)), 0.0)
    z = _finite(_get(position, "z", _get(target_state, "target_z_camera", MAX_DEPTH)), MAX_DEPTH)
    return x, y, z


def build_observation_v2(
    target_state=None,
    obstacle_risk=None,
    depth_debug=None,
    detection_quality=None,
    uav_state=None,
    prev_action=None,
    return_metadata=False,
):
    depth_debug = _parse_json(depth_debug)
    detection_quality = _parse_json(detection_quality)
    missing = {}

    obs = {spec["name"]: spec["default"] for spec in FIELD_SPECS}

    target_missing = target_state is None
    quality_missing = detection_quality is None
    depth_debug_missing = depth_debug is None
    risk_missing = obstacle_risk is None
    uav_missing = uav_state is None
    prev_missing = prev_action is None

    x, y, z = _target_camera(target_state)
    visible = bool(_get(target_state, "visible", _get(target_state, "target_visible", False)))
    distance = _finite(_get(target_state, "depth", _get(target_state, "target_distance", z)), z)

    obs.update(
        {
            "target_visible": 1.0 if visible else 0.0,
            "target_x_camera": x,
            "target_y_camera": y,
            "target_z_camera": z,
            "target_distance": distance,
            "target_u": _finite(_get(target_state, "u", _get(target_state, "target_u", 0.0)), 0.0),
            "target_v": _finite(_get(target_state, "v", _get(target_state, "target_v", 0.0)), 0.0),
            "target_confidence": _finite(_get(target_state, "confidence", _get(detection_quality, "confidence", 0.0)), 0.0),
            "target_visible_ratio_window": _finite(_get(detection_quality, "visible_ratio_window", 0.0), 0.0),
            "target_lost_frames_normalized": _finite(_get(detection_quality, "lost_frames", TARGET_LOST_WINDOW), TARGET_LOST_WINDOW)
            / TARGET_LOST_WINDOW,
            "depth_valid_ratio": _finite(_get(detection_quality, "depth_valid_ratio", 0.0), 0.0),
            "radius_px_smooth_normalized": _finite(_get(detection_quality, "radius_px_smooth", 0.0), 0.0)
            / RADIUS_NORMALIZER_PX,
            "detection_quality_score": _quality_score(_get(detection_quality, "quality", "lost")),
            "sector_far_left_q05": _sector_q05(depth_debug, "far_left"),
            "sector_left_q05": _sector_q05(depth_debug, "left"),
            "sector_front_q05": _sector_q05(depth_debug, "front"),
            "sector_right_q05": _sector_q05(depth_debug, "right"),
            "sector_far_right_q05": _sector_q05(depth_debug, "far_right"),
            "front_q05_smoothed": _finite(_get(depth_debug, "front_q05_smoothed", MAX_DEPTH), MAX_DEPTH),
            "obstacle_area_ratio": _finite(_get(obstacle_risk, "obstacle_area_ratio", 0.0), 0.0),
            "obstacle_danger": 1.0 if bool(_get(obstacle_risk, "danger", _get(obstacle_risk, "obstacle_danger", False))) else 0.0,
            "drone_z": _finite(_get(uav_state, "drone_z", _nested_get(uav_state, ["pose", "position", "z"], 0.0)), 0.0),
            "drone_vx": _finite(_get(uav_state, "drone_vx", _get(uav_state, "vx", 0.0)), 0.0),
            "drone_vy": _finite(_get(uav_state, "drone_vy", _get(uav_state, "vy", 0.0)), 0.0),
            "drone_vz": _finite(_get(uav_state, "drone_vz", _get(uav_state, "vz", 0.0)), 0.0),
            "prev_vx": _finite(_get(prev_action, "prev_vx", _get(prev_action, "vx", 0.0)), 0.0),
            "prev_vz": _finite(_get(prev_action, "prev_vz", _get(prev_action, "vz", 0.0)), 0.0),
            "prev_yaw_rate": _finite(_get(prev_action, "prev_yaw_rate", _get(prev_action, "yaw_rate", 0.0)), 0.0),
        }
    )

    for name in ("target_x_camera", "target_y_camera", "target_z_camera", "target_distance", "target_u", "target_v"):
        missing[name] = bool(target_missing)
    for name in (
        "target_visible_ratio_window",
        "target_lost_frames_normalized",
        "depth_valid_ratio",
        "radius_px_smooth_normalized",
        "detection_quality_score",
    ):
        missing[name] = bool(quality_missing)
    for name in (
        "sector_far_left_q05",
        "sector_left_q05",
        "sector_front_q05",
        "sector_right_q05",
        "sector_far_right_q05",
        "front_q05_smoothed",
    ):
        missing[name] = bool(depth_debug_missing)
    for name in ("obstacle_area_ratio", "obstacle_danger"):
        missing[name] = bool(risk_missing)
    for name in ("drone_z", "drone_vx", "drone_vy", "drone_vz"):
        missing[name] = bool(uav_missing)
    for name in ("prev_vx", "prev_vz", "prev_yaw_rate"):
        missing[name] = bool(prev_missing)
    missing["target_visible"] = bool(target_missing)
    missing["target_confidence"] = bool(target_missing and quality_missing)

    values = np.array([_clip(spec["name"], obs[spec["name"]]) for spec in FIELD_SPECS], dtype=np.float32)
    values = np.nan_to_num(values, nan=0.0, posinf=MAX_DEPTH, neginf=-MAX_DEPTH).astype(np.float32)

    if not return_metadata:
        return values
    return values, {"missing": missing, "field_names": list(FIELD_NAMES)}


def self_check():
    depth_debug = {
        "front_q05_smoothed": 1.2,
        "sectors": {
            "far_left": {"q05": 5.0},
            "left": {"q05": 4.0},
            "front": {"q05": 1.0},
            "right": {"q05": 4.5},
            "far_right": {"q05": 6.0},
        },
    }
    obs, meta = build_observation_v2(
        target_state={
            "visible": True,
            "position_camera": {"x": 0.1, "y": -0.1, "z": 2.0},
            "depth": 2.0,
            "u": 0.1,
            "v": -0.1,
            "confidence": 0.8,
        },
        obstacle_risk={"obstacle_area_ratio": 0.2, "danger": True},
        depth_debug=depth_debug,
        detection_quality={
            "visible_ratio_window": 0.9,
            "lost_frames": 0,
            "depth_valid_ratio": 0.8,
            "radius_px_smooth": 12.0,
            "quality": "good",
        },
        uav_state={"drone_z": 1.3, "drone_vx": 0.1, "drone_vy": 0.0, "drone_vz": 0.0},
        prev_action={"prev_vx": 0.2, "prev_vz": 0.0, "prev_yaw_rate": 0.1},
        return_metadata=True,
    )
    assert obs.shape == (OBSERVATION_V2_DIM,)
    assert np.all(np.isfinite(obs))
    assert meta["missing"]["sector_front_q05"] is False
    fallback, fallback_meta = build_observation_v2(return_metadata=True)
    assert fallback.shape == (OBSERVATION_V2_DIM,)
    assert np.all(np.isfinite(fallback))
    assert fallback_meta["missing"]["sector_front_q05"] is True
    return True


if __name__ == "__main__":
    print("observation_v2 self_check={}".format(self_check()))
