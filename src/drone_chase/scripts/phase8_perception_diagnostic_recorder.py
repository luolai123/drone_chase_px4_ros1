#!/usr/bin/env python3

import argparse
import csv
import json
import math
import os
import threading
import time
from datetime import datetime

import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String

from drone_chase.msg import DepthRisk, TargetState


def _finite(value, default=math.nan):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(default)
    return value if math.isfinite(value) else float(default)


def _mean(values):
    values = [float(v) for v in values if math.isfinite(float(v))]
    return float(sum(values)) / float(len(values)) if values else 0.0


def _min(values):
    values = [float(v) for v in values if math.isfinite(float(v))]
    return float(min(values)) if values else float("inf")


class PerceptionDiagnosticRecorder:
    CSV_FIELDS = [
        "time",
        "target_visible",
        "target_depth",
        "target_u",
        "target_v",
        "target_confidence",
        "target_quality",
        "target_visible_ratio_window",
        "target_lost_frames",
        "depth_valid_ratio",
        "radius_px",
        "front_q05",
        "left_q05",
        "right_q05",
        "obstacle_area_ratio",
        "obstacle_danger",
        "front_q05_smoothed",
        "sector_far_left_q05",
        "sector_left_q05",
        "sector_front_q05",
        "sector_right_q05",
        "sector_far_right_q05",
        "roi_valid_ratio",
        "drone_z",
        "safety_mode",
    ]

    def __init__(self, output_dir, rate_hz=10.0):
        rospy.init_node("phase8_perception_diagnostic_recorder")

        self.output_dir = output_dir
        self.rate_hz = float(rate_hz)
        self.lock = threading.Lock()

        self.latest_target = None
        self.latest_risk = None
        self.latest_depth_debug = None
        self.latest_quality_debug = None
        self.latest_pose = None
        self.latest_safety_mode = None

        self.rows = []
        self.prev_target_visible = None
        self.prev_quality = None
        self.target_lost_event_count = 0
        self.weak_depth_event_count = 0
        self.small_target_event_count = 0

        os.makedirs(self.output_dir, exist_ok=True)
        self.csv_path = os.path.join(self.output_dir, "perception_diagnostic_log.csv")
        self.summary_path = os.path.join(self.output_dir, "perception_diagnostic_summary.json")

        self._csv_handle = open(self.csv_path, "w", newline="")
        self._csv_writer = csv.DictWriter(self._csv_handle, fieldnames=list(self.CSV_FIELDS))
        self._csv_writer.writeheader()
        self._csv_handle.flush()

        rospy.Subscriber("/target/state", TargetState, self._target_cb, queue_size=10)
        rospy.Subscriber("/obstacle/risk", DepthRisk, self._risk_cb, queue_size=10)
        rospy.Subscriber("/debug/depth_risk_extended", String, self._depth_debug_cb, queue_size=10)
        rospy.Subscriber("/debug/target_detection_quality", String, self._quality_debug_cb, queue_size=10)
        rospy.Subscriber("/mavros/local_position/pose", PoseStamped, self._pose_cb, queue_size=10)
        rospy.Subscriber("/safety_filter/mode", String, self._mode_cb, queue_size=10)

        self._timer = rospy.Timer(rospy.Duration(1.0 / max(1e-3, self.rate_hz)), self._on_timer)
        rospy.on_shutdown(self._on_shutdown)
        rospy.loginfo("phase8_perception_diagnostic_recorder ready; writing %s", self.csv_path)

    def _target_cb(self, msg):
        with self.lock:
            self.latest_target = msg

    def _risk_cb(self, msg):
        with self.lock:
            self.latest_risk = msg

    def _pose_cb(self, msg):
        with self.lock:
            self.latest_pose = msg

    def _mode_cb(self, msg):
        with self.lock:
            self.latest_safety_mode = msg.data

    def _parse_json_msg(self, msg):
        if msg is None:
            return None
        try:
            return json.loads(msg.data)
        except Exception:
            return None

    def _depth_debug_cb(self, msg):
        payload = self._parse_json_msg(msg)
        with self.lock:
            self.latest_depth_debug = payload

    def _quality_debug_cb(self, msg):
        payload = self._parse_json_msg(msg)
        with self.lock:
            self.latest_quality_debug = payload

    def _extract_sector_q05(self, depth_debug, name):
        if not depth_debug:
            return math.nan
        sectors = depth_debug.get("sectors") or {}
        entry = sectors.get(name) or {}
        return _finite(entry.get("q05"))

    def _on_timer(self, _event):
        now = rospy.Time.now().to_sec()
        with self.lock:
            target = self.latest_target
            risk = self.latest_risk
            depth_debug = self.latest_depth_debug
            quality_debug = self.latest_quality_debug
            pose = self.latest_pose
            safety_mode = self.latest_safety_mode

        target_visible = False if target is None else bool(target.visible)
        target_depth = math.nan if target is None else _finite(target.depth)
        target_u = math.nan if target is None else _finite(target.u)
        target_v = math.nan if target is None else _finite(target.v)
        target_conf = math.nan if target is None else _finite(target.confidence)
        radius_px = math.nan if target is None else _finite(target.radius_px)

        front_q05 = math.nan if risk is None else _finite(risk.front_q05_depth)
        left_q05 = math.nan if risk is None else _finite(risk.left_q05_depth)
        right_q05 = math.nan if risk is None else _finite(risk.right_q05_depth)
        obstacle_area_ratio = math.nan if risk is None else _finite(risk.obstacle_area_ratio)
        obstacle_danger = False if risk is None else bool(risk.danger)

        drone_z = math.nan
        if pose is not None:
            drone_z = _finite(pose.pose.position.z)

        target_quality = ""
        visible_ratio_window = math.nan
        lost_frames = math.nan
        depth_valid_ratio = math.nan
        if quality_debug:
            target_quality = str(quality_debug.get("quality", ""))
            visible_ratio_window = _finite(quality_debug.get("visible_ratio_window"))
            lost_frames = _finite(quality_debug.get("lost_frames"))
            depth_valid_ratio = _finite(quality_debug.get("depth_valid_ratio"))

        front_q05_smoothed = math.nan
        roi_valid_ratio = math.nan
        if depth_debug:
            front_q05_smoothed = _finite(depth_debug.get("front_q05_smoothed"))
            roi_valid_ratio = _finite(depth_debug.get("roi_valid_ratio"))

        row = {
            "time": float(now),
            "target_visible": int(bool(target_visible)),
            "target_depth": float(target_depth),
            "target_u": float(target_u),
            "target_v": float(target_v),
            "target_confidence": float(target_conf),
            "target_quality": target_quality,
            "target_visible_ratio_window": float(visible_ratio_window),
            "target_lost_frames": float(lost_frames),
            "depth_valid_ratio": float(depth_valid_ratio),
            "radius_px": float(radius_px),
            "front_q05": float(front_q05),
            "left_q05": float(left_q05),
            "right_q05": float(right_q05),
            "obstacle_area_ratio": float(obstacle_area_ratio),
            "obstacle_danger": int(bool(obstacle_danger)),
            "front_q05_smoothed": float(front_q05_smoothed),
            "sector_far_left_q05": float(self._extract_sector_q05(depth_debug, "far_left")),
            "sector_left_q05": float(self._extract_sector_q05(depth_debug, "left")),
            "sector_front_q05": float(self._extract_sector_q05(depth_debug, "front")),
            "sector_right_q05": float(self._extract_sector_q05(depth_debug, "right")),
            "sector_far_right_q05": float(self._extract_sector_q05(depth_debug, "far_right")),
            "roi_valid_ratio": float(roi_valid_ratio),
            "drone_z": float(drone_z),
            "safety_mode": "" if safety_mode is None else str(safety_mode),
        }

        self._csv_writer.writerow(row)
        self._csv_handle.flush()
        self.rows.append(row)

        if self.prev_target_visible is not None and self.prev_target_visible and not target_visible:
            self.target_lost_event_count += 1
        self.prev_target_visible = bool(target_visible)

        if target_quality and target_quality != self.prev_quality:
            if target_quality == "weak_depth":
                self.weak_depth_event_count += 1
            if target_quality == "small_target":
                self.small_target_event_count += 1
        self.prev_quality = target_quality

    def _on_shutdown(self):
        try:
            if self._timer is not None:
                self._timer.shutdown()
        except Exception:
            pass

        try:
            self._csv_handle.flush()
            self._csv_handle.close()
        except Exception:
            pass

        rows = list(self.rows)
        if not rows:
            summary = {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "total_samples": 0,
            }
            with open(self.summary_path, "w") as handle:
                json.dump(summary, handle, indent=2, sort_keys=True)
            return

        target_visible_ratio = float(sum(int(r.get("target_visible", 0)) for r in rows)) / float(len(rows))
        target_depths = [r.get("target_depth", math.nan) for r in rows]
        front_q05s = [r.get("front_q05", math.nan) for r in rows]
        obstacle_danger_ratio = float(sum(int(r.get("obstacle_danger", 0)) for r in rows)) / float(len(rows))
        emergency_steps = 0
        depth_stop_steps = 0
        for row in rows:
            mode = str(row.get("safety_mode", "") or "")
            if "EMERGENCY_AVOID" in mode:
                emergency_steps += 1
            if "DEPTH_STOP" in mode:
                depth_stop_steps += 1

        summary = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "output_dir": self.output_dir,
            "total_samples": int(len(rows)),
            "target_visible_ratio": float(target_visible_ratio),
            "mean_target_depth": float(_mean(target_depths)),
            "min_target_depth": float(_min(target_depths)),
            "mean_front_q05": float(_mean(front_q05s)),
            "min_front_q05": float(_min(front_q05s)),
            "obstacle_danger_ratio": float(obstacle_danger_ratio),
            "target_lost_event_count": int(self.target_lost_event_count),
            "weak_depth_event_count": int(self.weak_depth_event_count),
            "small_target_event_count": int(self.small_target_event_count),
            "safety_emergency_steps": int(emergency_steps),
            "safety_depth_stop_steps": int(depth_stop_steps),
        }
        with open(self.summary_path, "w") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Phase 8 perception diagnostic recorder (CSV + summary).")
    parser.add_argument(
        "--output-dir",
        default="/home/whk/vf_ws/outputs/phase8/perception_diagnostics",
    )
    parser.add_argument("--rate-hz", type=float, default=10.0)
    return parser


def main():
    args = build_arg_parser().parse_args()
    PerceptionDiagnosticRecorder(output_dir=args.output_dir, rate_hz=args.rate_hz)
    rospy.spin()


if __name__ == "__main__":
    main()

