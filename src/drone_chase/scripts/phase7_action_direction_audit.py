#!/usr/bin/env python3

import argparse
import csv
import math
import os
import sys
import time

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "envs"))
if ENV_DIR not in sys.path:
    sys.path.insert(0, ENV_DIR)

from gazebo_chase_env import GazeboChaseEnv


DEFAULT_CONFIG = "/home/whk/vf_ws/src/drone_chase/config/phase7_ppo_world0_curriculum_v1.yaml"
DEFAULT_OUTPUT = "/home/whk/vf_ws/outputs/phase7/action_direction_audit.csv"


def load_yaml(path):
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read config files") from exc
    with open(path, "r") as handle:
        return yaml.safe_load(handle) or {}


def env_kwargs_from_config(config, seed=None):
    env_cfg = dict(config.get("env", {}))
    env_cfg.setdefault("world_type", "world_0")
    env_cfg.setdefault("reset_mode", "episode_soft")
    env_cfg["max_episode_steps"] = max(int(env_cfg.get("max_episode_steps", 300)), 120)
    if seed is not None:
        env_cfg["seed"] = int(seed)
    return {key: value for key, value in env_cfg.items() if value is not None}


class CommandMonitor:
    def __init__(self):
        import rospy
        from geometry_msgs.msg import TwistStamped

        self.raw = None
        self.filtered = None
        self.published = None
        rospy.Subscriber("/safety_filter/debug_cmd_raw", TwistStamped, self._raw_cb, queue_size=1)
        rospy.Subscriber("/safety_filter/debug_cmd_filtered", TwistStamped, self._filtered_cb, queue_size=1)
        rospy.Subscriber("/mavros/setpoint_velocity/cmd_vel", TwistStamped, self._published_cb, queue_size=1)

    def _raw_cb(self, msg):
        self.raw = msg

    def _filtered_cb(self, msg):
        self.filtered = msg

    def _published_cb(self, msg):
        self.published = msg

    @staticmethod
    def values(msg):
        if msg is None:
            return {
                "vx": math.nan,
                "vy": math.nan,
                "vz": math.nan,
                "yaw": math.nan,
            }
        return {
            "vx": float(msg.twist.linear.x),
            "vy": float(msg.twist.linear.y),
            "vz": float(msg.twist.linear.z),
            "yaw": float(msg.twist.angular.z),
        }


def snapshot(info, monitor):
    raw = monitor.values(monitor.raw)
    filtered = monitor.values(monitor.filtered)
    published = monitor.values(monitor.published)
    return {
        "x": float(info.get("drone_x", math.nan)),
        "y": float(info.get("drone_y", math.nan)),
        "z": float(info.get("drone_z", math.nan)),
        "yaw": float(info.get("drone_yaw", math.nan)),
        "target_depth": float(info.get("target_distance", math.nan)),
        "target_u": float(info.get("target_u", math.nan)),
        "target_v": float(info.get("target_v", math.nan)),
        "safety_mode": str(info.get("safety_mode", "")),
        "mavros_mode": str(info.get("mavros_mode", "")),
        "mavros_armed": bool(info.get("mavros_armed", False)),
        "raw": raw,
        "filtered": filtered,
        "published": published,
    }


def segment_specs():
    return [
        ("hover", np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)),
        ("forward", np.array([0.5, 0.0, 0.0, 0.0], dtype=np.float32)),
        ("backward", np.array([-0.5, 0.0, 0.0, 0.0], dtype=np.float32)),
        ("yaw_positive", np.array([0.0, 0.0, 0.0, 0.5], dtype=np.float32)),
        ("yaw_negative", np.array([0.0, 0.0, 0.0, -0.5], dtype=np.float32)),
        ("up", np.array([0.0, 0.0, 0.5, 0.0], dtype=np.float32)),
        ("down", np.array([0.0, 0.0, -0.5, 0.0], dtype=np.float32)),
    ]


def row_fieldnames():
    return [
        "segment",
        "action_vx",
        "action_vy",
        "action_vz",
        "action_yaw",
        "steps",
        "terminated",
        "truncated",
        "terminal_reason",
        "raw_timeout_seen",
        "emergency_seen",
        "depth_stop_seen",
        "height_violation",
        "start_x",
        "start_y",
        "start_z",
        "start_yaw",
        "end_x",
        "end_y",
        "end_z",
        "end_yaw",
        "delta_x",
        "delta_y",
        "delta_z",
        "delta_yaw",
        "start_target_depth",
        "end_target_depth",
        "delta_target_depth",
        "start_target_u",
        "end_target_u",
        "delta_target_u",
        "start_target_v",
        "end_target_v",
        "delta_target_v",
        "start_safety_mode",
        "end_safety_mode",
        "end_mavros_mode",
        "end_mavros_armed",
        "raw_vx",
        "raw_vy",
        "raw_vz",
        "raw_yaw",
        "filtered_vx",
        "filtered_vy",
        "filtered_vz",
        "filtered_yaw",
        "published_vx",
        "published_vy",
        "published_vz",
        "published_yaw",
    ]


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Phase 7.1 fixed-action direction audit.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=10)
    return parser


def main():
    args = build_arg_parser().parse_args()
    config = load_yaml(args.config)
    env = GazeboChaseEnv(**env_kwargs_from_config(config, seed=args.seed))
    monitor = CommandMonitor()
    rows = []

    try:
        for segment, action in segment_specs():
            obs, info = env.reset()
            if not bool(info.get("reset_success", False)):
                raise RuntimeError("segment {} reset failed: {}".format(segment, info))
            time.sleep(0.2)
            start = snapshot(info, monitor)
            steps = max(1, int(round(float(args.duration) / float(env.step_dt))))
            terminated = False
            truncated = False
            terminal_reason = ""
            raw_timeout_seen = False
            emergency_seen = False
            depth_stop_seen = False
            height_violation = False
            end_info = info

            for _step in range(steps):
                obs, reward, terminated, truncated, end_info = env.step(action)
                if obs.shape != (20,) or not np.all(np.isfinite(obs)) or not np.isfinite(reward):
                    raise RuntimeError("invalid env output in segment {}".format(segment))
                mode = str(end_info.get("safety_mode", ""))
                raw_timeout_seen = raw_timeout_seen or "RAW_TIMEOUT" in mode
                emergency_seen = emergency_seen or "EMERGENCY_AVOID" in mode
                depth_stop_seen = depth_stop_seen or "DEPTH_STOP" in mode
                height_violation = height_violation or bool(end_info.get("height_violation", False))
                terminal_reason = str(end_info.get("terminal_reason", ""))
                if terminated or truncated:
                    break

            end = snapshot(end_info, monitor)
            raw = end["raw"]
            filtered = end["filtered"]
            published = end["published"]
            row = {
                "segment": segment,
                "action_vx": float(action[0]),
                "action_vy": float(action[1]),
                "action_vz": float(action[2]),
                "action_yaw": float(action[3]),
                "steps": _step + 1,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "terminal_reason": terminal_reason,
                "raw_timeout_seen": bool(raw_timeout_seen),
                "emergency_seen": bool(emergency_seen),
                "depth_stop_seen": bool(depth_stop_seen),
                "height_violation": bool(height_violation),
                "start_x": start["x"],
                "start_y": start["y"],
                "start_z": start["z"],
                "start_yaw": start["yaw"],
                "end_x": end["x"],
                "end_y": end["y"],
                "end_z": end["z"],
                "end_yaw": end["yaw"],
                "delta_x": end["x"] - start["x"],
                "delta_y": end["y"] - start["y"],
                "delta_z": end["z"] - start["z"],
                "delta_yaw": end["yaw"] - start["yaw"],
                "start_target_depth": start["target_depth"],
                "end_target_depth": end["target_depth"],
                "delta_target_depth": end["target_depth"] - start["target_depth"],
                "start_target_u": start["target_u"],
                "end_target_u": end["target_u"],
                "delta_target_u": end["target_u"] - start["target_u"],
                "start_target_v": start["target_v"],
                "end_target_v": end["target_v"],
                "delta_target_v": end["target_v"] - start["target_v"],
                "start_safety_mode": start["safety_mode"],
                "end_safety_mode": end["safety_mode"],
                "end_mavros_mode": end["mavros_mode"],
                "end_mavros_armed": end["mavros_armed"],
                "raw_vx": raw["vx"],
                "raw_vy": raw["vy"],
                "raw_vz": raw["vz"],
                "raw_yaw": raw["yaw"],
                "filtered_vx": filtered["vx"],
                "filtered_vy": filtered["vy"],
                "filtered_vz": filtered["vz"],
                "filtered_yaw": filtered["yaw"],
                "published_vx": published["vx"],
                "published_vy": published["vy"],
                "published_vz": published["vz"],
                "published_yaw": published["yaw"],
            }
            rows.append(row)
            print("segment={} row={}".format(segment, row))
    finally:
        env.close()

    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=row_fieldnames())
        writer.writeheader()
        writer.writerows(rows)
    print("wrote {}".format(output_path))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("phase7_action_direction_audit failed: {}".format(exc), file=sys.stderr)
        sys.exit(1)
