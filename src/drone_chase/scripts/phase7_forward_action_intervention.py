#!/usr/bin/env python3

import argparse
import csv
import math
import os
import sys
import threading

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "envs"))
if ENV_DIR not in sys.path:
    sys.path.insert(0, ENV_DIR)

from gazebo_chase_env import GazeboChaseEnv


DEFAULT_CONFIG = "/home/whk/vf_ws/src/drone_chase/config/phase7_ppo_world0_curriculum_v2.yaml"
DEFAULT_OUTPUT = "/home/whk/vf_ws/outputs/phase7/forward_action_intervention.csv"


def load_yaml(path):
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required") from exc
    with open(path, "r") as handle:
        return yaml.safe_load(handle) or {}


def env_kwargs_from_config(config, max_steps=None):
    env_cfg = dict(config.get("env", {}))
    env_cfg.setdefault("world_type", "world_0")
    env_cfg.setdefault("reset_mode", "episode_soft")
    if max_steps is not None:
        env_cfg["max_episode_steps"] = int(max_steps)
    return {key: value for key, value in env_cfg.items() if value is not None}


def clamp(value, low, high):
    return max(low, min(high, value))


def normalize_body_cmd(vx, vy, vz, yaw_rate):
    action_vx = vx / 0.5 if vx >= 0.0 else vx / 0.2
    return np.clip(
        np.array([action_vx, vy / 0.3, vz / 0.25, yaw_rate / 0.6], dtype=np.float32),
        -1.0,
        1.0,
    )


def expert_action(obs):
    target_visible = bool(obs[0] > 0.5)
    target_distance = float(obs[4])
    target_u = float(obs[5])
    target_v = float(obs[6])
    if target_visible:
        vx = 0.35 * (target_distance - 0.8)
        vy = 0.0
        vz = -0.35 * target_v
        yaw_rate = -0.8 * target_u
    else:
        vx = 0.0
        vy = 0.0
        vz = 0.0
        yaw_rate = 0.3
    vx = clamp(vx, -0.2, 0.5)
    vy = clamp(vy, -0.3, 0.3)
    vz = clamp(vz, -0.25, 0.25)
    yaw_rate = clamp(yaw_rate, -0.6, 0.6)
    return normalize_body_cmd(vx, vy, vz, yaw_rate)


def expert_yaw_forward_action(obs):
    target_visible = bool(obs[0] > 0.5)
    target_u = float(obs[5])
    target_v = float(obs[6])
    if target_visible:
        return np.array(
            [
                0.5,
                0.0,
                clamp((-0.35 * target_v) / 0.25, -1.0, 1.0),
                clamp((-0.8 * target_u) / 0.6, -1.0, 1.0),
            ],
            dtype=np.float32,
        )
    return np.array([0.0, 0.0, 0.0, 0.5], dtype=np.float32)


def finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return math.nan
    return value if math.isfinite(value) else math.nan


def twist_values(msg, prefix, frame_suffix):
    if msg is None:
        return {
            "{}_vx_{}".format(prefix, frame_suffix): math.nan,
            "{}_vy_{}".format(prefix, frame_suffix): math.nan,
            "{}_vz_{}".format(prefix, frame_suffix): math.nan,
            "{}_yaw_{}".format(prefix, frame_suffix): math.nan,
        }
    return {
        "{}_vx_{}".format(prefix, frame_suffix): float(msg.twist.linear.x),
        "{}_vy_{}".format(prefix, frame_suffix): float(msg.twist.linear.y),
        "{}_vz_{}".format(prefix, frame_suffix): float(msg.twist.linear.z),
        "{}_yaw_{}".format(prefix, frame_suffix): float(msg.twist.angular.z),
    }


class CommandMonitor:
    def __init__(self):
        import rospy
        from geometry_msgs.msg import TwistStamped

        self.lock = threading.Lock()
        self.debug_raw = None
        self.debug_filtered = None
        self.published = None
        rospy.Subscriber("/safety_filter/debug_cmd_raw", TwistStamped, self._raw_cb, queue_size=1)
        rospy.Subscriber("/safety_filter/debug_cmd_filtered", TwistStamped, self._filtered_cb, queue_size=1)
        rospy.Subscriber("/mavros/setpoint_velocity/cmd_vel", TwistStamped, self._published_cb, queue_size=1)

    def _raw_cb(self, msg):
        with self.lock:
            self.debug_raw = msg

    def _filtered_cb(self, msg):
        with self.lock:
            self.debug_filtered = msg

    def _published_cb(self, msg):
        with self.lock:
            self.published = msg

    def snapshot(self):
        with self.lock:
            debug_raw = self.debug_raw
            debug_filtered = self.debug_filtered
            published = self.published
        row = {}
        row.update(twist_values(debug_raw, "debug_raw", "body"))
        row.update(twist_values(debug_filtered, "debug_filtered", "body"))
        row.update(twist_values(published, "published", "world"))
        return row


def row_fieldnames():
    return [
        "mode_name",
        "episode",
        "step",
        "action_vx",
        "raw_vx_body",
        "debug_raw_vx_body",
        "debug_filtered_vx_body",
        "published_vx_world",
        "published_vy_world",
        "target_depth",
        "target_depth_delta",
        "target_visible",
        "target_u",
        "target_v",
        "drone_z",
        "front_q05",
        "safety_mode",
        "done",
        "done_reason",
        "reward",
    ]


def mode_action(mode_name, obs):
    if mode_name == "zero":
        return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    if mode_name == "small_forward":
        return np.array([0.2, 0.0, 0.0, 0.0], dtype=np.float32)
    if mode_name == "medium_forward":
        return np.array([0.5, 0.0, 0.0, 0.0], dtype=np.float32)
    if mode_name == "expert_yaw_forward":
        return expert_yaw_forward_action(obs)
    if mode_name == "expert_full":
        return expert_action(obs)
    raise ValueError("unknown mode_name={}".format(mode_name))


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Fixed forward-action intervention through GazeboChaseEnv.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--modes", nargs="*", default=["zero", "small_forward", "medium_forward", "expert_yaw_forward", "expert_full"])
    return parser


def main():
    args = build_arg_parser().parse_args()
    import rospy

    config = load_yaml(args.config)
    env = GazeboChaseEnv(**env_kwargs_from_config(config, max_steps=args.max_steps))
    monitor = CommandMonitor()
    rows = []
    try:
        for episode, mode_name in enumerate(args.modes):
            obs, info = env.reset()
            if not bool(info.get("reset_success", False)):
                raise RuntimeError("{} reset failed: {}".format(mode_name, info))
            previous_depth = finite(info.get("target_distance", obs[4]))
            for step in range(int(args.max_steps)):
                action = mode_action(mode_name, obs)
                _, mapped_body = env._map_action(action)
                obs, reward, terminated, truncated, info = env.step(action)
                current_depth = finite(info.get("target_distance", math.nan))
                depth_delta = current_depth - previous_depth if math.isfinite(previous_depth) and math.isfinite(current_depth) else math.nan
                previous_depth = current_depth
                trace = monitor.snapshot()
                done = bool(terminated or truncated)
                rows.append(
                    {
                        "mode_name": mode_name,
                        "episode": episode,
                        "step": step,
                        "action_vx": float(action[0]),
                        "raw_vx_body": float(mapped_body[0]),
                        "debug_raw_vx_body": trace["debug_raw_vx_body"],
                        "debug_filtered_vx_body": trace["debug_filtered_vx_body"],
                        "published_vx_world": trace["published_vx_world"],
                        "published_vy_world": trace["published_vy_world"],
                        "target_depth": current_depth,
                        "target_depth_delta": depth_delta,
                        "target_visible": bool(info.get("target_visible", False)),
                        "target_u": finite(info.get("target_u", math.nan)),
                        "target_v": finite(info.get("target_v", math.nan)),
                        "drone_z": finite(info.get("drone_z", math.nan)),
                        "front_q05": finite(info.get("front_q05_depth", math.nan)),
                        "safety_mode": str(info.get("safety_mode", "")),
                        "done": done,
                        "done_reason": str(info.get("terminal_reason", "")),
                        "reward": float(reward),
                    }
                )
                if done or rospy.is_shutdown():
                    break
    finally:
        env.close()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=row_fieldnames())
        writer.writeheader()
        writer.writerows(rows)
    print("wrote {}".format(args.output))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("phase7_forward_action_intervention failed: {}".format(exc), file=sys.stderr)
        sys.exit(1)
