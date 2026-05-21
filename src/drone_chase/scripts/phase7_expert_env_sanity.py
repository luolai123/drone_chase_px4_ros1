#!/usr/bin/env python3

import argparse
import csv
import math
import os
import sys

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "envs"))
if ENV_DIR not in sys.path:
    sys.path.insert(0, ENV_DIR)

from gazebo_chase_env import GazeboChaseEnv


DEFAULT_CONFIG = "/home/whk/vf_ws/src/drone_chase/config/phase7_ppo_world0_curriculum_v1.yaml"
DEFAULT_OUTPUT = "/home/whk/vf_ws/outputs/phase7/expert_env_sanity.csv"


def load_yaml(path):
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read config files") from exc
    with open(path, "r") as handle:
        return yaml.safe_load(handle) or {}


def env_kwargs_from_config(config, max_steps=None, seed=None):
    env_cfg = dict(config.get("env", {}))
    env_cfg.setdefault("world_type", "world_0")
    env_cfg.setdefault("reset_mode", "episode_soft")
    if max_steps is not None:
        env_cfg["max_episode_steps"] = int(max_steps)
    if seed is not None:
        env_cfg["seed"] = int(seed)
    return {key: value for key, value in env_cfg.items() if value is not None}


def clamp(value, low, high):
    return max(low, min(high, value))


def normalize_body_cmd(vx, vy, vz, yaw_rate):
    if vx >= 0.0:
        a_vx = vx / 0.5
    else:
        a_vx = vx / 0.2
    return np.clip(
        np.array(
            [
                a_vx,
                vy / 0.3,
                vz / 0.25,
                yaw_rate / 0.6,
            ],
            dtype=np.float32,
        ),
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


class CommandMonitor:
    def __init__(self):
        import rospy
        from geometry_msgs.msg import TwistStamped

        self.raw = None
        self.filtered = None
        rospy.Subscriber("/safety_filter/debug_cmd_raw", TwistStamped, self._raw_cb, queue_size=1)
        rospy.Subscriber("/safety_filter/debug_cmd_filtered", TwistStamped, self._filtered_cb, queue_size=1)

    def _raw_cb(self, msg):
        self.raw = msg

    def _filtered_cb(self, msg):
        self.filtered = msg

    @staticmethod
    def _vx(msg):
        return math.nan if msg is None else float(msg.twist.linear.x)

    @staticmethod
    def _vz(msg):
        return math.nan if msg is None else float(msg.twist.linear.z)

    @staticmethod
    def _yaw(msg):
        return math.nan if msg is None else float(msg.twist.angular.z)

    def raw_vx(self):
        return self._vx(self.raw)

    def filtered_vx(self):
        return self._vx(self.filtered)


def row_fieldnames():
    return [
        "episode",
        "step",
        "reward",
        "done",
        "success",
        "timeout",
        "target_visible",
        "target_depth",
        "target_u",
        "target_v",
        "front_q05",
        "safety_mode",
        "mavros_connected",
        "mavros_mode",
        "mavros_armed",
        "height_violation",
        "action_vx",
        "action_vz",
        "action_yaw",
        "raw_vx",
        "filtered_vx",
        "drone_z",
        "terminal_reason",
    ]


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Phase 7.1 expert-through-GazeboChaseEnv sanity test.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=10)
    return parser


def main():
    args = build_arg_parser().parse_args()
    config = load_yaml(args.config)
    env = GazeboChaseEnv(**env_kwargs_from_config(config, max_steps=args.max_steps, seed=args.seed))
    monitor = CommandMonitor()
    rows = []
    episode_summaries = []

    try:
        for episode in range(args.episodes):
            obs, info = env.reset()
            if not bool(info.get("reset_success", False)):
                raise RuntimeError("episode {} reset failed: {}".format(episode, info))
            first_depth = float(info.get("target_distance", obs[4]))
            depths = [first_depth]
            success = False
            timeout = False
            terminal_reason = ""

            for step in range(args.max_steps):
                if obs.shape != (20,) or not np.all(np.isfinite(obs)):
                    raise RuntimeError("invalid observation at episode {} step {}".format(episode, step))
                action = expert_action(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                if not np.all(np.isfinite(obs)) or not np.isfinite(reward):
                    raise RuntimeError("non-finite env output at episode {} step {}".format(episode, step))

                _, raw_body_cmd = env._map_action(action)
                done = bool(terminated or truncated)
                success = bool(info.get("success", False))
                timeout = bool(info.get("timeout", False))
                terminal_reason = str(info.get("terminal_reason", ""))
                depths.append(float(info.get("target_distance", math.nan)))
                rows.append(
                    {
                        "episode": episode,
                        "step": step,
                        "reward": float(reward),
                        "done": done,
                        "success": success,
                        "timeout": timeout,
                        "target_visible": bool(info.get("target_visible", False)),
                        "target_depth": float(info.get("target_distance", math.nan)),
                        "target_u": float(info.get("target_u", math.nan)),
                        "target_v": float(info.get("target_v", math.nan)),
                        "front_q05": float(info.get("front_q05_depth", math.nan)),
                        "safety_mode": str(info.get("safety_mode", "")),
                        "mavros_connected": bool(info.get("mavros_connected", False)),
                        "mavros_mode": str(info.get("mavros_mode", "")),
                        "mavros_armed": bool(info.get("mavros_armed", False)),
                        "height_violation": bool(info.get("height_violation", False)),
                        "action_vx": float(action[0]),
                        "action_vz": float(action[2]),
                        "action_yaw": float(action[3]),
                        "raw_vx": float(raw_body_cmd[0]) if math.isnan(monitor.raw_vx()) else monitor.raw_vx(),
                        "filtered_vx": monitor.filtered_vx(),
                        "drone_z": float(info.get("drone_z", math.nan)),
                        "terminal_reason": terminal_reason,
                    }
                )
                if done:
                    break

            finite_depths = [value for value in depths if math.isfinite(value)]
            final_depth = finite_depths[-1] if finite_depths else math.nan
            median_depth = float(np.median(finite_depths)) if finite_depths else math.nan
            episode_summaries.append(
                {
                    "episode": episode,
                    "success": success,
                    "timeout": timeout,
                    "terminal_reason": terminal_reason,
                    "first_depth": first_depth,
                    "median_depth": median_depth,
                    "final_depth": final_depth,
                    "steps": step + 1,
                }
            )
            print("episode={} summary={}".format(episode, episode_summaries[-1]))
    finally:
        env.close()

    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=row_fieldnames())
        writer.writeheader()
        writer.writerows(rows)

    success_count = sum(1 for row in episode_summaries if row["success"])
    print("wrote {}".format(output_path))
    print("expert_success_rate={}/{}".format(success_count, len(episode_summaries)))
    if success_count < 4:
        return 2
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print("phase7_expert_env_sanity failed: {}".format(exc), file=sys.stderr)
        sys.exit(1)
