#!/usr/bin/env python3

import argparse
import csv
import json
import math
import os
import sys
import threading


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "envs"))
if ENV_DIR not in sys.path:
    sys.path.insert(0, ENV_DIR)

DEFAULT_RUN_DIR = "/home/whk/vf_ws/outputs/phase7/world0_20k_rewardfix_v2_run2"


def require_training_deps():
    missing = []
    try:
        import stable_baselines3  # noqa: F401
    except ImportError:
        missing.append("stable-baselines3")
    try:
        import gymnasium  # noqa: F401
    except ImportError:
        missing.append("gymnasium")
    try:
        import torch  # noqa: F401
    except ImportError:
        missing.append("torch")
    if missing:
        raise RuntimeError("Missing Phase 7 dependencies: {}".format(", ".join(missing)))


def load_yaml(path):
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required") from exc
    with open(path, "r") as handle:
        return yaml.safe_load(handle) or {}


def env_kwargs_from_config(config):
    kwargs = dict(config.get("env", {}))
    kwargs.setdefault("world_type", "world_0")
    kwargs.setdefault("reset_mode", "episode_soft")
    return {key: value for key, value in kwargs.items() if value is not None}


def finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return math.nan
    return value if math.isfinite(value) else math.nan


def nan_cmd(prefix, frame_suffix):
    return {
        "{}_vx_{}".format(prefix, frame_suffix): math.nan,
        "{}_vy_{}".format(prefix, frame_suffix): math.nan,
        "{}_vz_{}".format(prefix, frame_suffix): math.nan,
        "{}_yaw_{}".format(prefix, frame_suffix): math.nan,
    }


def twist_values(msg, prefix, frame_suffix):
    if msg is None:
        return nan_cmd(prefix, frame_suffix)
    return {
        "{}_vx_{}".format(prefix, frame_suffix): float(msg.twist.linear.x),
        "{}_vy_{}".format(prefix, frame_suffix): float(msg.twist.linear.y),
        "{}_vz_{}".format(prefix, frame_suffix): float(msg.twist.linear.z),
        "{}_yaw_{}".format(prefix, frame_suffix): float(msg.twist.angular.z),
    }


class CommandTraceMonitor:
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
        "episode",
        "step",
        "obs_target_visible",
        "obs_target_depth",
        "obs_target_u",
        "obs_target_v",
        "obs_front_q05",
        "obs_obstacle_danger",
        "obs_drone_z",
        "policy_action_vx",
        "policy_action_vy",
        "policy_action_vz",
        "policy_action_yaw",
        "mapped_raw_vx_body",
        "mapped_raw_vy_body",
        "mapped_raw_vz_body",
        "mapped_raw_yaw",
        "debug_raw_vx_body",
        "debug_raw_vy_body",
        "debug_raw_vz_body",
        "debug_raw_yaw_body",
        "debug_filtered_vx_body",
        "debug_filtered_vy_body",
        "debug_filtered_vz_body",
        "debug_filtered_yaw_body",
        "published_vx_world",
        "published_vy_world",
        "published_vz_world",
        "published_yaw_rate",
        "safety_mode",
        "px4_mode",
        "armed",
        "target_depth_delta",
        "reward",
        "done",
        "done_reason",
    ]


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Detailed command trace eval for Phase 7 best checkpoint.")
    parser.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--output", default=None)
    return parser


def main():
    args = build_arg_parser().parse_args()
    require_training_deps()

    import numpy as np
    import rospy
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from gazebo_chase_env import GazeboChaseEnv

    run_dir = os.path.abspath(args.run_dir)
    best_path = os.path.join(run_dir, "best_checkpoint.json")
    config_path = os.path.join(run_dir, "config_effective.yaml")
    if not os.path.exists(best_path):
        raise FileNotFoundError(best_path)
    if not os.path.exists(config_path):
        raise FileNotFoundError(config_path)
    best = json.load(open(best_path))
    config = load_yaml(config_path)
    model_path = best["model_path"]
    vecnormalize_path = best["vecnormalize_path"]
    output_path = args.output or os.path.join(run_dir, "best_command_trace_eval.csv")

    monitor = CommandTraceMonitor()

    def make_env():
        return Monitor(GazeboChaseEnv(**env_kwargs_from_config(config)))

    vec_env = DummyVecEnv([make_env])
    vec_env = VecNormalize.load(vecnormalize_path, vec_env)
    vec_env.training = False
    vec_env.norm_reward = False
    model = PPO.load(model_path, env=vec_env)
    rows = []

    try:
        for episode in range(int(args.episodes)):
            obs = vec_env.reset()
            previous_depth = math.nan
            step = 0
            while not rospy.is_shutdown():
                action, _state = model.predict(obs, deterministic=True)
                action_row = action[0] if len(action.shape) > 1 else action
                env = vec_env.envs[0].env
                _, mapped_body = env._map_action(action_row)
                obs, rewards, dones, infos = vec_env.step(action)
                info = infos[0]
                current_depth = finite(info.get("target_distance", math.nan))
                if math.isfinite(previous_depth) and math.isfinite(current_depth):
                    depth_delta = current_depth - previous_depth
                else:
                    depth_delta = math.nan
                previous_depth = current_depth
                trace = monitor.snapshot()
                rows.append(
                    {
                        "episode": episode,
                        "step": step,
                        "obs_target_visible": bool(info.get("target_visible", False)),
                        "obs_target_depth": current_depth,
                        "obs_target_u": finite(info.get("target_u", math.nan)),
                        "obs_target_v": finite(info.get("target_v", math.nan)),
                        "obs_front_q05": finite(info.get("front_q05_depth", math.nan)),
                        "obs_obstacle_danger": finite(info.get("obstacle_danger", math.nan)),
                        "obs_drone_z": finite(info.get("drone_z", math.nan)),
                        "policy_action_vx": float(action_row[0]),
                        "policy_action_vy": float(action_row[1]),
                        "policy_action_vz": float(action_row[2]),
                        "policy_action_yaw": float(action_row[3]),
                        "mapped_raw_vx_body": float(mapped_body[0]),
                        "mapped_raw_vy_body": float(mapped_body[1]),
                        "mapped_raw_vz_body": float(mapped_body[2]),
                        "mapped_raw_yaw": float(mapped_body[3]),
                        "debug_raw_vx_body": trace["debug_raw_vx_body"],
                        "debug_raw_vy_body": trace["debug_raw_vy_body"],
                        "debug_raw_vz_body": trace["debug_raw_vz_body"],
                        "debug_raw_yaw_body": trace["debug_raw_yaw_body"],
                        "debug_filtered_vx_body": trace["debug_filtered_vx_body"],
                        "debug_filtered_vy_body": trace["debug_filtered_vy_body"],
                        "debug_filtered_vz_body": trace["debug_filtered_vz_body"],
                        "debug_filtered_yaw_body": trace["debug_filtered_yaw_body"],
                        "published_vx_world": trace["published_vx_world"],
                        "published_vy_world": trace["published_vy_world"],
                        "published_vz_world": trace["published_vz_world"],
                        "published_yaw_rate": trace["published_yaw_world"],
                        "safety_mode": str(info.get("safety_mode", "")),
                        "px4_mode": str(info.get("mavros_mode", "")),
                        "armed": bool(info.get("mavros_armed", False)),
                        "target_depth_delta": depth_delta,
                        "reward": float(rewards[0]),
                        "done": bool(dones[0]),
                        "done_reason": str(info.get("terminal_reason", "")),
                    }
                )
                step += 1
                if bool(dones[0]):
                    break
    finally:
        vec_env.close()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=row_fieldnames())
        writer.writeheader()
        writer.writerows(rows)
    print("wrote {}".format(output_path))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("phase7_best_command_trace_eval failed: {}".format(exc), file=sys.stderr)
        sys.exit(1)
