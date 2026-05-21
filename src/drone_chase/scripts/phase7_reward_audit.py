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
DEFAULT_OUTPUT = "/home/whk/vf_ws/outputs/phase7/reward_audit_expert.csv"


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
        np.array([a_vx, vy / 0.3, vz / 0.25, yaw_rate / 0.6], dtype=np.float32),
        -1.0,
        1.0,
    )


def expert_action(obs):
    if bool(obs[0] > 0.5):
        vx = 0.35 * (float(obs[4]) - 0.8)
        vy = 0.0
        vz = -0.35 * float(obs[6])
        yaw_rate = -0.8 * float(obs[5])
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


def component(components, key):
    value = components.get(key, 0.0)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def row_fieldnames():
    return [
        "episode",
        "step",
        "r_approach",
        "r_distance",
        "r_visibility",
        "r_center",
        "r_obstacle",
        "r_smooth",
        "r_safety_mode",
        "r_lost_extra",
        "r_yaw",
        "r_forward",
        "r_terminal",
        "delta_distance",
        "delta_distance_clipped",
        "approach_valid",
        "total_reward",
        "target_depth",
        "d_prev",
        "d_curr",
        "target_u",
        "target_v",
        "front_q05",
        "action_norm",
        "done_reason",
        "success",
        "timeout",
        "height_violation",
        "safety_mode",
    ]


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Phase 7.1 reward audit using expert-through-env rollout.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=10)
    return parser


def main():
    args = build_arg_parser().parse_args()
    config = load_yaml(args.config)
    env = GazeboChaseEnv(**env_kwargs_from_config(config, max_steps=args.max_steps, seed=args.seed))
    rows = []
    success_count = 0

    try:
        for episode in range(args.episodes):
            obs, info = env.reset()
            if not bool(info.get("reset_success", False)):
                raise RuntimeError("episode {} reset failed: {}".format(episode, info))
            for step in range(args.max_steps):
                d_prev = float(env.prev_distance)
                action = expert_action(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                if obs.shape != (20,) or not np.all(np.isfinite(obs)) or not np.isfinite(reward):
                    raise RuntimeError("invalid env output at episode {} step {}".format(episode, step))
                components = dict(info.get("reward_components", {}))
                done_reason = str(info.get("terminal_reason", ""))
                rows.append(
                    {
                        "episode": episode,
                        "step": step,
                        "r_approach": component(components, "r_approach"),
                        "r_distance": component(components, "r_distance"),
                        "r_visibility": component(components, "r_visibility"),
                        "r_center": component(components, "r_center"),
                        "r_obstacle": component(components, "r_obstacle"),
                        "r_smooth": component(components, "r_smooth"),
                        "r_safety_mode": component(components, "r_safety_mode"),
                        "r_lost_extra": component(components, "r_lost_extra"),
                        "r_yaw": component(components, "r_yaw"),
                        "r_forward": component(components, "r_forward"),
                        "r_terminal": component(components, "r_terminal"),
                        "delta_distance": component(components, "delta_distance"),
                        "delta_distance_clipped": component(components, "delta_distance_clipped"),
                        "approach_valid": bool(component(components, "approach_valid") > 0.5),
                        "total_reward": float(reward),
                        "target_depth": float(info.get("target_distance", math.nan)),
                        "d_prev": d_prev,
                        "d_curr": float(info.get("target_distance", math.nan)),
                        "target_u": float(info.get("target_u", math.nan)),
                        "target_v": float(info.get("target_v", math.nan)),
                        "front_q05": float(info.get("front_q05_depth", math.nan)),
                        "action_norm": float(np.linalg.norm(action)),
                        "done_reason": done_reason,
                        "success": bool(info.get("success", False)),
                        "timeout": bool(info.get("timeout", False)),
                        "height_violation": bool(info.get("height_violation", False)),
                        "safety_mode": str(info.get("safety_mode", "")),
                    }
                )
                if bool(info.get("success", False)):
                    success_count += 1
                if terminated or truncated:
                    break
            print(
                "episode={} steps={} success={} reason={}".format(
                    episode,
                    step + 1,
                    bool(info.get("success", False)),
                    str(info.get("terminal_reason", "")),
                )
            )
    finally:
        env.close()

    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=row_fieldnames())
        writer.writeheader()
        writer.writerows(rows)
    print("wrote {}".format(output_path))
    return 0 if success_count > 0 else 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print("phase7_reward_audit failed: {}".format(exc), file=sys.stderr)
        sys.exit(1)
