#!/usr/bin/env python3

import argparse
import csv
import os
import sys

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "envs"))
if ENV_DIR not in sys.path:
    sys.path.insert(0, ENV_DIR)


def load_yaml(path):
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read config files") from exc
    with open(path, "r") as handle:
        return yaml.safe_load(handle) or {}


def env_kwargs_from_config(config):
    env_cfg = dict(config.get("env", {}))
    woods_cfg = config.get("woods", {})
    kwargs = dict(env_cfg)
    kwargs.setdefault("reset_mode", "episode_soft")
    if woods_cfg:
        mapping = {
            "seed": "seed",
            "num_trunks": "woods_easy_num_trunks",
            "num_branches": "woods_easy_num_branches",
            "num_fallen": "woods_easy_num_fallen",
            "area_x_min": "woods_easy_area_x_min",
            "area_x_max": "woods_easy_area_x_max",
            "area_y_min": "woods_easy_area_y_min",
            "area_y_max": "woods_easy_area_y_max",
            "uav_clearance": "woods_easy_uav_clearance",
            "target_clearance": "woods_easy_target_clearance",
        }
        for src, dst in mapping.items():
            if src in woods_cfg and dst not in kwargs:
                kwargs[dst] = woods_cfg[src]
    return {key: value for key, value in kwargs.items() if value is not None}


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Check Phase 7 episode_soft reset stability.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resets", type=int, default=5)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument(
        "--output-csv",
        default="/home/whk/vf_ws/outputs/phase7/reset_stability_check.csv",
    )
    return parser


def row_fieldnames():
    return [
        "reset_index",
        "reset_success",
        "target_respawn_success",
        "start_z",
        "final_z",
        "min_z",
        "max_z",
        "safety_mode",
        "terminated",
        "truncated",
        "terminated_reason",
        "height_violation",
        "raw_timeout_seen",
        "raw_timeout_persisted",
        "nan_detected",
        "steps_completed",
    ]


def main():
    args = build_arg_parser().parse_args()

    from gazebo_chase_env import GazeboChaseEnv

    config = load_yaml(args.config)
    env = GazeboChaseEnv(**env_kwargs_from_config(config))
    rows = []
    failed = False

    try:
        for reset_index in range(args.resets):
            obs, info = env.reset()
            reset_info = dict(info)
            reset_success = bool(info.get("reset_success", False))
            nan_detected = not np.all(np.isfinite(obs))
            z_values = [float(info.get("drone_z", 0.0))]
            modes = [str(info.get("safety_mode", ""))]
            terminated = False
            truncated = False
            terminated_reason = ""
            height_violation = bool(info.get("height_violation", False))
            steps_completed = 0

            if not reset_success:
                failed = True

            if nan_detected:
                failed = True

            if reset_success and not nan_detected:
                for step in range(args.steps):
                    obs, reward, terminated, truncated, info = env.step(np.zeros(4, dtype=np.float32))
                    steps_completed = step + 1
                    if not np.all(np.isfinite(obs)) or not np.isfinite(reward):
                        nan_detected = True
                        failed = True
                    z_values.append(float(info.get("drone_z", 0.0)))
                    modes.append(str(info.get("safety_mode", "")))
                    terminated_reason = str(info.get("terminal_reason", ""))
                    height_violation = height_violation or bool(info.get("height_violation", False))
                    if height_violation:
                        failed = True
                    if terminated or truncated:
                        break

            tail_modes = modes[-10:]
            raw_timeout_seen = any("RAW_TIMEOUT" in mode for mode in modes)
            raw_timeout_persisted = bool(tail_modes) and all("RAW_TIMEOUT" in mode for mode in tail_modes)
            if raw_timeout_persisted:
                failed = True

            row = {
                "reset_index": reset_index,
                "reset_success": reset_success,
                "target_respawn_success": bool(reset_info.get("target_respawn_success", False)),
                "start_z": z_values[0],
                "final_z": z_values[-1],
                "min_z": min(z_values),
                "max_z": max(z_values),
                "safety_mode": modes[-1],
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "terminated_reason": terminated_reason,
                "height_violation": bool(height_violation),
                "raw_timeout_seen": bool(raw_timeout_seen),
                "raw_timeout_persisted": bool(raw_timeout_persisted),
                "nan_detected": bool(nan_detected),
                "steps_completed": steps_completed,
            }
            rows.append(row)
            print("reset_index={} row={}".format(reset_index, row))

        os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
        with open(args.output_csv, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=row_fieldnames())
            writer.writeheader()
            writer.writerows(rows)
        print("wrote {}".format(args.output_csv))
    finally:
        env.close()

    return 1 if failed else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print("phase7_reset_stability_check failed: {}".format(exc), file=sys.stderr)
        sys.exit(1)
