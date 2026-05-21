#!/usr/bin/env python3

import argparse
import csv
import json
import math
import os
import sys

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "envs"))
MODEL_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "models"))
for path in (ENV_DIR, MODEL_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from bc_policy import load_bc_policy
from gazebo_chase_env import GazeboChaseEnv


DEFAULT_CONFIG = "/home/whk/vf_ws/src/drone_chase/config/phase7_bc_world0.yaml"
DEFAULT_MODEL = "/home/whk/vf_ws/outputs/phase7/bc_world0/bc_policy_best.pt"
DEFAULT_OUTPUT_CSV = "/home/whk/vf_ws/outputs/phase7/bc_world0/bc_action_ablation_eval.csv"
DEFAULT_OUTPUT_JSON = "/home/whk/vf_ws/outputs/phase7/bc_world0/bc_action_ablation_summary.json"
MODES = ["pure_bc", "bc_vx_expert_yaw_vz", "expert_vx_bc_yaw_vz", "expert_full"]


def load_yaml(path):
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required") from exc
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
    action_vx = vx / 0.5 if vx >= 0.0 else vx / 0.2
    return np.clip(np.array([action_vx, vy / 0.3, vz / 0.25, yaw_rate / 0.6], dtype=np.float32), -1.0, 1.0)


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
    return normalize_body_cmd(
        clamp(vx, -0.2, 0.5),
        clamp(vy, -0.3, 0.3),
        clamp(vz, -0.25, 0.25),
        clamp(yaw_rate, -0.6, 0.6),
    )


def finite(value, default=math.nan):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def compose_action(mode, bc_action, expert):
    if mode == "pure_bc":
        action = bc_action.copy()
    elif mode == "bc_vx_expert_yaw_vz":
        action = np.array([bc_action[0], 0.0, expert[2], expert[3]], dtype=np.float32)
    elif mode == "expert_vx_bc_yaw_vz":
        action = np.array([expert[0], 0.0, bc_action[2], bc_action[3]], dtype=np.float32)
    elif mode == "expert_full":
        action = expert.copy()
    else:
        raise ValueError("unknown mode={}".format(mode))
    action[1] = 0.0
    return np.clip(action, -1.0, 1.0).astype(np.float32)


def row_fieldnames():
    return [
        "mode", "episode", "step", "target_visible", "target_depth", "target_u", "target_v", "front_q05", "drone_z",
        "action_vx", "action_vy", "action_vz", "action_yaw",
        "bc_action_vx", "bc_action_vy", "bc_action_vz", "bc_action_yaw",
        "expert_action_vx", "expert_action_vy", "expert_action_vz", "expert_action_yaw",
        "reward", "done", "done_reason", "success", "timeout", "safety_mode",
    ]


def summarize_mode(rows, episodes):
    by_episode = {}
    for row in rows:
        by_episode.setdefault(row["episode"], []).append(row)
    episode_summaries = []
    done_reasons = {}
    for episode in range(int(episodes)):
        ep_rows = by_episode.get(episode, [])
        if not ep_rows:
            episode_summaries.append({"success": False, "timeout": False, "final_distance": math.nan, "min_distance": math.nan})
            continue
        distances = [row["target_depth"] for row in ep_rows if math.isfinite(row["target_depth"])]
        reason = str(ep_rows[-1]["done_reason"] or "unknown")
        done_reasons[reason] = done_reasons.get(reason, 0) + 1
        episode_summaries.append(
            {
                "success": any(row["success"] for row in ep_rows),
                "timeout": any(row["timeout"] for row in ep_rows),
                "final_distance": distances[-1] if distances else math.nan,
                "min_distance": min(distances) if distances else math.nan,
            }
        )
    final_distances = [item["final_distance"] for item in episode_summaries if math.isfinite(item["final_distance"])]
    min_distances = [item["min_distance"] for item in episode_summaries if math.isfinite(item["min_distance"])]
    action_vx = np.asarray([row["action_vx"] for row in rows], dtype=np.float32) if rows else np.zeros((0,), dtype=np.float32)
    action_yaw = np.asarray([row["action_yaw"] for row in rows], dtype=np.float32) if rows else np.zeros((0,), dtype=np.float32)
    safety_modes = [str(row["safety_mode"]) for row in rows]
    return {
        "episodes": int(episodes),
        "rows": int(len(rows)),
        "success_rate": float(sum(1 for item in episode_summaries if item["success"])) / float(max(1, episodes)),
        "timeout_rate": float(sum(1 for item in episode_summaries if item["timeout"])) / float(max(1, episodes)),
        "target_visible_ratio": float(sum(1 for row in rows if row["target_visible"])) / float(max(1, len(rows))),
        "final_distance_mean": float(np.mean(final_distances)) if final_distances else math.nan,
        "min_distance_mean": float(np.mean(min_distances)) if min_distances else math.nan,
        "target_depth_lt_1_count": int(sum(1 for row in rows if math.isfinite(row["target_depth"]) and row["target_depth"] < 1.0)),
        "action_vx_mean": float(action_vx.mean()) if len(action_vx) else math.nan,
        "action_vx_std": float(action_vx.std()) if len(action_vx) else math.nan,
        "action_yaw_mean": float(action_yaw.mean()) if len(action_yaw) else math.nan,
        "action_yaw_std": float(action_yaw.std()) if len(action_yaw) else math.nan,
        "target_lost_steps": int(sum(1 for row in rows if not row["target_visible"] or "TARGET_LOST" in str(row["safety_mode"]))),
        "raw_timeout_count": int(sum(1 for mode in safety_modes if "RAW_TIMEOUT" in mode)),
        "emergency_count": int(sum(1 for mode in safety_modes if "EMERGENCY" in mode)),
        "depth_stop_count": int(sum(1 for mode in safety_modes if "DEPTH_STOP" in mode)),
        "done_reasons": done_reasons,
    }


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Run BC action ablation evals.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--seed", type=int, default=271)
    return parser


def main():
    args = build_arg_parser().parse_args()
    import rospy

    config = load_yaml(args.config)
    policy, _checkpoint = load_bc_policy(args.model, device="cpu")
    env = GazeboChaseEnv(**env_kwargs_from_config(config, max_steps=args.max_steps, seed=args.seed))
    rows = []
    try:
        for mode in MODES:
            for episode in range(int(args.episodes)):
                obs, info = env.reset()
                if not bool(info.get("reset_success", False)):
                    raise RuntimeError("{} episode {} reset failed: {}".format(mode, episode, info))
                for step in range(int(args.max_steps)):
                    obs_before = np.asarray(obs, dtype=np.float32).copy()
                    bc_action = np.clip(policy.predict_numpy(obs_before), -1.0, 1.0).astype(np.float32)
                    expert = expert_action(obs_before)
                    action = compose_action(mode, bc_action, expert)
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = bool(terminated or truncated)
                    row = {
                        "mode": mode,
                        "episode": episode,
                        "step": step,
                        "target_visible": bool(info.get("target_visible", False)),
                        "target_depth": finite(info.get("target_distance", obs[4])),
                        "target_u": finite(info.get("target_u", obs[5])),
                        "target_v": finite(info.get("target_v", obs[6])),
                        "front_q05": finite(info.get("front_q05_depth", obs[7])),
                        "drone_z": finite(info.get("drone_z", obs[15])),
                        "action_vx": float(action[0]),
                        "action_vy": float(action[1]),
                        "action_vz": float(action[2]),
                        "action_yaw": float(action[3]),
                        "bc_action_vx": float(bc_action[0]),
                        "bc_action_vy": float(bc_action[1]),
                        "bc_action_vz": float(bc_action[2]),
                        "bc_action_yaw": float(bc_action[3]),
                        "expert_action_vx": float(expert[0]),
                        "expert_action_vy": float(expert[1]),
                        "expert_action_vz": float(expert[2]),
                        "expert_action_yaw": float(expert[3]),
                        "reward": float(reward),
                        "done": done,
                        "done_reason": str(info.get("terminal_reason", "")),
                        "success": bool(info.get("success", False)),
                        "timeout": bool(info.get("timeout", False)),
                        "safety_mode": str(info.get("safety_mode", "")),
                    }
                    rows.append(row)
                    if done or rospy.is_shutdown():
                        break
                print("mode={} episode={} steps={} reason={}".format(mode, episode, step + 1, rows[-1]["done_reason"]))
    finally:
        env.close()

    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    with open(args.output_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=row_fieldnames())
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for mode in MODES:
        summary[mode] = summarize_mode([row for row in rows if row["mode"] == mode], args.episodes)
    with open(args.output_json, "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print("wrote {}".format(args.output_csv))
    print("wrote {}".format(args.output_json))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("phase7_bc_action_ablation_eval failed: {}".format(exc), file=sys.stderr)
        sys.exit(1)
