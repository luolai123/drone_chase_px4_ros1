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
DEFAULT_OUTPUT_CSV = "/home/whk/vf_ws/outputs/phase7/bc_world0/bc_online_error_audit.csv"
DEFAULT_OUTPUT_MD = "/home/whk/vf_ws/outputs/phase7/bc_world0/bc_online_error_audit.md"


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


def depth_bucket(depth, visible=True):
    if not visible:
        return "lost"
    if depth < 0.8:
        return "capture"
    if depth < 1.5:
        return "near"
    if depth <= 3.0:
        return "mid"
    return "far"


def error_stats(rows):
    if not rows:
        return {"count": 0}
    result = {"count": len(rows)}
    for name in ("vx", "vy", "vz", "yaw"):
        values = np.asarray([float(row["abs_err_{}".format(name)]) for row in rows], dtype=np.float32)
        result["abs_err_{}_mean".format(name)] = float(values.mean())
        result["abs_err_{}_max".format(name)] = float(values.max())
    result["bc_vx_mean"] = float(np.mean([float(row["bc_action_vx"]) for row in rows]))
    result["expert_vx_mean"] = float(np.mean([float(row["expert_action_vx"]) for row in rows]))
    result["bc_yaw_mean"] = float(np.mean([float(row["bc_action_yaw"]) for row in rows]))
    result["expert_yaw_mean"] = float(np.mean([float(row["expert_action_yaw"]) for row in rows]))
    return result


def group_stats(rows, key_fn):
    groups = {}
    for row in rows:
        groups.setdefault(key_fn(row), []).append(row)
    return {key: error_stats(value) for key, value in sorted(groups.items())}


def largest_error_dim(stats):
    candidates = {name: stats.get("abs_err_{}_mean".format(name), 0.0) for name in ("vx", "vy", "vz", "yaw")}
    return max(candidates, key=candidates.get), candidates


def write_report(path, rows, summary):
    all_stats = error_stats(rows)
    largest_dim, dim_errors = largest_error_dim(all_stats)
    depth_stats = summary["by_depth_bucket"]
    near_capture_rows = [row for row in rows if row["depth_bucket"] in ("near", "capture")]
    near_stats = error_stats(near_capture_rows)
    near_dim, _ = largest_error_dim(near_stats)
    lost_rows = [row for row in rows if not row["target_visible"]]
    lost_recovery_issue = bool(lost_rows) and error_stats(lost_rows).get("abs_err_yaw_mean", 0.0) > 0.2
    near_vx_issue = near_stats.get("bc_vx_mean", 0.0) > near_stats.get("expert_vx_mean", 0.0) + 0.1
    yaw_issue = dim_errors.get("yaw", 0.0) >= max(dim_errors.get("vx", 0.0), dim_errors.get("vz", 0.0))
    vz_issue = dim_errors.get("vz", 0.0) >= max(dim_errors.get("vx", 0.0), dim_errors.get("yaw", 0.0))
    worst_bucket = ""
    worst_value = -1.0
    for bucket, stats in depth_stats.items():
        value = stats.get("abs_err_vx_mean", 0.0) + stats.get("abs_err_vz_mean", 0.0) + stats.get("abs_err_yaw_mean", 0.0)
        if value > worst_value:
            worst_bucket = bucket
            worst_value = value

    lines = [
        "# Phase 7.1F2 BC Online Error Audit",
        "",
        "rows: {}".format(len(rows)),
        "success_rate: {:.6g}".format(summary["success_rate"]),
        "timeout_rate: {:.6g}".format(summary["timeout_rate"]),
        "target_visible_ratio: {:.6g}".format(summary["target_visible_ratio"]),
        "final_distance_mean: {:.6g}".format(summary["final_distance_mean"]),
        "min_distance_mean: {:.6g}".format(summary["min_distance_mean"]),
        "",
        "## Answers",
        "",
        "1. BC 在线失败主要发生在哪个距离段：{}".format(worst_bucket),
        "2. near-capture 段是否 action_vx 过大导致冲过/丢目标：{}".format("yes" if near_vx_issue else "no"),
        "3. yaw 是否偏小导致目标偏离：{}".format("yes" if yaw_issue else "no"),
        "4. vz 是否错误导致目标从垂直方向丢失：{}".format("yes" if vz_issue else "no"),
        "5. target lost 后 BC 是否不会恢复：{}".format("yes" if lost_recovery_issue else "inconclusive/no"),
        "6. BC 与 expert 的最大误差集中在哪个动作维度：{}".format(largest_dim),
        "",
        "## Overall Error Means",
        "",
        json.dumps(all_stats, indent=2, sort_keys=True),
        "",
        "## Depth Buckets",
        "",
        json.dumps(depth_stats, indent=2, sort_keys=True),
        "",
        "## U Buckets",
        "",
        json.dumps(summary["by_abs_u_bucket"], indent=2, sort_keys=True),
        "",
        "## V Buckets",
        "",
        json.dumps(summary["by_abs_v_bucket"], indent=2, sort_keys=True),
        "",
    ]
    with open(path, "w") as handle:
        handle.write("\n".join(lines))


def row_fieldnames():
    return [
        "episode", "step", "target_visible", "target_depth", "target_u", "target_v", "front_q05", "drone_z", "safety_mode",
        "depth_bucket", "u_bucket", "v_bucket",
        "bc_action_vx", "bc_action_vy", "bc_action_vz", "bc_action_yaw",
        "expert_action_vx", "expert_action_vy", "expert_action_vz", "expert_action_yaw",
        "err_vx", "err_vy", "err_vz", "err_yaw", "abs_err_vx", "abs_err_vy", "abs_err_vz", "abs_err_yaw",
        "done", "done_reason", "success", "timeout",
    ]


def bucket_abs(value):
    value = abs(float(value))
    if value < 0.1:
        return "centered"
    if value <= 0.3:
        return "mild"
    return "large"


def summarize(rows, episodes):
    by_episode = {}
    for row in rows:
        by_episode.setdefault(row["episode"], []).append(row)
    final_distances = []
    min_distances = []
    success_count = 0
    timeout_count = 0
    for episode in range(int(episodes)):
        ep_rows = by_episode.get(episode, [])
        if not ep_rows:
            continue
        distances = [row["target_depth"] for row in ep_rows if math.isfinite(row["target_depth"])]
        if distances:
            final_distances.append(distances[-1])
            min_distances.append(min(distances))
        success_count += int(any(row["success"] for row in ep_rows))
        timeout_count += int(any(row["timeout"] for row in ep_rows))
    return {
        "episodes": int(episodes),
        "rows": int(len(rows)),
        "success_rate": float(success_count) / float(max(1, episodes)),
        "timeout_rate": float(timeout_count) / float(max(1, episodes)),
        "target_visible_ratio": float(sum(1 for row in rows if row["target_visible"])) / float(max(1, len(rows))),
        "final_distance_mean": float(np.mean(final_distances)) if final_distances else math.nan,
        "min_distance_mean": float(np.mean(min_distances)) if min_distances else math.nan,
        "by_visible": group_stats(rows, lambda row: "visible" if row["target_visible"] else "lost"),
        "by_depth_bucket": group_stats(rows, lambda row: row["depth_bucket"]),
        "by_abs_u_bucket": group_stats(rows, lambda row: row["u_bucket"]),
        "by_abs_v_bucket": group_stats(rows, lambda row: row["v_bucket"]),
    }


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Audit online BC actions against expert labels.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-md", default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--seed", type=int, default=171)
    return parser


def main():
    args = build_arg_parser().parse_args()
    import rospy

    config = load_yaml(args.config)
    policy, _checkpoint = load_bc_policy(args.model, device="cpu")
    env = GazeboChaseEnv(**env_kwargs_from_config(config, max_steps=args.max_steps, seed=args.seed))
    rows = []
    try:
        for episode in range(int(args.episodes)):
            obs, info = env.reset()
            if not bool(info.get("reset_success", False)):
                raise RuntimeError("episode {} reset failed: {}".format(episode, info))
            for step in range(int(args.max_steps)):
                obs_before = np.asarray(obs, dtype=np.float32).copy()
                bc_action = np.clip(policy.predict_numpy(obs_before), -1.0, 1.0).astype(np.float32)
                expert = expert_action(obs_before)
                obs, reward, terminated, truncated, info = env.step(bc_action)
                done = bool(terminated or truncated)
                visible = bool(info.get("target_visible", False))
                target_depth = finite(info.get("target_distance", obs[4]))
                target_u = finite(info.get("target_u", obs[5]))
                target_v = finite(info.get("target_v", obs[6]))
                err = bc_action - expert
                row = {
                    "episode": episode,
                    "step": step,
                    "target_visible": visible,
                    "target_depth": target_depth,
                    "target_u": target_u,
                    "target_v": target_v,
                    "front_q05": finite(info.get("front_q05_depth", obs[7])),
                    "drone_z": finite(info.get("drone_z", obs[15])),
                    "safety_mode": str(info.get("safety_mode", "")),
                    "depth_bucket": depth_bucket(target_depth, visible),
                    "u_bucket": bucket_abs(target_u),
                    "v_bucket": bucket_abs(target_v),
                    "bc_action_vx": float(bc_action[0]),
                    "bc_action_vy": float(bc_action[1]),
                    "bc_action_vz": float(bc_action[2]),
                    "bc_action_yaw": float(bc_action[3]),
                    "expert_action_vx": float(expert[0]),
                    "expert_action_vy": float(expert[1]),
                    "expert_action_vz": float(expert[2]),
                    "expert_action_yaw": float(expert[3]),
                    "err_vx": float(err[0]),
                    "err_vy": float(err[1]),
                    "err_vz": float(err[2]),
                    "err_yaw": float(err[3]),
                    "abs_err_vx": float(abs(err[0])),
                    "abs_err_vy": float(abs(err[1])),
                    "abs_err_vz": float(abs(err[2])),
                    "abs_err_yaw": float(abs(err[3])),
                    "done": done,
                    "done_reason": str(info.get("terminal_reason", "")),
                    "success": bool(info.get("success", False)),
                    "timeout": bool(info.get("timeout", False)),
                }
                rows.append(row)
                if done or rospy.is_shutdown():
                    break
            print("episode={} steps={} reason={}".format(episode, sum(1 for row in rows if row["episode"] == episode), rows[-1]["done_reason"]))
    finally:
        env.close()

    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    with open(args.output_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=row_fieldnames())
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows, args.episodes)
    write_report(args.output_md, rows, summary)
    print("wrote {}".format(args.output_csv))
    print("wrote {}".format(args.output_md))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("phase7_bc_online_error_audit failed: {}".format(exc), file=sys.stderr)
        sys.exit(1)
