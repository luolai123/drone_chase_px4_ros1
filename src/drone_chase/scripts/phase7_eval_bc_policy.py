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
DEFAULT_OUTPUT_DIR = "/home/whk/vf_ws/outputs/phase7/bc_world0"
DEFAULT_DEMO_SUMMARY = "/home/whk/vf_ws/outputs/phase7/bc_world0_demos/demo_summary.json"


def load_yaml(path):
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required") from exc
    with open(path, "r") as handle:
        return yaml.safe_load(handle) or {}


def load_json(path):
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r") as handle:
        return json.load(handle)


def maybe_apply_obs_norm(policy, obs_norm_path):
    if not obs_norm_path:
        return
    if not os.path.exists(obs_norm_path):
        raise FileNotFoundError(obs_norm_path)
    import torch

    data = np.load(obs_norm_path)
    obs_mean = torch.as_tensor(np.asarray(data["obs_mean"], dtype=np.float32))
    obs_std = torch.as_tensor(np.asarray(data["obs_std"], dtype=np.float32))
    if obs_mean.shape[0] != policy.obs_dim or obs_std.shape[0] != policy.obs_dim:
        raise ValueError("obs norm shape mismatch: mean={} std={}".format(obs_mean.shape, obs_std.shape))
    policy.obs_mean.data.copy_(obs_mean.to(policy.obs_mean.device))
    policy.obs_std.data.copy_(torch.clamp(obs_std, min=1.0e-6).to(policy.obs_std.device))


def env_kwargs_from_config(config, max_steps=None, seed=None):
    env_cfg = dict(config.get("env", {}))
    env_cfg.setdefault("world_type", "world_0")
    env_cfg.setdefault("reset_mode", "episode_soft")
    if max_steps is not None:
        env_cfg["max_episode_steps"] = int(max_steps)
    if seed is not None:
        env_cfg["seed"] = int(seed)
    return {key: value for key, value in env_cfg.items() if value is not None}


def finite(value, default=math.nan):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def action_stats(rows, prefix):
    values = np.asarray([[row["bc_action_vx"], row["bc_action_vy"], row["bc_action_vz"], row["bc_action_yaw"]] for row in rows], dtype=np.float32)
    if len(values) == 0:
        return {
            "{}_mean".format(prefix): [math.nan] * 4,
            "{}_std".format(prefix): [math.nan] * 4,
            "{}_min".format(prefix): [math.nan] * 4,
            "{}_max".format(prefix): [math.nan] * 4,
        }
    return {
        "{}_mean".format(prefix): values.mean(axis=0).astype(float).tolist(),
        "{}_std".format(prefix): values.std(axis=0).astype(float).tolist(),
        "{}_min".format(prefix): values.min(axis=0).astype(float).tolist(),
        "{}_max".format(prefix): values.max(axis=0).astype(float).tolist(),
    }


def rollout_fieldnames():
    return [
        "episode",
        "step",
        "target_visible",
        "target_depth",
        "target_u",
        "target_v",
        "front_q05",
        "drone_z",
        "bc_action_vx",
        "bc_action_vy",
        "bc_action_vz",
        "bc_action_yaw",
        "reward",
        "done",
        "done_reason",
        "success",
        "timeout",
        "safety_mode",
    ]


def summarize(rows, episodes):
    by_episode = {}
    for row in rows:
        by_episode.setdefault(row["episode"], []).append(row)
    episode_summaries = []
    for episode in range(int(episodes)):
        ep_rows = by_episode.get(episode, [])
        if not ep_rows:
            episode_summaries.append(
                {
                    "episode": episode,
                    "success": False,
                    "timeout": False,
                    "length": 0,
                    "final_distance": math.nan,
                    "min_distance": math.nan,
                    "mean_reward": math.nan,
                }
            )
            continue
        distances = [row["target_depth"] for row in ep_rows if math.isfinite(row["target_depth"])]
        episode_summaries.append(
            {
                "episode": episode,
                "success": any(bool(row["success"]) for row in ep_rows),
                "timeout": any(bool(row["timeout"]) for row in ep_rows),
                "length": len(ep_rows),
                "final_distance": distances[-1] if distances else math.nan,
                "min_distance": min(distances) if distances else math.nan,
                "mean_reward": float(np.mean([row["reward"] for row in ep_rows])) if ep_rows else math.nan,
            }
        )
    final_distances = [item["final_distance"] for item in episode_summaries if math.isfinite(item["final_distance"])]
    min_distances = [item["min_distance"] for item in episode_summaries if math.isfinite(item["min_distance"])]
    rewards = [row["reward"] for row in rows if math.isfinite(row["reward"])]
    safety_modes = [str(row["safety_mode"]) for row in rows]
    summary = {
        "episodes": int(episodes),
        "rows": int(len(rows)),
        "success_rate": float(sum(1 for item in episode_summaries if item["success"])) / float(max(1, episodes)),
        "timeout_rate": float(sum(1 for item in episode_summaries if item["timeout"])) / float(max(1, episodes)),
        "target_visible_ratio": float(sum(1 for row in rows if row["target_visible"])) / float(max(1, len(rows))),
        "final_distance_mean": float(np.mean(final_distances)) if final_distances else math.nan,
        "min_distance_mean": float(np.mean(min_distances)) if min_distances else math.nan,
        "target_depth_lt_1_count": int(sum(1 for row in rows if math.isfinite(row["target_depth"]) and row["target_depth"] < 1.0)),
        "mean_reward": float(np.mean(rewards)) if rewards else math.nan,
        "raw_timeout_count": int(sum(1 for mode in safety_modes if "RAW_TIMEOUT" in mode)),
        "emergency_count": int(sum(1 for mode in safety_modes if "EMERGENCY" in mode)),
        "depth_stop_count": int(sum(1 for mode in safety_modes if "DEPTH_STOP" in mode)),
        "episode_summaries": episode_summaries,
    }
    summary.update(action_stats(rows, "action"))
    return summary


def fmt(value):
    if isinstance(value, bool):
        return "yes" if value else "no"
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return "{:.6g}".format(value) if math.isfinite(value) else "nan"
    return str(value)


def dim_value(stats, key, index):
    values = stats.get(key, [])
    if isinstance(values, list) and len(values) > index:
        return values[index]
    return math.nan


def write_phase_report(output_dir, demo_summary, bc_summary, eval_summary):
    val_metrics = bc_summary.get("val_metrics_best", {})
    demo_action_mean = demo_summary.get("action_mean", [math.nan] * 4)
    demo_action_std = demo_summary.get("action_std", [math.nan] * 4)
    demo_action_min = demo_summary.get("action_min", [math.nan] * 4)
    demo_action_max = demo_summary.get("action_max", [math.nan] * 4)
    eval_action_mean = eval_summary.get("action_mean", [math.nan] * 4)
    eval_action_std = eval_summary.get("action_std", [math.nan] * 4)
    eval_action_min = eval_summary.get("action_min", [math.nan] * 4)
    eval_action_max = eval_summary.get("action_max", [math.nan] * 4)
    allow_71g = (
        eval_summary.get("target_visible_ratio", 0.0) > 0.85
        and eval_summary.get("min_distance_mean", float("inf")) < 1.5
        and (
            eval_summary.get("success_rate", 0.0) >= 0.2
            or eval_summary.get("target_depth_lt_1_count", 0) > 0
        )
    )
    current_issue = "BC warm-start 达到进入 7.1G 的最低门槛。" if allow_71g else "BC eval 未达到 warm-start 门槛，需扩大 demos 或检查预测动作分布。"
    lines = [
        "# Phase 7.1F BC Report",
        "",
        "1. demo collection 是否完成：{}".format("yes" if demo_summary else "no"),
        "2. demo episodes：{}".format(fmt(demo_summary.get("total_episodes"))),
        "3. demo success rate：{}".format(fmt(demo_summary.get("success_rate"))),
        "4. full dataset steps：{}".format(fmt(demo_summary.get("total_steps"))),
        "5. filtered dataset steps：{}".format(fmt(demo_summary.get("filtered_steps"))),
        "6. filtered ratio：{}".format(fmt(demo_summary.get("filtered_ratio"))),
        "7. target_visible_ratio：{}".format(fmt(demo_summary.get("target_visible_ratio"))),
        "8. expert action_vx mean/std/min/max：{} / {} / {} / {}".format(
            fmt(demo_action_mean[0]), fmt(demo_action_std[0]), fmt(demo_action_min[0]), fmt(demo_action_max[0])
        ),
        "9. expert action_yaw mean/std/min/max：{} / {} / {} / {}".format(
            fmt(demo_action_mean[3]), fmt(demo_action_std[3]), fmt(demo_action_min[3]), fmt(demo_action_max[3])
        ),
        "10. BC training 是否完成：{}".format("yes" if bc_summary else "no"),
        "11. train loss final：{}".format(fmt(bc_summary.get("train_loss_final"))),
        "12. val loss best：{}".format(fmt(bc_summary.get("val_loss_best"))),
        "13. action_vx MAE：{}".format(fmt(val_metrics.get("vx_mae"))),
        "14. action_yaw MAE：{}".format(fmt(val_metrics.get("yaw_mae"))),
        "15. action_vx sign agreement：{}".format(fmt(val_metrics.get("action_vx_sign_agreement"))),
        "16. action_yaw sign agreement：{}".format(fmt(val_metrics.get("action_yaw_sign_agreement"))),
        "17. BC eval 是否完成：{}".format("yes" if eval_summary else "no"),
        "18. BC eval episodes：{}".format(fmt(eval_summary.get("episodes"))),
        "19. BC eval success rate：{}".format(fmt(eval_summary.get("success_rate"))),
        "20. BC eval timeout rate：{}".format(fmt(eval_summary.get("timeout_rate"))),
        "21. BC eval target_visible_ratio：{}".format(fmt(eval_summary.get("target_visible_ratio"))),
        "22. BC eval final_distance_mean：{}".format(fmt(eval_summary.get("final_distance_mean"))),
        "23. BC eval min_distance_mean：{}".format(fmt(eval_summary.get("min_distance_mean"))),
        "24. 是否出现 target_depth < 1.0m：{}".format("yes" if eval_summary.get("target_depth_lt_1_count", 0) > 0 else "no"),
        "25. RAW_TIMEOUT 次数：{}".format(fmt(eval_summary.get("raw_timeout_count"))),
        "26. emergency/depth_stop 次数：{} / {}".format(fmt(eval_summary.get("emergency_count")), fmt(eval_summary.get("depth_stop_count"))),
        "27. BC action_vx mean/std/min/max：{} / {} / {} / {}".format(
            fmt(eval_action_mean[0]), fmt(eval_action_std[0]), fmt(eval_action_min[0]), fmt(eval_action_max[0])
        ),
        "28. 当前主要问题：{}".format(current_issue),
        "29. 是否允许进入 Phase 7.1G：{}".format("yes" if allow_71g else "no"),
        "30. 是否允许进入 Phase 7.2：no",
        "31. 是否允许进入 world1：no",
        "",
    ]
    path = os.path.join(output_dir, "phase7_1f_bc_report.md")
    with open(path, "w") as handle:
        handle.write("\n".join(lines))
    return path


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Evaluate a Phase 7.1F BC policy online in world0.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--demo-summary", default=DEFAULT_DEMO_SUMMARY)
    parser.add_argument("--bc-summary", default=None)
    parser.add_argument("--obs-norm", default=None)
    parser.add_argument("--seed", type=int, default=71)
    return parser


def main():
    args = build_arg_parser().parse_args()
    import rospy

    config = load_yaml(args.config)
    policy, _checkpoint = load_bc_policy(args.model, device="cpu")
    maybe_apply_obs_norm(policy, args.obs_norm)
    env = GazeboChaseEnv(**env_kwargs_from_config(config, max_steps=args.max_steps, seed=args.seed))
    rows = []
    try:
        for episode in range(int(args.episodes)):
            obs, info = env.reset()
            if not bool(info.get("reset_success", False)):
                raise RuntimeError("episode {} reset failed: {}".format(episode, info))
            for step in range(int(args.max_steps)):
                action = np.clip(policy.predict_numpy(obs), -1.0, 1.0).astype(np.float32)
                obs, reward, terminated, truncated, info = env.step(action)
                done = bool(terminated or truncated)
                row = {
                    "episode": episode,
                    "step": step,
                    "target_visible": bool(info.get("target_visible", False)),
                    "target_depth": finite(info.get("target_distance", math.nan)),
                    "target_u": finite(info.get("target_u", math.nan)),
                    "target_v": finite(info.get("target_v", math.nan)),
                    "front_q05": finite(info.get("front_q05_depth", math.nan)),
                    "drone_z": finite(info.get("drone_z", math.nan)),
                    "bc_action_vx": float(action[0]),
                    "bc_action_vy": float(action[1]),
                    "bc_action_vz": float(action[2]),
                    "bc_action_yaw": float(action[3]),
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
            print("episode={} rows={} success={} reason={}".format(
                episode,
                sum(1 for row in rows if row["episode"] == episode),
                rows[-1]["success"] if rows else False,
                rows[-1]["done_reason"] if rows else "",
            ))
    finally:
        env.close()

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    rollout_path = os.path.join(output_dir, "bc_eval_rollouts.csv")
    summary_path = os.path.join(output_dir, "bc_eval_summary.json")
    with open(rollout_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rollout_fieldnames())
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows, args.episodes)
    summary["model"] = os.path.abspath(args.model)
    summary["rollouts"] = rollout_path
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    bc_summary_path = args.bc_summary or os.path.join(output_dir, "bc_summary.json")
    report_path = write_phase_report(
        output_dir,
        load_json(args.demo_summary),
        load_json(bc_summary_path),
        summary,
    )
    print("wrote {}".format(rollout_path))
    print("wrote {}".format(summary_path))
    print("wrote {}".format(report_path))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print("phase7_eval_bc_policy failed: {}".format(exc), file=sys.stderr)
        sys.exit(1)
