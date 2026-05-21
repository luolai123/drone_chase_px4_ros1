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
if ENV_DIR not in sys.path:
    sys.path.insert(0, ENV_DIR)

from gazebo_chase_env import GazeboChaseEnv


DEFAULT_CONFIG = "/home/whk/vf_ws/src/drone_chase/config/phase7_bc_world0.yaml"
DEFAULT_OUTPUT_DIR = "/home/whk/vf_ws/outputs/phase7/bc_world0_demos"
BAD_DONE_REASONS = {"out_of_bounds", "height_violation"}


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


def finite(value, default=math.nan):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def action_stats(actions, prefix):
    if len(actions) == 0:
        return {
            "{}_mean".format(prefix): [math.nan] * 4,
            "{}_std".format(prefix): [math.nan] * 4,
            "{}_min".format(prefix): [math.nan] * 4,
            "{}_max".format(prefix): [math.nan] * 4,
        }
    arr = np.asarray(actions, dtype=np.float32)
    return {
        "{}_mean".format(prefix): arr.mean(axis=0).tolist(),
        "{}_std".format(prefix): arr.std(axis=0).tolist(),
        "{}_min".format(prefix): arr.min(axis=0).tolist(),
        "{}_max".format(prefix): arr.max(axis=0).tolist(),
    }


def row_fieldnames():
    return (
        ["episode", "step"]
        + ["obs_{}".format(i) for i in range(20)]
        + [
            "action_vx",
            "action_vy",
            "action_vz",
            "action_yaw",
            "target_visible",
            "target_depth",
            "target_u",
            "target_v",
            "front_q05",
            "drone_z",
            "reward",
            "done",
            "done_reason",
            "success",
            "timeout",
            "safety_mode",
            "episode_success",
        ]
    )


def rows_to_arrays(rows):
    if not rows:
        return (
            np.zeros((0, 20), dtype=np.float32),
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.bool_),
        )
    obs = np.asarray([[row["obs_{}".format(i)] for i in range(20)] for row in rows], dtype=np.float32)
    actions = np.asarray(
        [[row["action_vx"], row["action_vy"], row["action_vz"], row["action_yaw"]] for row in rows],
        dtype=np.float32,
    )
    episode_ids = np.asarray([row["episode"] for row in rows], dtype=np.int32)
    success_flags = np.asarray([row["episode_success"] for row in rows], dtype=np.bool_)
    return obs, actions, episode_ids, success_flags


def save_npz(path, rows, metadata):
    obs, actions, episode_ids, success_flags = rows_to_arrays(rows)
    np.savez_compressed(
        path,
        obs=obs,
        actions=actions,
        episode_ids=episode_ids,
        success_flags=success_flags,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )


def valid_row(row):
    safety_mode = str(row.get("safety_mode", ""))
    obs = np.asarray([row["obs_{}".format(i)] for i in range(20)], dtype=np.float32)
    action = np.asarray([row["action_vx"], row["action_vy"], row["action_vz"], row["action_yaw"]], dtype=np.float32)
    return (
        bool(row.get("target_visible", False))
        and str(row.get("done_reason", "")) not in BAD_DONE_REASONS
        and "EMERGENCY" not in safety_mode
        and "DEPTH_STOP" not in safety_mode
        and np.all(np.isfinite(obs))
        and np.all(np.isfinite(action))
    )


def summarize(rows, filtered_rows, episodes, episode_summaries, nan_rows_removed, prefer_success_episodes):
    actions = np.asarray(
        [[row["action_vx"], row["action_vy"], row["action_vz"], row["action_yaw"]] for row in filtered_rows],
        dtype=np.float32,
    ) if filtered_rows else np.zeros((0, 4), dtype=np.float32)
    depths = np.asarray([row["target_depth"] for row in rows if math.isfinite(row["target_depth"])], dtype=np.float32)
    success_episodes = sum(1 for item in episode_summaries if item["success"])
    summary = {
        "total_episodes": int(episodes),
        "success_episodes": int(success_episodes),
        "success_rate": float(success_episodes) / float(max(1, episodes)),
        "total_steps": int(len(rows)),
        "filtered_steps": int(len(filtered_rows)),
        "filtered_ratio": float(len(filtered_rows)) / float(max(1, len(rows))),
        "target_visible_ratio": float(sum(1 for row in rows if row["target_visible"])) / float(max(1, len(rows))),
        "mean_target_depth": float(depths.mean()) if len(depths) else math.nan,
        "min_target_depth": float(depths.min()) if len(depths) else math.nan,
        "nan_inf_rows_removed": int(nan_rows_removed),
        "prefer_success_episodes": bool(prefer_success_episodes),
        "episode_summaries": episode_summaries,
    }
    summary.update(action_stats(actions, "action"))
    return summary


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Collect Phase 7.1F expert demonstrations through GazeboChaseEnv.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--no-prefer-success-episodes", action="store_true")
    return parser


def main():
    args = build_arg_parser().parse_args()
    config = load_yaml(args.config)
    env = GazeboChaseEnv(**env_kwargs_from_config(config, max_steps=args.max_steps, seed=args.seed))
    rows = []
    episode_summaries = []
    try:
        for episode in range(int(args.episodes)):
            obs, info = env.reset()
            if not bool(info.get("reset_success", False)):
                raise RuntimeError("episode {} reset failed: {}".format(episode, info))

            episode_start = len(rows)
            episode_success = False
            terminal_reason = ""
            first_depth = finite(obs[4])
            min_depth = first_depth
            final_depth = first_depth
            total_reward = 0.0

            for step in range(int(args.max_steps)):
                if obs.shape != (20,) or not np.all(np.isfinite(obs)):
                    raise RuntimeError("invalid observation at episode {} step {}".format(episode, step))
                obs_before = obs.astype(np.float32).copy()
                action = expert_action(obs_before)
                obs, reward, terminated, truncated, info = env.step(action)
                done = bool(terminated or truncated)
                terminal_reason = str(info.get("terminal_reason", ""))
                episode_success = bool(info.get("success", False))
                target_depth = finite(obs_before[4])
                min_depth = min(min_depth, target_depth)
                final_depth = target_depth
                total_reward += float(reward)

                row = {
                    "episode": episode,
                    "step": step,
                    "action_vx": float(action[0]),
                    "action_vy": float(action[1]),
                    "action_vz": float(action[2]),
                    "action_yaw": float(action[3]),
                    "target_visible": bool(obs_before[0] > 0.5),
                    "target_depth": target_depth,
                    "target_u": finite(obs_before[5]),
                    "target_v": finite(obs_before[6]),
                    "front_q05": finite(obs_before[7]),
                    "drone_z": finite(obs_before[15]),
                    "reward": float(reward),
                    "done": done,
                    "done_reason": terminal_reason,
                    "success": episode_success,
                    "timeout": bool(info.get("timeout", False)),
                    "safety_mode": str(info.get("safety_mode", "")),
                    "episode_success": False,
                }
                for index in range(20):
                    row["obs_{}".format(index)] = float(obs_before[index])
                rows.append(row)
                if done:
                    break

            for index in range(episode_start, len(rows)):
                rows[index]["episode_success"] = bool(episode_success)
            summary = {
                "episode": episode,
                "success": bool(episode_success),
                "done_reason": terminal_reason,
                "steps": int(len(rows) - episode_start),
                "first_depth": first_depth,
                "min_depth": min_depth,
                "final_depth": final_depth,
                "total_reward": total_reward,
            }
            episode_summaries.append(summary)
            print("episode={} summary={}".format(episode, summary))
    finally:
        env.close()

    prefer_success_episodes = not bool(args.no_prefer_success_episodes)
    valid_rows = [row for row in rows if valid_row(row)]
    finite_rows = [
        row for row in rows
        if np.all(np.isfinite(np.asarray([row["obs_{}".format(i)] for i in range(20)], dtype=np.float32)))
        and np.all(np.isfinite(np.asarray([row["action_vx"], row["action_vy"], row["action_vz"], row["action_yaw"]], dtype=np.float32)))
    ]
    nan_rows_removed = len(rows) - len(finite_rows)
    success_episode_ids = {item["episode"] for item in episode_summaries if item["success"]}
    if prefer_success_episodes and success_episode_ids:
        filtered_rows = [row for row in valid_rows if row["episode"] in success_episode_ids]
    else:
        filtered_rows = valid_rows

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "expert_demos.csv")
    full_npz_path = os.path.join(output_dir, "expert_demos_full.npz")
    filtered_npz_path = os.path.join(output_dir, "expert_demos_filtered.npz")
    primary_npz_path = os.path.join(output_dir, "expert_demos.npz")
    summary_path = os.path.join(output_dir, "demo_summary.json")

    metadata = {
        "config": os.path.abspath(args.config),
        "episodes": int(args.episodes),
        "max_steps": int(args.max_steps),
        "seed": int(args.seed),
        "obs_dim": 20,
        "action_dim": 4,
        "expert": "phase7_expert_env_sanity",
    }
    summary = summarize(rows, filtered_rows, args.episodes, episode_summaries, nan_rows_removed, prefer_success_episodes)
    metadata.update(summary)

    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=row_fieldnames())
        writer.writeheader()
        writer.writerows(rows)
    save_npz(full_npz_path, rows, metadata)
    save_npz(filtered_npz_path, filtered_rows, metadata)
    save_npz(primary_npz_path, filtered_rows if filtered_rows else rows, metadata)
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print("wrote {}".format(csv_path))
    print("wrote {}".format(full_npz_path))
    print("wrote {}".format(filtered_npz_path))
    print("wrote {}".format(summary_path))
    print("success_rate={:.3f} filtered_steps={}/{}".format(summary["success_rate"], len(filtered_rows), len(rows)))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print("phase7_collect_expert_demos failed: {}".format(exc), file=sys.stderr)
        sys.exit(1)
