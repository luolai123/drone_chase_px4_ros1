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
DEFAULT_BC_MODEL = "/home/whk/vf_ws/outputs/phase7/bc_world0/bc_policy_best.pt"
DEFAULT_OUTPUT_DIR = "/home/whk/vf_ws/outputs/phase7/bc_world0_dagger"


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


def default_stem(beta):
    if abs(beta - 0.5) < 1.0e-9:
        return "dagger_batch1_beta05"
    if abs(beta - 0.25) < 1.0e-9:
        return "dagger_batch2_beta025"
    if abs(beta) < 1.0e-9:
        return "dagger_batch3_beta0"
    return "dagger_beta{}".format(str(beta).replace(".", "p"))


def row_fieldnames():
    return (
        ["episode", "step"]
        + ["obs_{}".format(i) for i in range(20)]
        + [
            "expert_action_vx", "expert_action_vy", "expert_action_vz", "expert_action_yaw",
            "bc_action_vx", "bc_action_vy", "bc_action_vz", "bc_action_yaw",
            "executed_action_vx", "executed_action_vy", "executed_action_vz", "executed_action_yaw",
            "executed_policy", "target_visible", "target_depth", "target_u", "target_v", "front_q05", "drone_z",
            "reward", "done", "done_reason", "success", "timeout", "safety_mode", "episode_success",
        ]
    )


def arrays_from_rows(rows):
    obs = np.asarray([[row["obs_{}".format(i)] for i in range(20)] for row in rows], dtype=np.float32)
    expert = np.asarray([[row["expert_action_vx"], row["expert_action_vy"], row["expert_action_vz"], row["expert_action_yaw"]] for row in rows], dtype=np.float32)
    bc = np.asarray([[row["bc_action_vx"], row["bc_action_vy"], row["bc_action_vz"], row["bc_action_yaw"]] for row in rows], dtype=np.float32)
    executed = np.asarray([[row["executed_action_vx"], row["executed_action_vy"], row["executed_action_vz"], row["executed_action_yaw"]] for row in rows], dtype=np.float32)
    episode_ids = np.asarray([row["episode"] for row in rows], dtype=np.int32)
    done_reasons = np.asarray([str(row["done_reason"]) for row in rows], dtype=object)
    safety_modes = np.asarray([str(row["safety_mode"]) for row in rows], dtype=object)
    success_flags = np.asarray([bool(row["episode_success"]) for row in rows], dtype=np.bool_)
    return obs, expert, bc, executed, episode_ids, done_reasons, safety_modes, success_flags


def save_npz(path, rows, metadata):
    obs, expert, bc, executed, episode_ids, done_reasons, safety_modes, success_flags = arrays_from_rows(rows)
    np.savez_compressed(
        path,
        obs=obs,
        expert_actions=expert,
        bc_actions=bc,
        executed_actions=executed,
        episode_ids=episode_ids,
        done_reasons=done_reasons,
        safety_modes=safety_modes,
        success_flags=success_flags,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )


def summarize(rows, episodes, beta):
    success_episodes = len({row["episode"] for row in rows if row["episode_success"]})
    visible = sum(1 for row in rows if row["target_visible"])
    expert_executed = sum(1 for row in rows if row["executed_policy"] == "expert")
    depths = [row["target_depth"] for row in rows if math.isfinite(row["target_depth"])]
    return {
        "episodes": int(episodes),
        "beta": float(beta),
        "rows": int(len(rows)),
        "success_episodes": int(success_episodes),
        "success_rate": float(success_episodes) / float(max(1, episodes)),
        "target_visible_ratio": float(visible) / float(max(1, len(rows))),
        "expert_execute_ratio": float(expert_executed) / float(max(1, len(rows))),
        "min_target_depth": float(min(depths)) if depths else math.nan,
        "mean_target_depth": float(np.mean(depths)) if depths else math.nan,
    }


def reset_with_recovery(env, episode, retries=2):
    obs, info = env.reset()
    if bool(info.get("reset_success", False)):
        return obs, info
    last_info = info
    for attempt in range(int(retries)):
        print("episode={} reset retry={} previous_info={}".format(episode, attempt + 1, last_info))
        obs, info = env.reset(options={"reset_mode": "soft"})
        last_info = info
        if bool(info.get("reset_success", False)):
            return obs, info
    raise RuntimeError("episode {} reset failed after recovery: {}".format(episode, last_info))


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Collect DAgger-lite BC-visited states with expert labels.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--bc-model", default=DEFAULT_BC_MODEL)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-stem", default=None)
    parser.add_argument("--seed", type=int, default=371)
    return parser


def main():
    args = build_arg_parser().parse_args()
    import rospy

    beta = float(np.clip(args.beta, 0.0, 1.0))
    rng = np.random.default_rng(int(args.seed) + int(round(beta * 1000.0)))
    config = load_yaml(args.config)
    policy, _checkpoint = load_bc_policy(args.bc_model, device="cpu")
    env = GazeboChaseEnv(**env_kwargs_from_config(config, max_steps=args.max_steps, seed=args.seed))
    rows = []
    episode_summaries = []
    try:
        for episode in range(int(args.episodes)):
            obs, info = reset_with_recovery(env, episode)
            episode_start = len(rows)
            episode_success = False
            terminal_reason = ""
            for step in range(int(args.max_steps)):
                obs_before = np.asarray(obs, dtype=np.float32).copy()
                expert = expert_action(obs_before)
                bc_action = np.clip(policy.predict_numpy(obs_before), -1.0, 1.0).astype(np.float32)
                use_expert = bool(rng.random() < beta)
                executed = expert if use_expert else bc_action
                obs, reward, terminated, truncated, info = env.step(executed)
                done = bool(terminated or truncated)
                terminal_reason = str(info.get("terminal_reason", ""))
                episode_success = bool(info.get("success", False))
                row = {
                    "episode": episode,
                    "step": step,
                    "expert_action_vx": float(expert[0]),
                    "expert_action_vy": float(expert[1]),
                    "expert_action_vz": float(expert[2]),
                    "expert_action_yaw": float(expert[3]),
                    "bc_action_vx": float(bc_action[0]),
                    "bc_action_vy": float(bc_action[1]),
                    "bc_action_vz": float(bc_action[2]),
                    "bc_action_yaw": float(bc_action[3]),
                    "executed_action_vx": float(executed[0]),
                    "executed_action_vy": float(executed[1]),
                    "executed_action_vz": float(executed[2]),
                    "executed_action_yaw": float(executed[3]),
                    "executed_policy": "expert" if use_expert else "bc",
                    "target_visible": bool(obs_before[0] > 0.5),
                    "target_depth": finite(obs_before[4]),
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
                if done or rospy.is_shutdown():
                    break
            for index in range(episode_start, len(rows)):
                rows[index]["episode_success"] = bool(episode_success)
            episode_summaries.append(
                {
                    "episode": episode,
                    "success": bool(episode_success),
                    "done_reason": terminal_reason,
                    "steps": len(rows) - episode_start,
                }
            )
            print("episode={} summary={}".format(episode, episode_summaries[-1]))
    finally:
        env.close()

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    stem = args.output_stem or default_stem(beta)
    csv_path = os.path.join(output_dir, "{}.csv".format(stem))
    npz_path = os.path.join(output_dir, "{}.npz".format(stem))
    summary_path = os.path.join(output_dir, "{}_summary.json".format(stem))
    metadata = {
        "config": os.path.abspath(args.config),
        "bc_model": os.path.abspath(args.bc_model),
        "episodes": int(args.episodes),
        "max_steps": int(args.max_steps),
        "beta": beta,
        "seed": int(args.seed),
        "episode_summaries": episode_summaries,
    }
    summary = summarize(rows, args.episodes, beta)
    metadata.update(summary)
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=row_fieldnames())
        writer.writeheader()
        writer.writerows(rows)
    save_npz(npz_path, rows, metadata)
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print("wrote {}".format(csv_path))
    print("wrote {}".format(npz_path))
    print("wrote {}".format(summary_path))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("phase7_collect_dagger_demos failed: {}".format(exc), file=sys.stderr)
        sys.exit(1)
