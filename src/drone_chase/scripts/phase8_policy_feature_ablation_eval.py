#!/usr/bin/env python3

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import Counter
from datetime import datetime

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "envs"))
if ENV_DIR not in sys.path:
    sys.path.insert(0, ENV_DIR)


REGISTRY_DIR = "/home/whk/vf_ws/outputs/final_policy_registry"
DEFAULT_MODEL = os.path.join(REGISTRY_DIR, "best_world0_world1_policy.zip")
DEFAULT_VECNORMALIZE = os.path.join(REGISTRY_DIR, "best_world0_world1_vecnormalize.pkl")
DEFAULT_CONFIG = (
    "/home/whk/vf_ws/outputs/phase7/world0_ppo_from_bc_v2_10k_conservative_run4/config_effective.yaml"
)
DEFAULT_OUTPUT_DIR = "/home/whk/vf_ws/outputs/phase8/policy_feature_ablation"


WORLD_MAP = {
    "world0": "world_0",
    "world1": "world_1",
    "woods_easy": "woods_easy",
    "random_woods": "random_woods",
}

ABLATION_GROUPS = [
    "baseline",
    "no_target_uv",
    "no_target_depth",
    "no_target_position_camera",
    "no_obstacle_risk",
    "no_side_depth",
    "no_front_depth",
    "no_velocity",
    "no_prev_action",
    "noisy_target_uv",
    "noisy_depth",
]


def require_eval_deps():
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
        raise RuntimeError("Missing eval dependencies: {}".format(", ".join(missing)))


def load_yaml(path):
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read config files") from exc
    with open(path, "r") as handle:
        return yaml.safe_load(handle) or {}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite(value, default=math.nan):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(default)
    return value if math.isfinite(value) else float(default)


def mean(values):
    finite_values = []
    for value in values:
        value = finite(value)
        if math.isfinite(value):
            finite_values.append(value)
    return float(sum(finite_values)) / float(len(finite_values)) if finite_values else 0.0


def std(values):
    finite_values = []
    for value in values:
        value = finite(value)
        if math.isfinite(value):
            finite_values.append(value)
    if not finite_values:
        return 0.0
    avg = mean(finite_values)
    return float(math.sqrt(sum((value - avg) ** 2 for value in finite_values) / float(len(finite_values))))


def rate(count, total):
    return float(count) / float(total) if total else 0.0


def parse_tokens(values, default):
    if values is None:
        values = default
    if isinstance(values, str):
        values = [values]
    tokens = []
    for item in values:
        for token in str(item).split(","):
            token = token.strip()
            if token:
                tokens.append(token)
    return tokens


def env_kwargs_from_config(config, seed, world_type):
    env_cfg = dict(config.get("env", {}))
    env_cfg["world_type"] = str(world_type)
    env_cfg["reset_mode"] = "soft"
    env_cfg["respawn_target_on_reset"] = False
    env_cfg["seed"] = int(seed)
    env_cfg.setdefault("reset_ready_timeout", 20.0)
    env_cfg.setdefault("reset_zero_cmd_duration", 1.0)
    return {key: value for key, value in env_cfg.items() if value is not None}


def ablate_obs(obs, group, rng, max_depth):
    ablated = np.asarray(obs, dtype=np.float32).copy()
    max_depth = float(max_depth)

    if group == "baseline":
        pass
    elif group == "no_target_uv":
        ablated[5] = 0.0
        ablated[6] = 0.0
    elif group == "no_target_depth":
        ablated[3] = max_depth
        ablated[4] = max_depth
    elif group == "no_target_position_camera":
        ablated[1:4] = 0.0
    elif group == "no_obstacle_risk":
        ablated[7] = max_depth
        ablated[8] = max_depth
        ablated[9] = max_depth
        ablated[10] = 0.0
        ablated[11] = 0.0
    elif group == "no_side_depth":
        ablated[8] = max_depth
        ablated[9] = max_depth
    elif group == "no_front_depth":
        ablated[7] = max_depth
        ablated[10] = 0.0
        ablated[11] = 0.0
    elif group == "no_velocity":
        ablated[12:15] = 0.0
    elif group == "no_prev_action":
        ablated[16:20] = 0.0
    elif group == "noisy_target_uv":
        ablated[5] = float(np.clip(ablated[5] + rng.normal(0.0, 0.1), -1.5, 1.5))
        ablated[6] = float(np.clip(ablated[6] + rng.normal(0.0, 0.1), -1.5, 1.5))
    elif group == "noisy_depth":
        ablated[4] = float(np.clip(ablated[4] * rng.uniform(0.8, 1.2), 0.0, max_depth))
        for idx in (7, 8, 9):
            ablated[idx] = float(np.clip(ablated[idx] * rng.uniform(0.8, 1.2), 0.0, max_depth))
    else:
        raise RuntimeError("Unsupported ablation group '{}'".format(group))

    return np.nan_to_num(ablated, nan=0.0, posinf=max_depth, neginf=-max_depth).astype(np.float32)


def rollout_fieldnames():
    return [
        "world",
        "ablation_group",
        "episode",
        "seed",
        "step",
        "target_visible",
        "target_depth",
        "target_u",
        "target_v",
        "front_q05_depth",
        "obstacle_area_ratio",
        "obstacle_danger",
        "safety_mode",
        "action_vx",
        "action_vz",
        "action_yaw",
        "raw_vx_body",
        "filtered_vx_body",
        "reward",
        "done",
        "done_reason",
    ]


def episode_fieldnames():
    return [
        "world",
        "ablation_group",
        "episode",
        "seed",
        "success",
        "timeout",
        "collision_or_too_close",
        "out_of_bounds",
        "height_violation",
        "done_reason",
        "final_distance",
        "min_distance",
        "episode_length",
        "target_visible_ratio",
        "emergency_count",
        "depth_stop_count",
        "raw_timeout_count",
        "offboard_drop",
    ]


def run_group(raw_env, vec_env, model, world_name, world_type, group, episodes, seed_base, deterministic, max_depth):
    rollout_rows = []
    episode_rows = []

    for episode in range(int(episodes)):
        seed = int(seed_base) + int(episode)
        group_noise_offset = (ABLATION_GROUPS.index(group) + 1) * 100000
        rng = np.random.default_rng(seed + group_noise_offset)
        raw_env.config["seed"] = int(seed)
        raw_env.config["world_type"] = str(world_type)
        raw_env.world_type = str(world_type)
        obs, _info = raw_env.reset(seed=seed, options={"reset_mode": "soft"})

        visible_steps = 0
        raw_timeout_count = 0
        offboard_drop = 0
        min_distance = float("inf")
        final_distance = float("inf")
        total_steps = 0
        done_reason = ""
        last_info = {}

        while True:
            policy_obs = ablate_obs(obs, group, rng, max_depth)
            norm_obs = vec_env.normalize_obs(np.asarray([policy_obs], dtype=np.float32))
            action, _ = model.predict(norm_obs, deterministic=bool(deterministic))
            action_arr = np.asarray(action)
            action_row = action_arr[0] if action_arr.ndim > 1 else action_arr
            action_row, body_cmd = raw_env._map_action(action_row)

            obs, reward, terminated, truncated, info = raw_env.step(action_row)
            reward = finite(reward, 0.0)
            done = bool(terminated or truncated)
            last_info = info
            if done:
                done_reason = str(info.get("terminal_reason", "") or "")

            target_visible = bool(info.get("target_visible", False))
            visible_steps += int(target_visible)
            distance = finite(info.get("target_distance", math.nan))
            if math.isfinite(distance):
                final_distance = distance
                min_distance = min(min_distance, distance)

            safety_mode = str(info.get("safety_mode", "") or "")
            raw_timeout_count += int("RAW_TIMEOUT" in safety_mode)
            mavros_mode = str(info.get("mavros_mode", "") or "")
            mavros_armed = bool(info.get("mavros_armed", False))
            offboard_drop += int(mavros_armed and mavros_mode != "OFFBOARD")

            snap = raw_env._snapshot()
            filtered_vx_body, _, _ = raw_env._filtered_body_cmd(snap.get("pose"), snap.get("filtered_cmd"))

            rollout_rows.append(
                {
                    "world": world_name,
                    "ablation_group": group,
                    "episode": int(episode),
                    "seed": int(seed),
                    "step": int(total_steps),
                    "target_visible": int(target_visible),
                    "target_depth": finite(info.get("target_distance", math.nan)),
                    "target_u": finite(info.get("target_u", math.nan)),
                    "target_v": finite(info.get("target_v", math.nan)),
                    "front_q05_depth": finite(info.get("front_q05_depth", math.nan)),
                    "obstacle_area_ratio": finite(info.get("obstacle_area_ratio", math.nan)),
                    "obstacle_danger": int(bool(info.get("obstacle_danger", False))),
                    "safety_mode": safety_mode,
                    "action_vx": finite(action_row[0], 0.0),
                    "action_vz": finite(action_row[2], 0.0),
                    "action_yaw": finite(action_row[3], 0.0),
                    "raw_vx_body": finite(body_cmd[0], 0.0),
                    "filtered_vx_body": finite(filtered_vx_body, 0.0),
                    "reward": reward,
                    "done": int(done),
                    "done_reason": done_reason,
                }
            )

            total_steps += 1
            if done:
                break

        episode_rows.append(
            {
                "world": world_name,
                "ablation_group": group,
                "episode": int(episode),
                "seed": int(seed),
                "success": int(bool(last_info.get("success", False))),
                "timeout": int(bool(last_info.get("timeout", False))),
                "collision_or_too_close": int(done_reason == "collision_or_too_close"),
                "out_of_bounds": int(bool(last_info.get("out_of_bounds", False))),
                "height_violation": int(bool(last_info.get("height_violation", False))),
                "done_reason": done_reason,
                "final_distance": finite(final_distance),
                "min_distance": finite(min_distance),
                "episode_length": int(total_steps),
                "target_visible_ratio": float(visible_steps) / float(max(1, total_steps)),
                "emergency_count": int(last_info.get("emergency_count", 0)),
                "depth_stop_count": int(last_info.get("depth_stop_count", 0)),
                "raw_timeout_count": int(raw_timeout_count),
                "offboard_drop": int(offboard_drop),
            }
        )

    return episode_rows, rollout_rows


def summarize(episode_rows, rollout_rows):
    total = len(episode_rows)
    if total <= 0:
        return {}
    terminal_reasons = Counter(str(row.get("done_reason", "")) for row in episode_rows)
    return {
        "episodes": int(total),
        "success_rate": rate(sum(int(row.get("success", 0)) for row in episode_rows), total),
        "timeout_rate": rate(sum(int(row.get("timeout", 0)) for row in episode_rows), total),
        "collision_or_too_close_rate": rate(
            sum(int(row.get("collision_or_too_close", 0)) for row in episode_rows),
            total,
        ),
        "out_of_bounds_rate": rate(sum(int(row.get("out_of_bounds", 0)) for row in episode_rows), total),
        "height_violation_rate": rate(sum(int(row.get("height_violation", 0)) for row in episode_rows), total),
        "final_distance_mean": mean(row.get("final_distance", math.nan) for row in episode_rows),
        "min_distance_mean": mean(row.get("min_distance", math.nan) for row in episode_rows),
        "target_visible_ratio_mean": mean(row.get("target_visible_ratio", 0.0) for row in episode_rows),
        "mean_episode_length": mean(row.get("episode_length", 0) for row in episode_rows),
        "emergency_count_total": int(sum(int(row.get("emergency_count", 0)) for row in episode_rows)),
        "depth_stop_count_total": int(sum(int(row.get("depth_stop_count", 0)) for row in episode_rows)),
        "raw_timeout_count_total": int(sum(int(row.get("raw_timeout_count", 0)) for row in episode_rows)),
        "offboard_drop_total": int(sum(int(row.get("offboard_drop", 0)) for row in episode_rows)),
        "action_vx_mean": mean(row.get("action_vx", 0.0) for row in rollout_rows),
        "action_vx_std": std(row.get("action_vx", 0.0) for row in rollout_rows),
        "yaw_abs_mean": mean(abs(finite(row.get("action_yaw", 0.0), 0.0)) for row in rollout_rows),
        "terminal_reason_counts": dict(terminal_reasons),
    }


def add_performance_drop(summary):
    per_world = summary.get("per_world", {})
    for world, group_rows in per_world.items():
        baseline_success = float((group_rows.get("baseline") or {}).get("success_rate", 0.0))
        for group_summary in group_rows.values():
            group_summary["performance_drop_vs_baseline"] = float(
                baseline_success - float(group_summary.get("success_rate", 0.0))
            )


def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_ablation_report(path, summary):
    per_world = summary.get("per_world", {})
    lines = [
        "# Phase 8.2 Ablation Report",
        "",
        "Generated at: {}".format(summary.get("generated_at", "")),
        "",
        "This is diagnostic only: no training, no fine-tune, no policy/reward/action/safety changes.",
        "",
        "## Baseline Success",
        "",
        "| world | success_rate |",
        "|---|---:|",
    ]
    for world in summary.get("worlds", []):
        baseline = (per_world.get(world, {}) or {}).get("baseline", {}) or {}
        lines.append("| {} | {:.3f} |".format(world, float(baseline.get("success_rate", 0.0))))

    lines.extend(["", "## Performance Drop vs Baseline", "", "| group | mean_drop | max_drop |", "|---|---:|---:|"])
    groups = summary.get("ablation_groups", [])
    for group in groups:
        if group == "baseline":
            continue
        drops = []
        for world in summary.get("worlds", []):
            drops.append(float(((per_world.get(world, {}) or {}).get(group, {}) or {}).get("performance_drop_vs_baseline", 0.0)))
        lines.append("| {} | {:.3f} | {:.3f} |".format(group, mean(drops), max(drops) if drops else 0.0))

    lines.extend(["", "## Raw Summary", "", "```json", json.dumps(summary, indent=2, sort_keys=True), "```"])
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Phase 8.2 frozen-policy observation ablation eval.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--vecnormalize", default=DEFAULT_VECNORMALIZE)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--episodes-per-world", type=int, default=5)
    parser.add_argument("--seed-base", type=int, default=82000)
    parser.add_argument("--max-depth", type=float, default=10.0)
    parser.add_argument("--noise-seed", type=int, default=8282)
    parser.add_argument("--deterministic", dest="deterministic", action="store_true", default=True)
    parser.add_argument("--stochastic", dest="deterministic", action="store_false")
    parser.add_argument("--worlds", nargs="+", default=list(WORLD_MAP.keys()))
    parser.add_argument("--ablation-groups", nargs="+", default=list(ABLATION_GROUPS))
    parser.add_argument(
        "--allow-inprocess-world-switch",
        action="store_true",
        help="unsafe diagnostic mode; final policy regression should relaunch Gazebo/PX4 per world",
    )
    return parser


def main():
    args = build_arg_parser().parse_args()
    require_eval_deps()

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from gazebo_chase_env import GazeboChaseEnv

    worlds = parse_tokens(args.worlds, list(WORLD_MAP.keys()))
    groups = parse_tokens(args.ablation_groups, list(ABLATION_GROUPS))
    for world in worlds:
        if world not in WORLD_MAP:
            raise RuntimeError("Unsupported world '{}'; expected {}".format(world, sorted(WORLD_MAP)))
    for group in groups:
        if group not in ABLATION_GROUPS:
            raise RuntimeError("Unsupported ablation group '{}'; expected {}".format(group, ABLATION_GROUPS))
    if len(worlds) > 1 and not bool(args.allow_inprocess_world_switch):
        raise RuntimeError(
            "Multiple worlds require separate Gazebo/PX4 launches. "
            "Run this evaluator once per launched world and aggregate outputs, "
            "or pass --allow-inprocess-world-switch only for unsafe diagnostics."
        )

    os.makedirs(args.output_dir, exist_ok=True)
    config = load_yaml(args.config)
    first_world = worlds[0]
    raw_env = GazeboChaseEnv(**env_kwargs_from_config(config, args.seed_base, WORLD_MAP[first_world]))
    vec_env = DummyVecEnv([lambda: raw_env])
    vec_env = VecNormalize.load(args.vecnormalize, vec_env)
    vec_env.training = False
    vec_env.norm_reward = False
    model = PPO.load(args.model)

    all_episode_rows = []
    all_rollout_rows = []
    per_world = {}

    try:
        for world_index, world in enumerate(worlds):
            world_type = WORLD_MAP[world]
            per_world[world] = {}
            for group in groups:
                seed_base = int(args.seed_base) + world_index * 10000
                episode_rows, rollout_rows = run_group(
                    raw_env,
                    vec_env,
                    model,
                    world,
                    world_type,
                    group,
                    args.episodes_per_world,
                    seed_base,
                    args.deterministic,
                    args.max_depth,
                )
                all_episode_rows.extend(episode_rows)
                all_rollout_rows.extend(rollout_rows)
                per_world[world][group] = summarize(episode_rows, rollout_rows)
                print(
                    "world={} group={} episodes={} success_rate={:.3f} collision_rate={:.3f} raw_timeout={} offboard={}".format(
                        world,
                        group,
                        len(episode_rows),
                        per_world[world][group].get("success_rate", 0.0),
                        per_world[world][group].get("collision_or_too_close_rate", 0.0),
                        per_world[world][group].get("raw_timeout_count_total", 0),
                        per_world[world][group].get("offboard_drop_total", 0),
                    )
                )
    finally:
        vec_env.close()

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model_path": args.model,
        "vecnormalize_path": args.vecnormalize,
        "model_sha256": sha256(args.model) if os.path.exists(args.model) else "",
        "vecnormalize_sha256": sha256(args.vecnormalize) if os.path.exists(args.vecnormalize) else "",
        "config_path": args.config,
        "episodes_per_world": int(args.episodes_per_world),
        "worlds": worlds,
        "ablation_groups": groups,
        "deterministic": bool(args.deterministic),
        "per_world": per_world,
    }
    add_performance_drop(summary)

    rollout_csv = os.path.join(args.output_dir, "policy_feature_ablation_rollouts.csv")
    episode_csv = os.path.join(args.output_dir, "policy_feature_ablation_episodes.csv")
    summary_json = os.path.join(args.output_dir, "policy_feature_ablation_summary.json")
    report_md = os.path.join(args.output_dir, "phase8_2_ablation_report.md")

    write_csv(rollout_csv, rollout_fieldnames(), all_rollout_rows)
    write_csv(episode_csv, episode_fieldnames(), all_episode_rows)
    with open(summary_json, "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    write_ablation_report(report_md, summary)


if __name__ == "__main__":
    main()
