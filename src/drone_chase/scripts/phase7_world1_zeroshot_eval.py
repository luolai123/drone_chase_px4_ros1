#!/usr/bin/env python3

import argparse
import csv
import json
import math
import os
import sys
import threading
import time
from datetime import datetime

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "envs"))
if ENV_DIR not in sys.path:
    sys.path.insert(0, ENV_DIR)


DEFAULT_OUTPUT_DIR = "/home/whk/vf_ws/outputs/phase7/world1_zeroshot_from_world0_best"
DEFAULT_CONFIG = "/home/whk/vf_ws/src/drone_chase/config/phase7_ppo_world0_from_bc_v2.yaml"
WOODS_PREFIXES = ("woods", "woods_easy", "woods_hard", "random_woods")


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
        raise RuntimeError("PyYAML is required to read config files") from exc
    with open(path, "r") as handle:
        return yaml.safe_load(handle) or {}


def infer_config_path(model_path):
    model_dir = os.path.dirname(os.path.abspath(model_path))
    run_dir = os.path.dirname(model_dir) if os.path.basename(model_dir) == "checkpoints" else model_dir
    for name in ("config_effective.yaml", "config_used.yaml"):
        candidate = os.path.join(run_dir, name)
        if os.path.exists(candidate):
            return candidate
    return DEFAULT_CONFIG


def env_kwargs_from_config(config, world):
    world = str(world)
    if world != "world_1" or world.startswith(WOODS_PREFIXES):
        raise RuntimeError("Phase 7.2A is world1-only; refusing world={}".format(world))
    env_cfg = dict(config.get("env", {}))
    env_cfg["world_type"] = "world_1"
    env_cfg.setdefault("reset_mode", "episode_soft")
    env_cfg.setdefault("respawn_target_on_reset", True)
    return {key: value for key, value in env_cfg.items() if value is not None}


def finite(value, default=math.nan):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(default)
    return value if math.isfinite(value) else float(default)


def mean(values):
    values = [float(v) for v in values]
    return float(sum(values)) / float(len(values)) if values else 0.0


def stats(values):
    values = [float(v) for v in values if math.isfinite(float(v))]
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    mu = mean(values)
    var = mean((v - mu) ** 2.0 for v in values)
    return {"mean": mu, "std": var ** 0.5, "min": min(values), "max": max(values)}


def rate(count, total):
    return float(count) / float(total) if total else 0.0


def action_to_raw_cmd(action, env_cfg):
    max_vx = float(env_cfg.get("max_vx", 0.5))
    min_vx = float(env_cfg.get("min_vx", -0.2))
    max_vy = float(env_cfg.get("max_vy", 0.3))
    max_vz = float(env_cfg.get("max_vz", 0.25))
    max_yaw_rate = float(env_cfg.get("max_yaw_rate", 0.6))
    ax = float(action[0])
    return {
        "commanded_raw_vx_body": ax * max_vx if ax >= 0.0 else ax * abs(min_vx),
        "commanded_raw_vy_body": float(action[1]) * max_vy,
        "commanded_raw_vz_body": float(action[2]) * max_vz,
        "commanded_raw_yaw": float(action[3]) * max_yaw_rate,
    }


def twist_values(msg):
    if msg is None:
        return None
    return {
        "vx": float(msg.twist.linear.x),
        "vy": float(msg.twist.linear.y),
        "vz": float(msg.twist.linear.z),
        "yaw": float(msg.twist.angular.z),
    }


class CommandMonitor:
    def __init__(self):
        import rospy
        from geometry_msgs.msg import TwistStamped

        self.lock = threading.Lock()
        self.debug_raw = None
        self.debug_filtered = None
        self.published = None
        self.subscribers = [
            rospy.Subscriber("/safety_filter/debug_cmd_raw", TwistStamped, self._raw_cb, queue_size=1),
            rospy.Subscriber("/safety_filter/debug_cmd_filtered", TwistStamped, self._filtered_cb, queue_size=1),
            rospy.Subscriber("/mavros/setpoint_velocity/cmd_vel", TwistStamped, self._published_cb, queue_size=1),
        ]

    def _raw_cb(self, msg):
        with self.lock:
            self.debug_raw = msg

    def _filtered_cb(self, msg):
        with self.lock:
            self.debug_filtered = msg

    def _published_cb(self, msg):
        with self.lock:
            self.published = msg

    def snapshot(self, fallback_raw):
        with self.lock:
            raw = twist_values(self.debug_raw)
            filtered = twist_values(self.debug_filtered)
            published = twist_values(self.published)
        return {
            "raw_vx_body": finite(raw["vx"]) if raw is not None else finite(fallback_raw["commanded_raw_vx_body"]),
            "raw_vz_body": finite(raw["vz"]) if raw is not None else finite(fallback_raw["commanded_raw_vz_body"]),
            "raw_yaw": finite(raw["yaw"]) if raw is not None else finite(fallback_raw["commanded_raw_yaw"]),
            "filtered_vx_body": finite(filtered["vx"]) if filtered is not None else math.nan,
            "filtered_vz_body": finite(filtered["vz"]) if filtered is not None else math.nan,
            "filtered_yaw": finite(filtered["yaw"]) if filtered is not None else math.nan,
            "published_vx_world": finite(published["vx"]) if published is not None else math.nan,
            "published_vy_world": finite(published["vy"]) if published is not None else math.nan,
            "published_vz_world": finite(published["vz"]) if published is not None else math.nan,
            "published_yaw": finite(published["yaw"]) if published is not None else math.nan,
            "raw_vx_body_source": "debug" if raw is not None else "action_mapping",
            "filtered_vx_body_available": bool(filtered is not None),
            "published_vx_world_available": bool(published is not None),
        }

    def close(self):
        for subscriber in self.subscribers:
            try:
                subscriber.unregister()
            except Exception:
                pass


def reset_for_episode(env, seed):
    return env.reset(seed=int(seed), options={"reset_mode": "episode_soft"})


def make_reset_row(episode, seed, attempt, reset_info):
    return {
        "episode": int(episode),
        "seed": int(seed),
        "attempt": int(attempt),
        "reset_success": bool(reset_info.get("reset_success", False)),
        "topics_ready": bool(reset_info.get("topics_ready", False)),
        "training_ready": bool(reset_info.get("training_ready", False)),
        "target_respawn_success": bool(reset_info.get("target_respawn_success", True)),
        "target_respawn_message": str(reset_info.get("target_respawn_message", "")),
        "reset_mode_used": str(reset_info.get("reset_mode_used", reset_info.get("reset_mode", ""))),
        "reset_promoted_to_soft": bool(reset_info.get("reset_promoted_to_soft", False)),
        "mavros_mode": str(reset_info.get("mavros_mode", "")),
        "safety_mode": str(reset_info.get("safety_mode", "")),
        "initial_target_visible": bool(reset_info.get("target_visible", False)),
        "initial_target_distance": finite(reset_info.get("target_distance", math.nan)),
        "initial_drone_z": finite(reset_info.get("drone_z", math.nan)),
        "accepted_for_episode": False,
        "reset_accepted_with_warning": False,
    }


def reset_ready_enough(row):
    safety_mode = str(row["safety_mode"])
    blocked = ("WAIT_FCU", "PRESTREAM", "SET_MODE", "ARMING", "TAKEOFF")
    return bool(
        row["topics_ready"]
        and row["target_respawn_success"]
        and row["reset_mode_used"] == "episode_soft"
        and row["mavros_mode"] == "OFFBOARD"
        and not any(token in safety_mode for token in blocked)
    )


def reset_with_recovery(env, episode, seed, retries):
    last_obs = None
    last_info = None
    attempts = []
    for attempt in range(int(retries) + 1):
        obs, reset_info = reset_for_episode(env, seed)
        row = make_reset_row(episode, seed, attempt, reset_info)
        attempts.append(row)
        last_obs = obs
        last_info = reset_info
        if row["reset_success"]:
            row["accepted_for_episode"] = True
            return obs, reset_info, row, attempts
        time.sleep(0.5)
    final_row = attempts[-1]
    if reset_ready_enough(final_row):
        final_row["accepted_for_episode"] = True
        final_row["reset_accepted_with_warning"] = True
        return last_obs, last_info, final_row, attempts
    return last_obs, last_info, final_row, attempts


def rollout_fieldnames():
    return [
        "episode",
        "seed",
        "step",
        "target_visible",
        "target_depth",
        "target_u",
        "target_v",
        "front_q05_depth",
        "left_q05_depth",
        "right_q05_depth",
        "obstacle_area_ratio",
        "obstacle_danger",
        "safety_mode",
        "action_vx",
        "action_vz",
        "action_yaw",
        "raw_vx_body",
        "filtered_vx_body",
        "published_vx_world",
        "drone_z",
        "reward",
        "done",
        "done_reason",
        "success",
        "timeout",
        "collision",
        "out_of_bounds",
        "height_violation",
        "mavros_connected",
        "mavros_mode",
        "mavros_armed",
        "action_vy",
        "raw_vz_body",
        "raw_yaw",
        "filtered_vz_body",
        "filtered_yaw",
        "published_vy_world",
        "published_vz_world",
        "published_yaw",
        "commanded_raw_vx_body",
        "commanded_raw_vy_body",
        "commanded_raw_vz_body",
        "commanded_raw_yaw",
        "raw_vx_body_source",
        "filtered_vx_body_available",
        "published_vx_world_available",
        "target_visible_ratio_so_far",
        "min_distance_so_far",
        "drone_x",
        "drone_y",
        "success_count",
    ]


def classify_failure_modes(episode_rows):
    failures = [row for row in episode_rows if not row["success"]]
    if not failures:
        return "none"
    counts = {
        "timeout": sum(1 for row in failures if row["timeout"]),
        "collision_or_too_close": sum(1 for row in failures if row["collision"]),
        "out_of_bounds": sum(1 for row in failures if row["out_of_bounds"]),
        "height_violation": sum(1 for row in failures if row["height_violation"]),
        "target_lost_or_low_visibility": sum(1 for row in failures if row["target_visible_ratio"] < 0.8),
        "emergency_avoid_active": sum(1 for row in failures if row["emergency_count"] > 0),
        "depth_stop_active": sum(1 for row in failures if row["depth_stop_count"] > 0),
        "raw_timeout_active": sum(1 for row in failures if row["raw_timeout_count"] > 0),
    }
    active = ["{}={}".format(key, value) for key, value in counts.items() if value > 0]
    return ", ".join(active) if active else "unknown"


def build_summary(args, config_path, config, episode_rows, rollout_rows, reset_rows):
    n = len(episode_rows)
    success_count = sum(1 for row in episode_rows if row["success"])
    timeout_count = sum(1 for row in episode_rows if row["timeout"])
    collision_count = sum(1 for row in episode_rows if row["collision"])
    out_of_bounds_count = sum(1 for row in episode_rows if row["out_of_bounds"])
    height_violation_count = sum(1 for row in episode_rows if row["height_violation"])
    raw_timeout_count = sum(int(row["raw_timeout_count"]) for row in episode_rows)
    emergency_count = sum(int(row["emergency_count"]) for row in episode_rows)
    depth_stop_count = sum(int(row["depth_stop_count"]) for row in episode_rows)
    obstacle_danger_steps = sum(int(row["obstacle_danger_steps"]) for row in episode_rows)
    emergency_failure_count = sum(1 for row in episode_rows if not row["success"] and row["emergency_count"] > 0)
    collision_emergency_failure_count = sum(
        1 for row in episode_rows if not row["success"] and (row["collision"] or row["emergency_count"] > 0)
    )
    offboard_drop_count = sum(1 for row in rollout_rows if str(row["mavros_mode"]) != "OFFBOARD")
    offboard_drop_episodes = len(set(row["episode"] for row in rollout_rows if str(row["mavros_mode"]) != "OFFBOARD"))
    episode_reset_rows = [row for row in reset_rows if row.get("accepted_for_episode", False)]
    reset_failures = [row for row in episode_reset_rows if not row.get("reset_success", False)]
    reset_attempt_failures = [row for row in reset_rows if not row.get("reset_success", False)]
    reset_warning_accepts = [row for row in episode_reset_rows if row.get("reset_accepted_with_warning", False)]
    target_respawn_failures = [row for row in episode_reset_rows if not row.get("target_respawn_success", True)]
    episodes_requiring_reset_recovery = len(
        set(row["episode"] for row in episode_reset_rows if int(row.get("attempt", 0)) > 0)
    )
    consecutive_duplicate_respawns = 0
    previous_msg = None
    for row in episode_reset_rows:
        msg = str(row.get("target_respawn_message", ""))
        if msg and previous_msg and msg == previous_msg:
            consecutive_duplicate_respawns += 1
        previous_msg = msg

    action_vx_stats = stats(row["action_vx"] for row in rollout_rows)
    yaw_abs_mean = mean(abs(float(row["action_yaw"])) for row in rollout_rows) if rollout_rows else 0.0
    total_steps = len(rollout_rows)
    reset_pollution_detected = bool(
        reset_failures or target_respawn_failures or consecutive_duplicate_respawns > 0
    )
    offboard_frequent_drop = bool(offboard_drop_count > max(1, int(0.01 * max(1, total_steps))))
    target_visible_ratio_mean = mean(row["target_visible_ratio"] for row in episode_rows)
    final_distance_mean = mean(row["final_distance"] for row in episode_rows)
    min_distance_mean = mean(row["min_distance"] for row in episode_rows)
    front_q05_min_mean = mean(row["front_q05_min"] for row in episode_rows)
    obstacle_danger_steps_mean = mean(row["obstacle_danger_steps"] for row in episode_rows)
    success_rate = rate(success_count, n)
    ideal_gate_pass = bool(
        n == int(args.episodes)
        and success_rate >= 0.70
        and target_visible_ratio_mean > 0.85
        and final_distance_mean <= 1.2
        and min_distance_mean <= 1.0
        and raw_timeout_count == 0
        and out_of_bounds_count == 0
        and height_violation_count == 0
        and collision_emergency_failure_count <= max(1, int(0.20 * max(1, n)))
        and not reset_pollution_detected
        and not offboard_frequent_drop
    )
    minimum_gate_pass = bool(
        n == int(args.episodes)
        and success_rate >= 0.50
        and min_distance_mean <= 1.5
        and target_visible_ratio_mean > 0.80
        and collision_count <= max(1, int(0.20 * max(1, n)))
        and not reset_pollution_detected
        and not offboard_frequent_drop
    )

    return {
        "phase": "7.2A",
        "report_date": datetime.now().isoformat(timespec="seconds"),
        "evaluated_model": os.path.abspath(args.model),
        "evaluated_vecnormalize": os.path.abspath(args.vecnormalize),
        "config": os.path.abspath(config_path),
        "world": "world_1",
        "obstacles_per_episode": int(args.num_obstacles),
        "reset_mode": env_kwargs_from_config(config, args.world).get("reset_mode", "episode_soft"),
        "respawn_target_on_reset": bool(env_kwargs_from_config(config, args.world).get("respawn_target_on_reset", True)),
        "episodes": int(n),
        "requested_episodes": int(args.episodes),
        "deterministic": bool(args.deterministic),
        "seed_base": int(args.seed_base),
        "success_count": int(success_count),
        "success_rate": success_rate,
        "timeout_count": int(timeout_count),
        "timeout_rate": rate(timeout_count, n),
        "collision_count": int(collision_count),
        "collision_rate": rate(collision_count, n),
        "emergency_failure_count": int(emergency_failure_count),
        "collision_emergency_failure_count": int(collision_emergency_failure_count),
        "collision_emergency_failure_rate": rate(collision_emergency_failure_count, n),
        "out_of_bounds_count": int(out_of_bounds_count),
        "out_of_bounds_rate": rate(out_of_bounds_count, n),
        "height_violation_count": int(height_violation_count),
        "height_violation_rate": rate(height_violation_count, n),
        "target_visible_ratio_mean": target_visible_ratio_mean,
        "final_distance_mean": final_distance_mean,
        "min_distance_mean": min_distance_mean,
        "mean_episode_length": mean(row["episode_length"] for row in episode_rows),
        "raw_timeout_count": int(raw_timeout_count),
        "emergency_count": int(emergency_count),
        "depth_stop_count": int(depth_stop_count),
        "front_q05_min_mean": front_q05_min_mean,
        "obstacle_danger_steps": int(obstacle_danger_steps),
        "obstacle_danger_steps_mean": obstacle_danger_steps_mean,
        "action_vx_mean": action_vx_stats["mean"],
        "action_vx_std": action_vx_stats["std"],
        "action_vx_min": action_vx_stats["min"],
        "action_vx_max": action_vx_stats["max"],
        "yaw_abs_mean": yaw_abs_mean,
        "failure_modes": classify_failure_modes(episode_rows),
        "total_steps": int(total_steps),
        "offboard_drop_count": int(offboard_drop_count),
        "offboard_drop_episodes": int(offboard_drop_episodes),
        "offboard_frequent_drop": bool(offboard_frequent_drop),
        "reset_failures": int(len(reset_failures)),
        "reset_attempt_failures": int(len(reset_attempt_failures)),
        "reset_warning_accepts": int(len(reset_warning_accepts)),
        "episodes_requiring_reset_recovery": int(episodes_requiring_reset_recovery),
        "target_respawn_failures": int(len(target_respawn_failures)),
        "consecutive_duplicate_respawns": int(consecutive_duplicate_respawns),
        "reset_pollution_detected": bool(reset_pollution_detected),
        "ideal_gate_passed": bool(ideal_gate_pass),
        "minimum_gate_passed": bool(minimum_gate_pass),
        "world1_zeroshot_gate_passed": bool(ideal_gate_pass),
        "phase7_2b_allowed": bool(ideal_gate_pass or minimum_gate_pass),
        "woods_allowed": False,
        "episode_rows": episode_rows,
        "reset_rows": reset_rows,
    }


def write_report(path, summary):
    def fmt(value, digits=4):
        if isinstance(value, bool):
            return "是" if value else "否"
        if isinstance(value, float):
            return "{:.{}f}".format(value, digits)
        return str(value)

    lines = [
        "# Phase 7.2A World1 Sparse Obstacle Zero-Shot Validation",
        "",
        "1. evaluated model：{}".format(summary["evaluated_model"]),
        "2. evaluated vecnormalize：{}".format(summary["evaluated_vecnormalize"]),
        "3. world：{}".format(summary["world"]),
        "4. obstacles per episode：{}".format(summary["obstacles_per_episode"]),
        "5. episodes：{}".format(summary["episodes"]),
        "6. success rate：{} ({}/{})".format(
            fmt(summary["success_rate"]), summary["success_count"], summary["episodes"]
        ),
        "7. timeout rate：{}".format(fmt(summary["timeout_rate"])),
        "8. collision/emergency failure rate：{} ({}/{})".format(
            fmt(summary["collision_emergency_failure_rate"]),
            summary["collision_emergency_failure_count"],
            summary["episodes"],
        ),
        "9. out_of_bounds rate：{}".format(fmt(summary["out_of_bounds_rate"])),
        "10. height_violation rate：{}".format(fmt(summary["height_violation_rate"])),
        "11. target_visible_ratio_mean：{}".format(fmt(summary["target_visible_ratio_mean"])),
        "12. final_distance_mean：{} m".format(fmt(summary["final_distance_mean"])),
        "13. min_distance_mean：{} m".format(fmt(summary["min_distance_mean"])),
        "14. mean episode length：{}".format(fmt(summary["mean_episode_length"])),
        "15. RAW_TIMEOUT count：{}".format(summary["raw_timeout_count"]),
        "16. emergency_count：{}".format(summary["emergency_count"]),
        "17. depth_stop_count：{}".format(summary["depth_stop_count"]),
        "18. front_q05_min_mean：{} m".format(fmt(summary["front_q05_min_mean"])),
        "19. obstacle_danger_steps_mean：{}".format(fmt(summary["obstacle_danger_steps_mean"])),
        "20. action_vx mean/std/min/max：{:.4f} / {:.4f} / {:.4f} / {:.4f}".format(
            summary["action_vx_mean"],
            summary["action_vx_std"],
            summary["action_vx_min"],
            summary["action_vx_max"],
        ),
        "21. yaw_abs_mean：{}".format(fmt(summary["yaw_abs_mean"])),
        "22. 主要失败模式：{}".format(summary["failure_modes"]),
        "23. 是否通过 world1 zero-shot gate：{}".format(fmt(summary["world1_zeroshot_gate_passed"])),
        "24. 是否允许进入 Phase 7.2B：{}".format(fmt(summary["phase7_2b_allowed"])),
        "25. 是否允许进入 woods：{}".format(fmt(summary["woods_allowed"])),
        "",
        "Additional checks:",
        "- ideal_gate_passed：{}".format(fmt(summary["ideal_gate_passed"])),
        "- minimum_gate_passed：{}".format(fmt(summary["minimum_gate_passed"])),
        "- reset_failures：{}".format(summary["reset_failures"]),
        "- reset_attempt_failures：{}".format(summary["reset_attempt_failures"]),
        "- reset_warning_accepts：{}".format(summary["reset_warning_accepts"]),
        "- episodes_requiring_reset_recovery：{}".format(summary["episodes_requiring_reset_recovery"]),
        "- target_respawn_failures：{}".format(summary["target_respawn_failures"]),
        "- consecutive_duplicate_respawns：{}".format(summary["consecutive_duplicate_respawns"]),
        "- reset_pollution_detected：{}".format(fmt(summary["reset_pollution_detected"])),
        "- offboard_drop_count：{}".format(summary["offboard_drop_count"]),
        "- offboard_drop_episodes：{}".format(summary["offboard_drop_episodes"]),
        "- offboard_frequent_drop：{}".format(fmt(summary["offboard_frequent_drop"])),
    ]
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Phase 7.2A world1 zero-shot eval for the world0 best PPO policy.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--vecnormalize", required=True)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--world", default="world_1")
    parser.add_argument("--deterministic", dest="deterministic", action="store_true", default=True)
    parser.add_argument("--stochastic", dest="deterministic", action="store_false")
    parser.add_argument("--config", default=None)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed-base", type=int, default=72000)
    parser.add_argument("--reset-retries", type=int, default=3)
    parser.add_argument("--num-obstacles", type=int, default=4)
    return parser


def main():
    args = build_arg_parser().parse_args()
    require_training_deps()

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from gazebo_chase_env import GazeboChaseEnv

    config_path = args.config or infer_config_path(args.model)
    config = load_yaml(config_path)
    env_kwargs = env_kwargs_from_config(config, args.world)
    env_cfg = dict(config.get("env", {}))
    env_cfg.update(env_kwargs)

    os.makedirs(args.output_dir, exist_ok=True)
    rollout_csv_path = os.path.join(args.output_dir, "world1_zeroshot_rollouts.csv")
    summary_path = os.path.join(args.output_dir, "world1_zeroshot_summary.json")
    report_path = os.path.join(args.output_dir, "phase7_2a_report.md")

    raw_env = GazeboChaseEnv(**env_kwargs)
    vec_env = DummyVecEnv([lambda: raw_env])
    vec_env = VecNormalize.load(args.vecnormalize, vec_env)
    vec_env.training = False
    vec_env.norm_reward = False
    model = PPO.load(args.model)
    monitor = CommandMonitor()

    rollout_rows = []
    episode_rows = []
    reset_rows = []
    try:
        for episode in range(int(args.episodes)):
            seed = int(args.seed_base) + int(episode)
            obs, reset_info, reset_row, reset_attempts = reset_with_recovery(
                raw_env, episode, seed, args.reset_retries
            )
            reset_rows.extend(reset_attempts)
            if not reset_row["accepted_for_episode"]:
                raise RuntimeError("episode {} reset failed after recovery: {}".format(episode, reset_row))
            if reset_row["reset_mode_used"] != "episode_soft":
                raise RuntimeError(
                    "episode {} used reset_mode={} instead of episode_soft".format(
                        episode, reset_row["reset_mode_used"]
                    )
                )
            if not reset_row["target_respawn_success"]:
                raise RuntimeError("episode {} target respawn failed: {}".format(episode, reset_row))

            visible_count = 0
            min_distance = float("inf")
            final_distance = float("inf")
            front_q05_min = float("inf")
            obstacle_danger_steps = 0
            emergency_count = 0
            depth_stop_count = 0
            raw_timeout_count = 0
            offboard_drop_count = 0
            total_reward = 0.0
            done_reason = ""
            success = False
            timeout = False
            collision = False
            out_of_bounds = False
            height_violation = False
            step = 0

            while True:
                norm_obs = vec_env.normalize_obs(np.asarray([obs], dtype=np.float32))
                action, _state = model.predict(norm_obs, deterministic=bool(args.deterministic))
                action_row = action[0] if len(action.shape) > 1 else action
                raw_cmd = action_to_raw_cmd(action_row, env_cfg)
                obs, reward, terminated, truncated, info = raw_env.step(action_row)
                command_snapshot = monitor.snapshot(raw_cmd)
                reward = float(reward)
                done = bool(terminated or truncated)
                done_reason = str(info.get("terminal_reason", ""))
                mode = str(info.get("safety_mode", ""))
                target_visible = bool(info.get("target_visible", False))
                target_depth = finite(info.get("target_distance", math.nan))
                front_q05_depth = finite(info.get("front_q05_depth", obs[7] if len(obs) > 7 else math.nan))
                left_q05_depth = finite(obs[8] if len(obs) > 8 else math.nan)
                right_q05_depth = finite(obs[9] if len(obs) > 9 else math.nan)
                obstacle_area_ratio = finite(info.get("obstacle_area_ratio", obs[10] if len(obs) > 10 else math.nan))
                obstacle_danger = bool(float(info.get("obstacle_danger", obs[11] if len(obs) > 11 else 0.0)) > 0.5)

                min_distance = min(min_distance, target_depth)
                final_distance = target_depth
                front_q05_min = min(front_q05_min, front_q05_depth)
                visible_count += int(target_visible)
                obstacle_danger_steps += int(obstacle_danger)
                emergency_count += int("EMERGENCY_AVOID" in mode)
                depth_stop_count += int("DEPTH_STOP" in mode)
                raw_timeout_count += int("RAW_TIMEOUT" in mode)
                offboard_drop_count += int(str(info.get("mavros_mode", "")) != "OFFBOARD")
                total_reward += reward
                success = bool(info.get("success", False)) or done_reason == "success"
                timeout = bool(info.get("timeout", False)) or done_reason == "timeout"
                collision = bool(info.get("collision", False)) or done_reason == "collision_or_too_close"
                out_of_bounds = bool(info.get("out_of_bounds", False)) or done_reason == "out_of_bounds"
                height_violation = bool(info.get("height_violation", False)) or done_reason == "height_violation"

                rollout_rows.append(
                    {
                        "episode": int(episode),
                        "seed": int(seed),
                        "step": int(step),
                        "target_visible": bool(target_visible),
                        "target_depth": target_depth,
                        "target_u": finite(info.get("target_u", math.nan)),
                        "target_v": finite(info.get("target_v", math.nan)),
                        "front_q05_depth": front_q05_depth,
                        "left_q05_depth": left_q05_depth,
                        "right_q05_depth": right_q05_depth,
                        "obstacle_area_ratio": obstacle_area_ratio,
                        "obstacle_danger": bool(obstacle_danger),
                        "safety_mode": mode,
                        "action_vx": finite(action_row[0]),
                        "action_vz": finite(action_row[2]),
                        "action_yaw": finite(action_row[3]),
                        "raw_vx_body": command_snapshot["raw_vx_body"],
                        "filtered_vx_body": command_snapshot["filtered_vx_body"],
                        "published_vx_world": command_snapshot["published_vx_world"],
                        "drone_z": finite(info.get("drone_z", math.nan)),
                        "reward": reward,
                        "done": bool(done),
                        "done_reason": done_reason,
                        "success": bool(success),
                        "timeout": bool(timeout),
                        "collision": bool(collision),
                        "out_of_bounds": bool(out_of_bounds),
                        "height_violation": bool(height_violation),
                        "mavros_connected": bool(info.get("mavros_connected", False)),
                        "mavros_mode": str(info.get("mavros_mode", "")),
                        "mavros_armed": bool(info.get("mavros_armed", False)),
                        "action_vy": finite(action_row[1]),
                        "raw_vz_body": command_snapshot["raw_vz_body"],
                        "raw_yaw": command_snapshot["raw_yaw"],
                        "filtered_vz_body": command_snapshot["filtered_vz_body"],
                        "filtered_yaw": command_snapshot["filtered_yaw"],
                        "published_vy_world": command_snapshot["published_vy_world"],
                        "published_vz_world": command_snapshot["published_vz_world"],
                        "published_yaw": command_snapshot["published_yaw"],
                        "commanded_raw_vx_body": raw_cmd["commanded_raw_vx_body"],
                        "commanded_raw_vy_body": raw_cmd["commanded_raw_vy_body"],
                        "commanded_raw_vz_body": raw_cmd["commanded_raw_vz_body"],
                        "commanded_raw_yaw": raw_cmd["commanded_raw_yaw"],
                        "raw_vx_body_source": command_snapshot["raw_vx_body_source"],
                        "filtered_vx_body_available": command_snapshot["filtered_vx_body_available"],
                        "published_vx_world_available": command_snapshot["published_vx_world_available"],
                        "target_visible_ratio_so_far": rate(visible_count, step + 1),
                        "min_distance_so_far": min_distance,
                        "drone_x": finite(info.get("drone_x", math.nan)),
                        "drone_y": finite(info.get("drone_y", math.nan)),
                        "success_count": int(info.get("success_count", 0)),
                    }
                )
                step += 1
                if done:
                    break

            episode_row = {
                "episode": int(episode),
                "seed": int(seed),
                "success": bool(success),
                "timeout": bool(timeout),
                "collision": bool(collision),
                "out_of_bounds": bool(out_of_bounds),
                "height_violation": bool(height_violation),
                "done_reason": done_reason or "unknown",
                "final_distance": final_distance,
                "min_distance": min_distance,
                "episode_length": int(step),
                "target_visible_ratio": rate(visible_count, step),
                "emergency_count": int(emergency_count),
                "depth_stop_count": int(depth_stop_count),
                "raw_timeout_count": int(raw_timeout_count),
                "front_q05_min": front_q05_min,
                "obstacle_danger_steps": int(obstacle_danger_steps),
                "offboard_drop_count": int(offboard_drop_count),
                "reward": float(total_reward),
            }
            episode_rows.append(episode_row)
            print(
                "episode={} seed={} success={} reason={} final={:.3f} min={:.3f} "
                "visible={:.3f} front_min={:.3f} emergency={} depth_stop={}".format(
                    episode,
                    seed,
                    episode_row["success"],
                    episode_row["done_reason"],
                    episode_row["final_distance"],
                    episode_row["min_distance"],
                    episode_row["target_visible_ratio"],
                    episode_row["front_q05_min"],
                    episode_row["emergency_count"],
                    episode_row["depth_stop_count"],
                )
            )
    finally:
        monitor.close()
        vec_env.close()

    with open(rollout_csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rollout_fieldnames())
        writer.writeheader()
        writer.writerows(rollout_rows)

    summary = build_summary(args, config_path, config, episode_rows, rollout_rows, reset_rows)
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    write_report(report_path, summary)

    print("wrote {}".format(rollout_csv_path))
    print("wrote {}".format(summary_path))
    print("wrote {}".format(report_path))
    print("world1 zero-shot ideal gate passed={}".format(summary["world1_zeroshot_gate_passed"]))
    print("world1 zero-shot minimum gate passed={}".format(summary["minimum_gate_passed"]))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("phase7_world1_zeroshot_eval failed: {}".format(exc), file=sys.stderr)
        sys.exit(1)
