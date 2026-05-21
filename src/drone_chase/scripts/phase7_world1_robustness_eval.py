#!/usr/bin/env python3

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import threading
import time
from datetime import datetime

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "envs"))
if ENV_DIR not in sys.path:
    sys.path.insert(0, ENV_DIR)


DEFAULT_OUTPUT_DIR = "/home/whk/vf_ws/outputs/phase7/world1_robustness_from_world0_best"
DEFAULT_CONFIG = "/home/whk/vf_ws/src/drone_chase/config/phase7_ppo_world0_from_bc_v2.yaml"
WOODS_PREFIXES = ("woods", "woods_easy", "woods_hard", "random_woods")
GROUP_SPECS = [
    {"group": "A", "name": "standard_sparse_recheck", "num_obstacles": 4, "episodes": 30},
    {"group": "B", "name": "more_obstacles", "num_obstacles": 6, "episodes": 30},
    {"group": "C", "name": "sparse_medium_boundary", "num_obstacles": 8, "episodes": 20},
    {"group": "D", "name": "front_obstacle_intervention", "num_obstacles": 4, "episodes": 10},
]


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
        raise RuntimeError("Phase 7.2B is world1-only; refusing world={}".format(world))
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


def run_helper(script_name, args, timeout):
    cmd = [os.path.join(SCRIPT_DIR, script_name)] + list(args)
    result = subprocess.run(
        cmd,
        check=False,
        timeout=timeout,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )
    return bool(result.returncode == 0), result.stdout[-2000:]


def delete_diagnostic_obstacle(timeout=10.0):
    return run_helper("diagnostic_spawn_front_obstacle.py", ["--delete-only"], timeout)


def reset_world_obstacles(num_obstacles, seed, timeout=35.0):
    _delete_ok, output = delete_diagnostic_obstacle(timeout=10.0)
    obstacle_ok, obstacle_output = run_helper(
        "spawn_random_obstacles.py",
        ["--num", str(int(num_obstacles)), "--seed", str(int(seed))],
        timeout,
    )
    return bool(obstacle_ok), "{}\n{}".format(output, obstacle_output)[-2500:]


def spawn_front_obstacle(front_face_distance, timeout=10.0):
    return run_helper(
        "diagnostic_spawn_front_obstacle.py",
        [
            "--relative-to-uav",
            "--front-face-distance",
            str(float(front_face_distance)),
            "--center-z-mode",
            "uav",
            "--x-size",
            "0.30",
            "--y-size",
            "1.20",
            "--z-size",
            "2.00",
        ],
        timeout,
    )


def reset_for_episode(env, seed):
    return env.reset(seed=int(seed), options={"reset_mode": "episode_soft"})


def make_reset_row(group, episode, seed, attempt, reset_info, external_reset_ok, external_reset_tail):
    return {
        "group": str(group),
        "episode": int(episode),
        "seed": int(seed),
        "attempt": int(attempt),
        "external_reset_success": bool(external_reset_ok),
        "external_reset_output_tail": str(external_reset_tail or ""),
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
        row["external_reset_success"]
        and row["topics_ready"]
        and row["target_respawn_success"]
        and row["initial_target_visible"]
        and row["reset_mode_used"] == "episode_soft"
        and row["mavros_mode"] == "OFFBOARD"
        and not any(token in safety_mode for token in blocked)
    )


def reset_with_recovery(env, group, episode, seed, num_obstacles, retries):
    attempts = []
    last_obs = None
    last_info = None
    for attempt in range(int(retries) + 1):
        attempt_seed = seed + attempt * 100000
        external_reset_ok, external_tail = reset_world_obstacles(num_obstacles, attempt_seed)
        obs, reset_info = reset_for_episode(env, attempt_seed)
        row = make_reset_row(group, episode, seed, attempt, reset_info, external_reset_ok, external_tail)
        row["reset_seed"] = int(attempt_seed)
        attempts.append(row)
        last_obs = obs
        last_info = reset_info
        if row["external_reset_success"] and row["reset_success"] and row["initial_target_visible"]:
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
        "global_episode",
        "group",
        "seed",
        "num_obstacles",
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
        "front_obstacle_present",
        "front_obstacle_deleted",
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


def group_rows(episode_rows, group):
    return [row for row in episode_rows if row["group"] == group]


def success_rate(rows):
    return rate(sum(1 for row in rows if row["success"]), len(rows))


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
    collision_emergency_failure_count = sum(
        1 for row in episode_rows if not row["success"] and (row["collision"] or row["emergency_count"] > 0)
    )
    offboard_drop_count = sum(int(row["offboard_drop"]) for row in episode_rows)
    reset_failure_rows = [row for row in reset_rows if row.get("accepted_for_episode", False) and not row.get("reset_success", False)]
    external_reset_failures = [
        row for row in reset_rows if row.get("accepted_for_episode", False) and not row.get("external_reset_success", False)
    ]
    target_respawn_failures = [
        row for row in reset_rows if row.get("accepted_for_episode", False) and not row.get("target_respawn_success", True)
    ]
    reset_warning_accepts = [row for row in reset_rows if row.get("reset_accepted_with_warning", False)]
    episode_reset_rows = [row for row in reset_rows if row.get("accepted_for_episode", False)]
    consecutive_duplicate_respawns = 0
    previous_by_group = {}
    for row in episode_reset_rows:
        group = row["group"]
        msg = str(row.get("target_respawn_message", ""))
        if msg and previous_by_group.get(group) == msg:
            consecutive_duplicate_respawns += 1
        previous_by_group[group] = msg
    reset_pollution_detected = bool(
        reset_failure_rows
        or external_reset_failures
        or target_respawn_failures
        or consecutive_duplicate_respawns > 0
    )
    offboard_frequent_drop = bool(offboard_drop_count > max(1, int(0.01 * max(1, len(rollout_rows)))))
    action_vx_stats = stats(row["action_vx"] for row in rollout_rows)
    yaw_abs_mean = mean(abs(float(row["action_yaw"])) for row in rollout_rows) if rollout_rows else 0.0
    group_rates = {spec["group"]: success_rate(group_rows(episode_rows, spec["group"])) for spec in GROUP_SPECS}

    d_rows = group_rows(episode_rows, "D")
    d_danger = sum(1 for row in d_rows if row["intervention_danger_seen"]) == len(d_rows) if d_rows else False
    d_safety = sum(1 for row in d_rows if row["intervention_safety_triggered"]) == len(d_rows) if d_rows else False
    d_filtered_stop = sum(1 for row in d_rows if row["intervention_filtered_vx_nonpositive"]) == len(d_rows) if d_rows else False
    d_no_collision = all(not row["collision"] for row in d_rows) if d_rows else False
    d_no_offboard = all(int(row["offboard_drop"]) == 0 for row in d_rows) if d_rows else False
    d_recovered_or_safe = all(row["intervention_recovered_or_safe"] for row in d_rows) if d_rows else False
    group_d_passed = bool(d_danger and d_safety and d_filtered_stop and d_no_collision and d_no_offboard and d_recovered_or_safe)

    total_success_rate = rate(success_count, n)
    phase_passed = bool(
        n >= 90
        and total_success_rate >= 0.80
        and group_rates.get("A", 0.0) >= 0.90
        and group_rates.get("B", 0.0) >= 0.80
        and group_rates.get("C", 0.0) >= 0.60
        and raw_timeout_count == 0
        and out_of_bounds_count == 0
        and height_violation_count == 0
        and not offboard_frequent_drop
        and not reset_pollution_detected
        and group_d_passed
    )

    return {
        "phase": "7.2B",
        "report_date": datetime.now().isoformat(timespec="seconds"),
        "evaluated_model": os.path.abspath(args.model),
        "evaluated_vecnormalize": os.path.abspath(args.vecnormalize),
        "config": os.path.abspath(config_path),
        "world": "world_1",
        "deterministic": bool(args.deterministic),
        "seed_base": int(args.seed_base),
        "front_face_distance": float(args.front_face_distance),
        "total_episodes": int(n),
        "total_success_count": int(success_count),
        "total_success_rate": total_success_rate,
        "group_success_rates": group_rates,
        "group_episode_counts": {spec["group"]: len(group_rows(episode_rows, spec["group"])) for spec in GROUP_SPECS},
        "timeout_rate": rate(timeout_count, n),
        "collision_count": int(collision_count),
        "collision_rate": rate(collision_count, n),
        "collision_emergency_failure_count": int(collision_emergency_failure_count),
        "collision_emergency_failure_rate": rate(collision_emergency_failure_count, n),
        "out_of_bounds_count": int(out_of_bounds_count),
        "out_of_bounds_rate": rate(out_of_bounds_count, n),
        "height_violation_count": int(height_violation_count),
        "height_violation_rate": rate(height_violation_count, n),
        "target_visible_ratio_mean": mean(row["target_visible_ratio"] for row in episode_rows),
        "final_distance_mean": mean(row["final_distance"] for row in episode_rows),
        "min_distance_mean": mean(row["min_distance"] for row in episode_rows),
        "raw_timeout_count": int(raw_timeout_count),
        "emergency_count": int(emergency_count),
        "depth_stop_count": int(depth_stop_count),
        "front_q05_min_mean": mean(row["front_q05_min"] for row in episode_rows),
        "obstacle_danger_steps_mean": mean(row["obstacle_danger_steps"] for row in episode_rows),
        "offboard_drop_count": int(offboard_drop_count),
        "offboard_frequent_drop": bool(offboard_frequent_drop),
        "reset_failures": int(len(reset_failure_rows)),
        "external_reset_failures": int(len(external_reset_failures)),
        "reset_warning_accepts": int(len(reset_warning_accepts)),
        "target_respawn_failures": int(len(target_respawn_failures)),
        "consecutive_duplicate_respawns": int(consecutive_duplicate_respawns),
        "reset_pollution_detected": bool(reset_pollution_detected),
        "action_vx_mean": action_vx_stats["mean"],
        "action_vx_std": action_vx_stats["std"],
        "action_vx_min": action_vx_stats["min"],
        "action_vx_max": action_vx_stats["max"],
        "yaw_abs_mean": yaw_abs_mean,
        "main_failure_modes": classify_failure_modes(episode_rows),
        "group_d_danger_seen_all": bool(d_danger),
        "group_d_safety_triggered_all": bool(d_safety),
        "group_d_filtered_vx_nonpositive_all": bool(d_filtered_stop),
        "group_d_no_collision": bool(d_no_collision),
        "group_d_no_offboard_drop": bool(d_no_offboard),
        "group_d_recovered_or_safe": bool(d_recovered_or_safe),
        "group_d_safety_intervention_passed": bool(group_d_passed),
        "phase7_2b_passed": bool(phase_passed),
        "phase7_2c_allowed": bool(phase_passed),
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

    d_result = (
        "danger_seen={} safety_triggered={} filtered_vx<=0={} no_collision={} "
        "no_offboard={} recovered_or_safe={} passed={}"
    ).format(
        fmt(summary["group_d_danger_seen_all"]),
        fmt(summary["group_d_safety_triggered_all"]),
        fmt(summary["group_d_filtered_vx_nonpositive_all"]),
        fmt(summary["group_d_no_collision"]),
        fmt(summary["group_d_no_offboard_drop"]),
        fmt(summary["group_d_recovered_or_safe"]),
        fmt(summary["group_d_safety_intervention_passed"]),
    )
    lines = [
        "# Phase 7.2B World1 Sparse Obstacle Robustness / Stress Validation",
        "",
        "1. evaluated model：{}".format(summary["evaluated_model"]),
        "2. evaluated vecnormalize：{}".format(summary["evaluated_vecnormalize"]),
        "3. total episodes：{}".format(summary["total_episodes"]),
        "4. Group A success rate：{}".format(fmt(summary["group_success_rates"].get("A", 0.0))),
        "5. Group B success rate：{}".format(fmt(summary["group_success_rates"].get("B", 0.0))),
        "6. Group C success rate：{}".format(fmt(summary["group_success_rates"].get("C", 0.0))),
        "7. Group D safety intervention result：{}".format(d_result),
        "8. total success rate：{} ({}/{})".format(
            fmt(summary["total_success_rate"]),
            summary["total_success_count"],
            summary["total_episodes"],
        ),
        "9. timeout rate：{}".format(fmt(summary["timeout_rate"])),
        "10. collision/emergency failure rate：{} ({}/{})".format(
            fmt(summary["collision_emergency_failure_rate"]),
            summary["collision_emergency_failure_count"],
            summary["total_episodes"],
        ),
        "11. out_of_bounds rate：{}".format(fmt(summary["out_of_bounds_rate"])),
        "12. height_violation rate：{}".format(fmt(summary["height_violation_rate"])),
        "13. target_visible_ratio_mean：{}".format(fmt(summary["target_visible_ratio_mean"])),
        "14. final_distance_mean：{} m".format(fmt(summary["final_distance_mean"])),
        "15. min_distance_mean：{} m".format(fmt(summary["min_distance_mean"])),
        "16. RAW_TIMEOUT count：{}".format(summary["raw_timeout_count"]),
        "17. emergency_count：{}".format(summary["emergency_count"]),
        "18. depth_stop_count：{}".format(summary["depth_stop_count"]),
        "19. front_q05_min_mean：{} m".format(fmt(summary["front_q05_min_mean"])),
        "20. obstacle_danger_steps_mean：{}".format(fmt(summary["obstacle_danger_steps_mean"])),
        "21. OFFBOARD drop count：{}".format(summary["offboard_drop_count"]),
        "22. reset pollution detected：{}".format(fmt(summary["reset_pollution_detected"])),
        "23. main failure modes：{}".format(summary["main_failure_modes"]),
        "24. 是否通过 Phase 7.2B：{}".format(fmt(summary["phase7_2b_passed"])),
        "25. 是否允许进入 Phase 7.2C：{}".format(fmt(summary["phase7_2c_allowed"])),
        "26. 是否允许进入 woods：{}".format(fmt(summary["woods_allowed"])),
    ]
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Phase 7.2B world1 sparse robustness/stress eval.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--vecnormalize", required=True)
    parser.add_argument("--world", default="world_1")
    parser.add_argument("--deterministic", dest="deterministic", action="store_true", default=True)
    parser.add_argument("--stochastic", dest="deterministic", action="store_false")
    parser.add_argument("--config", default=None)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed-base", type=int, default=72100)
    parser.add_argument("--reset-retries", type=int, default=2)
    parser.add_argument("--front-face-distance", type=float, default=0.45)
    parser.add_argument("--intervention-delete-after-steps", type=int, default=12)
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
    rollout_csv_path = os.path.join(args.output_dir, "world1_robustness_rollouts.csv")
    summary_path = os.path.join(args.output_dir, "world1_robustness_summary.json")
    report_path = os.path.join(args.output_dir, "phase7_2b_report.md")

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
    global_episode = 0
    try:
        for spec in GROUP_SPECS:
            group = spec["group"]
            for local_episode in range(int(spec["episodes"])):
                seed = int(args.seed_base) + global_episode
                obs, reset_info, reset_row, reset_attempts = reset_with_recovery(
                    raw_env,
                    group,
                    local_episode,
                    seed,
                    spec["num_obstacles"],
                    args.reset_retries,
                )
                reset_rows.extend(reset_attempts)
                if not reset_row["accepted_for_episode"]:
                    raise RuntimeError(
                        "group {} episode {} reset failed after recovery: {}".format(
                            group, local_episode, reset_row
                        )
                    )

                front_obstacle_present = False
                front_obstacle_deleted = False
                front_spawn_success = False
                front_spawn_tail = ""
                front_delete_success = False
                front_delete_tail = ""
                if group == "D":
                    front_spawn_success, front_spawn_tail = spawn_front_obstacle(args.front_face_distance)
                    if not front_spawn_success:
                        raise RuntimeError(
                            "group D episode {} front obstacle spawn failed: {}".format(
                                local_episode, front_spawn_tail
                            )
                        )
                    front_obstacle_present = True
                    time.sleep(0.3)

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
                intervention_danger_seen = False
                intervention_safety_triggered = False
                intervention_filtered_vx_nonpositive = False
                intervention_recovered_after_delete = False
                post_delete_safe_steps = 0
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

                    if group == "D":
                        if front_obstacle_present and obstacle_danger:
                            intervention_danger_seen = True
                        if front_obstacle_present and ("EMERGENCY_AVOID" in mode or "DEPTH_STOP" in mode):
                            intervention_safety_triggered = True
                        if (
                            front_obstacle_present
                            and obstacle_danger
                            and finite(command_snapshot["filtered_vx_body"], 1.0) <= 0.0
                        ):
                            intervention_filtered_vx_nonpositive = True
                        if front_obstacle_present and step + 1 >= int(args.intervention_delete_after_steps):
                            front_delete_success, front_delete_tail = delete_diagnostic_obstacle()
                            front_obstacle_present = False
                            front_obstacle_deleted = True
                        if front_obstacle_deleted and not front_obstacle_present:
                            if "EMERGENCY_AVOID" not in mode and "DEPTH_STOP" not in mode and str(info.get("mavros_mode", "")) == "OFFBOARD":
                                post_delete_safe_steps += 1
                            if post_delete_safe_steps >= 5 or success:
                                intervention_recovered_after_delete = True

                    rollout_rows.append(
                        {
                            "episode": int(local_episode),
                            "global_episode": int(global_episode),
                            "group": group,
                            "seed": int(seed),
                            "num_obstacles": int(spec["num_obstacles"]),
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
                            "front_obstacle_present": bool(front_obstacle_present),
                            "front_obstacle_deleted": bool(front_obstacle_deleted),
                        }
                    )
                    step += 1
                    if done:
                        break

                if group == "D" and front_obstacle_present:
                    front_delete_success, front_delete_tail = delete_diagnostic_obstacle()
                    front_obstacle_deleted = True
                    front_obstacle_present = False

                episode_row = {
                    "episode": int(local_episode),
                    "global_episode": int(global_episode),
                    "group": group,
                    "seed": int(seed),
                    "num_obstacles": int(spec["num_obstacles"]),
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
                    "offboard_drop": int(offboard_drop_count),
                    "reset_recovery_used": bool(int(reset_row.get("attempt", 0)) > 0),
                    "reward": float(total_reward),
                    "front_obstacle_spawn_success": bool(front_spawn_success) if group == "D" else None,
                    "front_obstacle_spawn_output_tail": front_spawn_tail if group == "D" else "",
                    "front_obstacle_delete_success": bool(front_delete_success) if group == "D" else None,
                    "front_obstacle_delete_output_tail": front_delete_tail if group == "D" else "",
                    "intervention_danger_seen": bool(intervention_danger_seen) if group == "D" else None,
                    "intervention_safety_triggered": bool(intervention_safety_triggered) if group == "D" else None,
                    "intervention_filtered_vx_nonpositive": bool(intervention_filtered_vx_nonpositive) if group == "D" else None,
                    "intervention_recovered_or_safe": bool(intervention_recovered_after_delete or success) if group == "D" else None,
                }
                episode_rows.append(episode_row)
                print(
                    "group={} episode={} seed={} success={} reason={} final={:.3f} min={:.3f} "
                    "visible={:.3f} front_min={:.3f} danger_steps={} emergency={} depth_stop={} offboard_drop={}".format(
                        group,
                        local_episode,
                        seed,
                        episode_row["success"],
                        episode_row["done_reason"],
                        episode_row["final_distance"],
                        episode_row["min_distance"],
                        episode_row["target_visible_ratio"],
                        episode_row["front_q05_min"],
                        episode_row["obstacle_danger_steps"],
                        episode_row["emergency_count"],
                        episode_row["depth_stop_count"],
                        episode_row["offboard_drop"],
                    )
                )
                global_episode += 1
    finally:
        try:
            delete_diagnostic_obstacle()
        except Exception:
            pass
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
    print("phase7_2b_passed={}".format(summary["phase7_2b_passed"]))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("phase7_world1_robustness_eval failed: {}".format(exc), file=sys.stderr)
        sys.exit(1)
