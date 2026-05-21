#!/usr/bin/env python3

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "envs"))
if ENV_DIR not in sys.path:
    sys.path.insert(0, ENV_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from phase7_random_woods_zeroshot_eval import (  # noqa: E402
    CommandMonitor,
    MANIFEST_PATH,
    REGISTERED_MODEL,
    REGISTERED_VECNORMALIZE,
    action_to_raw_cmd,
    build_diagnostic_answers,
    finite,
    load_yaml,
    mean,
    rate,
    require_eval_deps,
    verify_registry_policy,
)


DEFAULT_OUTPUT_DIR = "/home/whk/vf_ws/outputs/phase7/random_woods_robustness_from_final_policy"
DEFAULT_CONFIG = (
    "/home/whk/vf_ws/outputs/phase7/world0_ppo_from_bc_v2_10k_conservative_run4/config_effective.yaml"
)

GROUP_SPECS = [
    {
        "group": "A",
        "name": "random_woods_standard_recheck",
        "episodes": 30,
        "num_trunks": 18,
        "num_branches": 45,
        "num_fallen": 10,
        "area_x_min": 1.0,
        "area_x_max": 6.5,
        "area_y_min": -3.5,
        "area_y_max": 3.5,
        "uav_clearance": 1.0,
        "target_clearance": 0.6,
        "target_distance": 4.0,
        "target_lateral_cycle": [0.0],
    },
    {
        "group": "B",
        "name": "random_woods_multi_seed",
        "episodes": 30,
        "num_trunks": 18,
        "num_branches": 45,
        "num_fallen": 10,
        "area_x_min": 1.0,
        "area_x_max": 6.5,
        "area_y_min": -3.5,
        "area_y_max": 3.5,
        "uav_clearance": 1.0,
        "target_clearance": 0.6,
        "target_distance": 4.0,
        "target_lateral_cycle": [-0.45, 0.0, 0.45],
    },
    {
        "group": "C",
        "name": "random_woods_safety_stress",
        "episodes": 20,
        "num_trunks": 22,
        "num_branches": 55,
        "num_fallen": 12,
        "area_x_min": 0.8,
        "area_x_max": 6.0,
        "area_y_min": -3.0,
        "area_y_max": 3.0,
        "uav_clearance": 0.75,
        "target_clearance": 0.35,
        "target_distance": 4.0,
        "target_lateral_cycle": [-0.55, 0.0, 0.55],
    },
    {
        "group": "D",
        "name": "front_obstacle_intervention",
        "episodes": 10,
        "num_trunks": 18,
        "num_branches": 45,
        "num_fallen": 10,
        "area_x_min": 1.0,
        "area_x_max": 6.5,
        "area_y_min": -3.5,
        "area_y_max": 3.5,
        "uav_clearance": 1.0,
        "target_clearance": 0.6,
        "target_distance": 4.0,
        "target_lateral_cycle": [0.0],
    },
]


def is_safety_mode(mode):
    mode = str(mode)
    return "EMERGENCY_AVOID" in mode or "DEPTH_STOP" in mode


def group_spec(group):
    for spec in GROUP_SPECS:
        if spec["group"] == group:
            return spec
    raise KeyError(group)


def run_helper(script_name, args, timeout):
    cmd = [os.path.join(SCRIPT_DIR, script_name)] + list(args)
    result = subprocess.run(
        cmd,
        check=False,
        timeout=float(timeout),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )
    return bool(result.returncode == 0), (result.stdout or "")[-2500:]


def delete_front_obstacle(timeout=10.0):
    return run_helper("diagnostic_spawn_front_obstacle.py", ["--delete-only"], timeout)


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


def base_env_kwargs(config, seed, world):
    env_cfg = dict(config.get("env", {}))
    env_cfg["world_type"] = str(world)
    env_cfg["reset_mode"] = "soft"
    env_cfg["respawn_target_on_reset"] = False
    env_cfg["seed"] = int(seed)
    env_cfg.setdefault("reset_ready_timeout", 20.0)
    env_cfg.setdefault("reset_zero_cmd_duration", 1.0)
    env_cfg.setdefault("max_target_respawn_attempts", 5)
    return {key: value for key, value in env_cfg.items() if value is not None}


def apply_group_config(env, env_cfg, spec, seed, local_episode):
    lateral_cycle = spec.get("target_lateral_cycle", [0.0])
    lateral = float(lateral_cycle[int(local_episode) % len(lateral_cycle)])
    env.config.update(env_cfg)
    env.config.update(
        {
            "world_type": "random_woods",
            "reset_mode": "soft",
            "respawn_target_on_reset": False,
            "seed": int(seed),
            "random_woods_num_trunks": int(spec["num_trunks"]),
            "random_woods_num_branches": int(spec["num_branches"]),
            "random_woods_num_fallen": int(spec["num_fallen"]),
            "random_woods_area_x_min": float(spec["area_x_min"]),
            "random_woods_area_x_max": float(spec["area_x_max"]),
            "random_woods_area_y_min": float(spec["area_y_min"]),
            "random_woods_area_y_max": float(spec["area_y_max"]),
            "random_woods_uav_clearance": float(spec["uav_clearance"]),
            "random_woods_target_clearance": float(spec["target_clearance"]),
            "woods_reset_target_distance": float(spec["target_distance"]),
            "woods_reset_target_lateral": lateral,
            "woods_reset_target_relative_to_uav": True,
        }
    )
    env.world_type = "random_woods"
    env.reset_mode = "soft"
    env.respawn_target_on_reset = False
    env.woods_reset_target_distance = float(spec["target_distance"])
    env.woods_reset_target_lateral = lateral
    env.woods_reset_target_relative_to_uav = True
    return lateral


def reset_ready_enough(info):
    safety_mode = str(info.get("safety_mode", ""))
    blocked = ("WAIT_FCU", "PRESTREAM", "SET_MODE", "ARMING", "TAKEOFF")
    return bool(
        info.get("topics_ready", False)
        and info.get("training_ready", False)
        and info.get("mavros_mode", "") == "OFFBOARD"
        and not any(token in safety_mode for token in blocked)
    )


def reset_with_recovery(env, env_cfg, spec, group, local_episode, global_episode, seed, retries):
    attempts = []
    last_obs = None
    last_info = {}
    for attempt in range(int(retries) + 1):
        attempt_seed = int(seed) + attempt * 100000
        target_lateral = apply_group_config(env, env_cfg, spec, attempt_seed, local_episode)
        obs, info = env.reset(seed=attempt_seed, options={"reset_mode": "soft"})
        row = {
            "group": group,
            "episode": int(local_episode),
            "global_episode": int(global_episode),
            "seed": int(seed),
            "attempt": int(attempt),
            "reset_seed": int(attempt_seed),
            "target_lateral": float(target_lateral),
            "reset_success": bool(info.get("reset_success", False)),
            "topics_ready": bool(info.get("topics_ready", False)),
            "training_ready": bool(info.get("training_ready", False)),
            "reset_mode_used": str(info.get("reset_mode_used", info.get("reset_mode", ""))),
            "mavros_mode": str(info.get("mavros_mode", "")),
            "safety_mode": str(info.get("safety_mode", "")),
            "target_visible": bool(info.get("target_visible", False)),
            "target_distance": finite(info.get("target_distance", math.nan)),
            "drone_z": finite(info.get("drone_z", math.nan)),
            "target_respawn_success": bool(info.get("target_respawn_success", True)),
            "target_respawn_message": str(info.get("target_respawn_message", "")),
            "reset_output_tail": str(info.get("reset_output_tail", "")),
            "accepted_for_episode": False,
            "accepted_with_warning": False,
        }
        attempts.append(row)
        last_obs = obs
        last_info = info
        if row["reset_success"] and row["target_visible"]:
            row["accepted_for_episode"] = True
            return obs, info, row, attempts
        time.sleep(0.5)
    final_row = attempts[-1]
    if reset_ready_enough(last_info):
        final_row["accepted_for_episode"] = True
        final_row["accepted_with_warning"] = True
        return last_obs, last_info, final_row, attempts
    return last_obs, last_info, final_row, attempts


def rollout_fieldnames():
    return [
        "episode",
        "global_episode",
        "group",
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
        "action_vy",
        "raw_vy_body",
        "raw_vz_body",
        "raw_yaw_rate",
        "filtered_vy_body",
        "filtered_vz_body",
        "filtered_yaw_rate",
        "published_vy_world",
        "published_vz_world",
        "published_yaw_rate",
        "mavros_mode",
        "mavros_armed",
        "safety_filtered",
        "filtered_vx_limited",
        "front_obstacle_present",
        "front_obstacle_deleted",
    ]


def episode_fieldnames():
    return [
        "group",
        "episode",
        "global_episode",
        "seed",
        "success",
        "timeout",
        "collision",
        "emergency_failure",
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
        "front_q05_min",
        "obstacle_danger_steps",
        "offboard_drop",
        "reset_recovery_used",
        "reset_pollution_detected",
        "reward",
        "target_lateral",
        "front_obstacle_spawn_success",
        "front_obstacle_delete_success",
        "intervention_danger_seen",
        "intervention_safety_triggered",
        "intervention_filtered_vx_nonpositive",
        "intervention_recovered_or_safe",
    ]


def make_reset_failure_episode(group, local_episode, global_episode, seed, reset_row):
    safety_mode = str(reset_row.get("safety_mode", ""))
    offboard_drop = int(reset_row.get("mavros_mode", "") != "OFFBOARD")
    return {
        "group": group,
        "episode": int(local_episode),
        "global_episode": int(global_episode),
        "seed": int(seed),
        "success": False,
        "timeout": False,
        "collision": False,
        "emergency_failure": False,
        "out_of_bounds": False,
        "height_violation": False,
        "done_reason": "reset_failure",
        "final_distance": finite(reset_row.get("target_distance", math.nan)),
        "min_distance": finite(reset_row.get("target_distance", math.nan)),
        "episode_length": 0,
        "target_visible_ratio": 1.0 if reset_row.get("target_visible", False) else 0.0,
        "emergency_count": int("EMERGENCY_AVOID" in safety_mode),
        "depth_stop_count": int("DEPTH_STOP" in safety_mode),
        "raw_timeout_count": int("RAW_TIMEOUT" in safety_mode),
        "front_q05_min": math.nan,
        "obstacle_danger_steps": 0,
        "offboard_drop": offboard_drop,
        "reset_recovery_used": bool(int(reset_row.get("attempt", 0)) > 0),
        "reset_pollution_detected": True,
        "reward": 0.0,
        "target_lateral": finite(reset_row.get("target_lateral", 0.0)),
        "front_obstacle_spawn_success": None,
        "front_obstacle_delete_success": None,
        "intervention_danger_seen": None,
        "intervention_safety_triggered": None,
        "intervention_filtered_vx_nonpositive": None,
        "intervention_recovered_or_safe": None,
    }


def run_episode(
    raw_env,
    vec_env,
    model,
    monitor,
    env_cfg,
    spec,
    local_episode,
    global_episode,
    seed,
    deterministic,
    reset_retries,
    front_face_distance,
    intervention_delete_after_steps,
):
    group = spec["group"]
    obs, _reset_info, reset_row, reset_attempts = reset_with_recovery(
        raw_env, env_cfg, spec, group, local_episode, global_episode, seed, reset_retries
    )
    if not reset_row["accepted_for_episode"]:
        episode_row = make_reset_failure_episode(group, local_episode, global_episode, seed, reset_row)
        return episode_row, [], [], reset_attempts

    front_obstacle_present = False
    front_obstacle_deleted = False
    front_spawn_success = None
    front_delete_success = None
    intervention_danger_seen = False
    intervention_safety_triggered = False
    intervention_filtered_vx_nonpositive = False
    intervention_recovered_or_safe = False
    post_delete_safe_steps = 0
    if group == "D":
        front_spawn_success, _tail = spawn_front_obstacle(front_face_distance)
        if not front_spawn_success:
            raise RuntimeError("Group D front obstacle spawn failed: {}".format(_tail))
        front_obstacle_present = True
        time.sleep(0.3)

    rollout_rows = []
    safety_events = []
    visible_count = 0
    min_distance = float("inf")
    final_distance = float("inf")
    front_depths = []
    raw_vx_values = []
    raw_vy_values = []
    raw_vz_values = []
    raw_yaw_values = []
    reward_total = 0.0
    safety_filtered_count = 0
    raw_timeout_count = 0
    obstacle_danger_steps = 0
    emergency_count = 0
    depth_stop_count = 0
    offboard_drop_count = 0
    success = False
    timeout = False
    collision = False
    out_of_bounds = False
    height_violation = False
    done_reason = ""
    step = 0

    while True:
        norm_obs = vec_env.normalize_obs(np.asarray([obs], dtype=np.float32))
        action, _state = model.predict(norm_obs, deterministic=bool(deterministic))
        action_row = action[0] if len(action.shape) > 1 else action
        raw_cmd = action_to_raw_cmd(action_row, env_cfg)
        obs, reward, terminated, truncated, info = raw_env.step(action_row)
        snapshot = monitor.snapshot(raw_cmd)

        mode = str(info.get("safety_mode", ""))
        done = bool(terminated or truncated)
        done_reason = str(info.get("terminal_reason", "")) if done else ""
        target_visible = bool(info.get("target_visible", False))
        target_depth = finite(info.get("target_distance", math.nan))
        front_q05_depth = finite(info.get("front_q05_depth", obs[7] if len(obs) > 7 else math.nan))
        left_q05_depth = finite(obs[8] if len(obs) > 8 else math.nan)
        right_q05_depth = finite(obs[9] if len(obs) > 9 else math.nan)
        obstacle_area_ratio = finite(info.get("obstacle_area_ratio", obs[10] if len(obs) > 10 else math.nan))
        obstacle_danger_value = finite(info.get("obstacle_danger", obs[11] if len(obs) > 11 else 0.0), 0.0)
        obstacle_danger = bool(obstacle_danger_value > 0.5)
        raw_vx = snapshot["raw_vx_body"]
        filtered_vx = snapshot["filtered_vx_body"]
        filtered_vx_limited = bool(
            math.isfinite(raw_vx)
            and math.isfinite(filtered_vx)
            and raw_vx > 0.05
            and filtered_vx < raw_vx - 0.05
        )
        safety_filtered = bool(is_safety_mode(mode) or filtered_vx_limited)

        visible_count += int(target_visible)
        min_distance = min(min_distance, target_depth)
        final_distance = target_depth
        front_depths.append(front_q05_depth)
        raw_vx_values.append(snapshot["raw_vx_body"])
        raw_vy_values.append(snapshot["raw_vy_body"])
        raw_vz_values.append(snapshot["raw_vz_body"])
        raw_yaw_values.append(snapshot["raw_yaw_rate"])
        reward_total += float(reward)
        safety_filtered_count += int(safety_filtered)
        raw_timeout_count += int("RAW_TIMEOUT" in mode)
        obstacle_danger_steps += int(obstacle_danger)
        emergency_count += int("EMERGENCY_AVOID" in mode)
        depth_stop_count += int("DEPTH_STOP" in mode)
        offboard_drop_count += int(str(info.get("mavros_mode", "")) != "OFFBOARD")
        success = bool(success or info.get("success", False) or done_reason == "success")
        timeout = bool(timeout or info.get("timeout", False) or done_reason == "timeout")
        collision = bool(collision or info.get("collision", False) or done_reason == "collision_or_too_close")
        out_of_bounds = bool(out_of_bounds or info.get("out_of_bounds", False) or done_reason == "out_of_bounds")
        height_violation = bool(
            height_violation
            or info.get("height_violation_condition", False)
            or info.get("height_violation", False)
            or done_reason == "height_violation"
        )

        if group == "D":
            if front_obstacle_present and obstacle_danger:
                intervention_danger_seen = True
            if front_obstacle_present and is_safety_mode(mode):
                intervention_safety_triggered = True
            if (
                front_obstacle_present
                and (obstacle_danger or is_safety_mode(mode))
                and math.isfinite(filtered_vx)
                and filtered_vx <= 0.0
            ):
                intervention_filtered_vx_nonpositive = True
            if front_obstacle_present and step + 1 >= int(intervention_delete_after_steps):
                front_delete_success, _tail = delete_front_obstacle()
                front_obstacle_present = False
                front_obstacle_deleted = True
            if front_obstacle_deleted and str(info.get("mavros_mode", "")) == "OFFBOARD":
                if not is_safety_mode(mode) or obstacle_danger:
                    post_delete_safe_steps += 1
                if post_delete_safe_steps >= 5 or success:
                    intervention_recovered_or_safe = True

        rollout_rows.append(
            {
                "episode": int(local_episode),
                "global_episode": int(global_episode),
                "group": group,
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
                "raw_vx_body": snapshot["raw_vx_body"],
                "filtered_vx_body": snapshot["filtered_vx_body"],
                "published_vx_world": snapshot["published_vx_world"],
                "drone_z": finite(info.get("drone_z", math.nan)),
                "reward": float(reward),
                "done": bool(done),
                "done_reason": done_reason if done else "",
                "action_vy": finite(action_row[1]),
                "raw_vy_body": snapshot["raw_vy_body"],
                "raw_vz_body": snapshot["raw_vz_body"],
                "raw_yaw_rate": snapshot["raw_yaw_rate"],
                "filtered_vy_body": snapshot["filtered_vy_body"],
                "filtered_vz_body": snapshot["filtered_vz_body"],
                "filtered_yaw_rate": snapshot["filtered_yaw_rate"],
                "published_vy_world": snapshot["published_vy_world"],
                "published_vz_world": snapshot["published_vz_world"],
                "published_yaw_rate": snapshot["published_yaw_rate"],
                "mavros_mode": str(info.get("mavros_mode", "")),
                "mavros_armed": bool(info.get("mavros_armed", False)),
                "safety_filtered": bool(safety_filtered),
                "filtered_vx_limited": bool(filtered_vx_limited),
                "front_obstacle_present": bool(front_obstacle_present),
                "front_obstacle_deleted": bool(front_obstacle_deleted),
            }
        )
        if is_safety_mode(mode) or obstacle_danger or filtered_vx_limited:
            safety_events.append(
                {
                    "group": group,
                    "episode": int(local_episode),
                    "global_episode": int(global_episode),
                    "seed": int(seed),
                    "step": int(step),
                    "event_type": "+".join(
                        token
                        for token, active in (
                            ("safety_mode", is_safety_mode(mode)),
                            ("obstacle_danger", obstacle_danger),
                            ("filtered_vx_limited", filtered_vx_limited),
                        )
                        if active
                    ),
                    "safety_mode": mode,
                    "obstacle_danger": bool(obstacle_danger),
                    "front_q05_depth": front_q05_depth,
                    "obstacle_area_ratio": obstacle_area_ratio,
                    "raw_vx_body": snapshot["raw_vx_body"],
                    "filtered_vx_body": snapshot["filtered_vx_body"],
                    "drone_z": finite(info.get("drone_z", math.nan)),
                }
            )

        step += 1
        if done:
            break

    if group == "D" and front_obstacle_present:
        front_delete_success, _tail = delete_front_obstacle()
        front_obstacle_present = False
        front_obstacle_deleted = True

    episode_length = int(step)
    emergency_failure = bool((not success) and (collision or emergency_count > 0 or depth_stop_count > 0))
    episode_row = {
        "group": group,
        "episode": int(local_episode),
        "global_episode": int(global_episode),
        "seed": int(seed),
        "success": bool(success),
        "timeout": bool(timeout),
        "collision": bool(collision),
        "emergency_failure": bool(emergency_failure),
        "out_of_bounds": bool(out_of_bounds),
        "height_violation": bool(height_violation),
        "done_reason": done_reason or "unknown",
        "final_distance": final_distance,
        "min_distance": min_distance,
        "episode_length": episode_length,
        "target_visible_ratio": rate(visible_count, episode_length),
        "emergency_count": int(emergency_count),
        "depth_stop_count": int(depth_stop_count),
        "raw_timeout_count": int(raw_timeout_count),
        "front_q05_min": min(front_depths) if front_depths else math.nan,
        "obstacle_danger_steps": int(obstacle_danger_steps),
        "offboard_drop": int(offboard_drop_count),
        "reset_recovery_used": bool(int(reset_row.get("attempt", 0)) > 0),
        "reset_pollution_detected": bool(
            (not reset_row.get("reset_success", False))
            or reset_row.get("accepted_with_warning", False)
            or (not reset_row.get("target_respawn_success", True))
        ),
        "reward": float(reward_total),
        "target_lateral": finite(reset_row.get("target_lateral", 0.0)),
        "front_obstacle_spawn_success": bool(front_spawn_success) if group == "D" else None,
        "front_obstacle_delete_success": bool(front_delete_success) if group == "D" else None,
        "intervention_danger_seen": bool(intervention_danger_seen) if group == "D" else None,
        "intervention_safety_triggered": bool(intervention_safety_triggered) if group == "D" else None,
        "intervention_filtered_vx_nonpositive": (
            bool(intervention_filtered_vx_nonpositive) if group == "D" else None
        ),
        "intervention_recovered_or_safe": bool(
            intervention_recovered_or_safe or success or (group == "D" and not collision and offboard_drop_count == 0)
        )
        if group == "D"
        else None,
    }
    return episode_row, rollout_rows, safety_events, reset_attempts


def summarize_rows(rows):
    total = len(rows)
    success_count = sum(1 for row in rows if row["success"])
    timeout_count = sum(1 for row in rows if row["timeout"])
    collision_count = sum(1 for row in rows if row["collision"])
    emergency_failure_count = sum(1 for row in rows if row["emergency_failure"])
    out_of_bounds_count = sum(1 for row in rows if row["out_of_bounds"])
    height_violation_count = sum(1 for row in rows if row["height_violation"])
    offboard_drop_count = sum(int(row["offboard_drop"]) for row in rows)
    raw_timeout_count = sum(int(row["raw_timeout_count"]) for row in rows)
    return {
        "episodes": int(total),
        "success_count": int(success_count),
        "success_rate": rate(success_count, total),
        "timeout_count": int(timeout_count),
        "timeout_rate": rate(timeout_count, total),
        "collision_count": int(collision_count),
        "collision_rate": rate(collision_count, total),
        "emergency_failure_count": int(emergency_failure_count),
        "collision_emergency_failure_count": int(
            sum(1 for row in rows if (not row["success"]) and (row["collision"] or row["emergency_failure"]))
        ),
        "collision_emergency_failure_rate": rate(
            sum(1 for row in rows if (not row["success"]) and (row["collision"] or row["emergency_failure"])),
            total,
        ),
        "out_of_bounds_count": int(out_of_bounds_count),
        "out_of_bounds_rate": rate(out_of_bounds_count, total),
        "height_violation_count": int(height_violation_count),
        "height_violation_rate": rate(height_violation_count, total),
        "target_visible_ratio_mean": mean(row["target_visible_ratio"] for row in rows),
        "final_distance_mean": mean(row["final_distance"] for row in rows),
        "min_distance_mean": mean(row["min_distance"] for row in rows),
        "mean_episode_length": mean(row["episode_length"] for row in rows),
        "raw_timeout_count": int(raw_timeout_count),
        "emergency_count": int(sum(int(row["emergency_count"]) for row in rows)),
        "depth_stop_count": int(sum(int(row["depth_stop_count"]) for row in rows)),
        "front_q05_min_mean": mean(row["front_q05_min"] for row in rows),
        "obstacle_danger_steps_mean": mean(row["obstacle_danger_steps"] for row in rows),
        "offboard_drop_count": int(offboard_drop_count),
        "reset_pollution_detected": any(bool(row["reset_pollution_detected"]) for row in rows),
        "termination_reason_distribution": dict(
            sorted(Counter(str(row["done_reason"]) for row in rows).items())
        ),
    }


def classify_failure_modes(rows):
    failures = [row for row in rows if not row["success"]]
    if not failures:
        return "none"
    counts = {
        "timeout": sum(1 for row in failures if row["timeout"]),
        "collision_or_too_close": sum(1 for row in failures if row["collision"]),
        "emergency_failure": sum(1 for row in failures if row["emergency_failure"]),
        "out_of_bounds": sum(1 for row in failures if row["out_of_bounds"]),
        "height_violation": sum(1 for row in failures if row["height_violation"]),
        "low_visibility": sum(1 for row in failures if row["target_visible_ratio"] < 0.8),
        "reset_pollution": sum(1 for row in failures if row["reset_pollution_detected"]),
    }
    active = ["{}={}".format(key, value) for key, value in sorted(counts.items()) if value > 0]
    return "; ".join(active) if active else "unknown"


def failure_category(row):
    if row["success"]:
        return ""
    if row["reset_pollution_detected"]:
        return "reset_pollution"
    if row["collision"]:
        return "collision_or_too_close"
    if row["emergency_failure"]:
        return "emergency_failure"
    if row["out_of_bounds"]:
        return "out_of_bounds"
    if row["height_violation"]:
        return "height_violation"
    if row["timeout"] and row["target_visible_ratio"] < 0.8:
        return "timeout_target_lost_or_occluded"
    if row["timeout"] and (row["emergency_count"] > 0 or row["depth_stop_count"] > 0):
        return "timeout_safety_filtered"
    if row["timeout"]:
        return "timeout_control_or_generalization"
    if row["target_visible_ratio"] < 0.8:
        return "target_lost_or_low_visibility"
    return "unknown"


def build_failure_rows(rows):
    failure_rows = []
    for row in rows:
        if row["success"]:
            continue
        failure_row = dict(row)
        failure_row["failure_category"] = failure_category(row)
        failure_rows.append(failure_row)
    return failure_rows


def build_summary(args, model_sha, vec_sha, episode_rows, rollout_rows, reset_rows, safety_events):
    groups = {spec["group"]: [row for row in episode_rows if row["group"] == spec["group"]] for spec in GROUP_SPECS}
    group_summaries = {group: summarize_rows(rows) for group, rows in groups.items()}
    total_summary = summarize_rows(episode_rows)
    d_rows = groups["D"]
    d_danger_seen = bool(d_rows and any(bool(row["intervention_danger_seen"]) for row in d_rows))
    d_safety_triggered = bool(d_rows and any(bool(row["intervention_safety_triggered"]) for row in d_rows))
    d_filtered_stop = bool(d_rows and any(bool(row["intervention_filtered_vx_nonpositive"]) for row in d_rows))
    d_no_collision = bool(d_rows and all(not row["collision"] for row in d_rows))
    d_no_offboard = bool(d_rows and all(int(row["offboard_drop"]) == 0 for row in d_rows))
    d_recovered_or_safe = bool(d_rows and any(bool(row["intervention_recovered_or_safe"]) for row in d_rows))
    group_d_passed = bool(
        d_danger_seen and d_safety_triggered and d_filtered_stop and d_no_collision and d_no_offboard and d_recovered_or_safe
    )
    offboard_frequent_drop = bool(total_summary["offboard_drop_count"] > max(1, int(0.01 * max(1, len(rollout_rows)))))
    reset_pollution_detected = bool(
        total_summary["reset_pollution_detected"]
        or any(
            row.get("accepted_for_episode", False)
            and (
                not row.get("reset_success", False)
                or row.get("accepted_with_warning", False)
                or not row.get("target_respawn_success", True)
            )
            for row in reset_rows
        )
    )
    phase_passed = bool(
        total_summary["episodes"] >= 90
        and total_summary["success_rate"] >= 0.80
        and group_summaries["A"]["success_rate"] >= 0.90
        and group_summaries["B"]["success_rate"] >= 0.80
        and group_summaries["C"]["success_rate"] >= 0.70
        and group_d_passed
        and total_summary["raw_timeout_count"] == 0
        and total_summary["out_of_bounds_count"] == 0
        and total_summary["height_violation_count"] == 0
        and total_summary["collision_emergency_failure_rate"] <= 0.10
        and not offboard_frequent_drop
        and not reset_pollution_detected
    )
    diagnostic_summary = {
        "mean_target_visible_ratio": total_summary["target_visible_ratio_mean"],
        "target_lost_episode_count": sum(1 for row in episode_rows if row["target_visible_ratio"] < 0.8),
        "emergency_count": total_summary["emergency_count"],
        "depth_stop_count": total_summary["depth_stop_count"],
        "success_rate": total_summary["success_rate"],
        "collision_rate": total_summary["collision_rate"],
        "collision_emergency_failure_rate": total_summary["collision_emergency_failure_rate"],
        "termination_reason_distribution": total_summary["termination_reason_distribution"],
        "obstacle_danger_steps_mean": total_summary["obstacle_danger_steps_mean"],
        "mean_safety_filtered_count": mean(
            row["emergency_count"] + row["depth_stop_count"] for row in episode_rows
        ),
        "mean_min_distance": total_summary["min_distance_mean"],
    }
    return {
        "phase": "7.4B",
        "report_date": datetime.now().isoformat(timespec="seconds"),
        "evaluated_model": os.path.abspath(args.model),
        "evaluated_vecnormalize": os.path.abspath(args.vecnormalize),
        "manifest": os.path.abspath(args.manifest),
        "model_sha256": model_sha,
        "vecnormalize_sha256": vec_sha,
        "world": args.world,
        "deterministic": bool(args.deterministic),
        "seed_base": int(args.seed_base),
        "group_specs": GROUP_SPECS,
        "total_episodes": total_summary["episodes"],
        "total_success_count": total_summary["success_count"],
        "total_success_rate": total_summary["success_rate"],
        "group_summaries": group_summaries,
        "group_success_rates": {group: summary["success_rate"] for group, summary in group_summaries.items()},
        "group_d_danger_seen": d_danger_seen,
        "group_d_safety_triggered": d_safety_triggered,
        "group_d_filtered_vx_nonpositive": d_filtered_stop,
        "group_d_no_collision": d_no_collision,
        "group_d_no_offboard": d_no_offboard,
        "group_d_recovered_or_safe": d_recovered_or_safe,
        "group_d_safety_intervention_passed": group_d_passed,
        "timeout_rate": total_summary["timeout_rate"],
        "collision_emergency_failure_count": total_summary["collision_emergency_failure_count"],
        "collision_emergency_failure_rate": total_summary["collision_emergency_failure_rate"],
        "out_of_bounds_rate": total_summary["out_of_bounds_rate"],
        "height_violation_rate": total_summary["height_violation_rate"],
        "target_visible_ratio_mean": total_summary["target_visible_ratio_mean"],
        "final_distance_mean": total_summary["final_distance_mean"],
        "min_distance_mean": total_summary["min_distance_mean"],
        "mean_episode_length": total_summary["mean_episode_length"],
        "raw_timeout_count": total_summary["raw_timeout_count"],
        "emergency_count": total_summary["emergency_count"],
        "depth_stop_count": total_summary["depth_stop_count"],
        "front_q05_min_mean": total_summary["front_q05_min_mean"],
        "obstacle_danger_steps_mean": total_summary["obstacle_danger_steps_mean"],
        "offboard_drop_count": total_summary["offboard_drop_count"],
        "offboard_frequent_drop": offboard_frequent_drop,
        "reset_pollution_detected": reset_pollution_detected,
        "main_failure_modes": classify_failure_modes(episode_rows),
        "diagnostic_answers": build_diagnostic_answers(diagnostic_summary),
        "phase7_4b_passed": phase_passed,
        "phase7_4c_allowed": phase_passed,
        "random_woods_claim_allowed": phase_passed,
        "all_woods_claim_allowed": False,
        "episode_rows": episode_rows,
        "reset_rows": reset_rows,
        "safety_event_count": len(safety_events),
    }


def write_report(path, summary):
    d_result = (
        "danger_seen={} safety_triggered={} filtered_vx_body<=0={} no_collision={} "
        "no_offboard={} recovered_or_safe={} passed={}"
    ).format(
        "是" if summary["group_d_danger_seen"] else "否",
        "是" if summary["group_d_safety_triggered"] else "否",
        "是" if summary["group_d_filtered_vx_nonpositive"] else "否",
        "是" if summary["group_d_no_collision"] else "否",
        "是" if summary["group_d_no_offboard"] else "否",
        "是" if summary["group_d_recovered_or_safe"] else "否",
        "是" if summary["group_d_safety_intervention_passed"] else "否",
    )
    lines = [
        "# Phase 7.4B Random Woods Robustness / Stress Validation",
        "",
        "1. evaluated model：{}".format(summary["evaluated_model"]),
        "2. evaluated vecnormalize：{}".format(summary["evaluated_vecnormalize"]),
        "3. world：{}".format(summary["world"]),
        "4. total episodes：{}".format(summary["total_episodes"]),
        "5. Group A success rate：{:.4f}".format(summary["group_success_rates"].get("A", 0.0)),
        "6. Group B success rate：{:.4f}".format(summary["group_success_rates"].get("B", 0.0)),
        "7. Group C success rate：{:.4f}".format(summary["group_success_rates"].get("C", 0.0)),
        "8. Group D safety intervention result：{}".format(d_result),
        "9. total success rate：{:.4f} ({}/{})".format(
            summary["total_success_rate"], summary["total_success_count"], summary["total_episodes"]
        ),
        "10. timeout rate：{:.4f}".format(summary["timeout_rate"]),
        "11. collision/emergency failure rate：{:.4f} ({}/{})".format(
            summary["collision_emergency_failure_rate"],
            summary["collision_emergency_failure_count"],
            summary["total_episodes"],
        ),
        "12. out_of_bounds rate：{:.4f}".format(summary["out_of_bounds_rate"]),
        "13. height_violation rate：{:.4f}".format(summary["height_violation_rate"]),
        "14. target_visible_ratio_mean：{:.4f}".format(summary["target_visible_ratio_mean"]),
        "15. final_distance_mean：{:.4f} m".format(summary["final_distance_mean"]),
        "16. min_distance_mean：{:.4f} m".format(summary["min_distance_mean"]),
        "17. RAW_TIMEOUT count：{}".format(summary["raw_timeout_count"]),
        "18. emergency_count：{}".format(summary["emergency_count"]),
        "19. depth_stop_count：{}".format(summary["depth_stop_count"]),
        "20. front_q05_min_mean：{:.4f} m".format(summary["front_q05_min_mean"]),
        "21. obstacle_danger_steps_mean：{:.4f}".format(summary["obstacle_danger_steps_mean"]),
        "22. OFFBOARD drop count：{}".format(summary["offboard_drop_count"]),
        "23. reset pollution detected：{}".format("是" if summary["reset_pollution_detected"] else "否"),
        "24. main failure modes：{}".format(summary["main_failure_modes"]),
        "25. 是否通过 Phase 7.4B：{}".format("是" if summary["phase7_4b_passed"] else "否"),
        "26. 是否允许进入 Phase 7.4C：{}".format("是" if summary["phase7_4c_allowed"] else "否"),
        "27. 是否允许声称 random_woods 已通过：{}".format(
            "是" if summary["random_woods_claim_allowed"] else "否"
        ),
        "28. 是否允许声称全部 woods 已通过：{}".format(
            "是" if summary["all_woods_claim_allowed"] else "否"
        ),
        "",
        "Additional detail:",
        "- mean episode length：{:.4f}".format(summary["mean_episode_length"]),
        "",
        "Diagnostics:" if summary["phase7_4b_passed"] else "Failure diagnosis:",
    ]
    lines.extend("- {}".format(item) for item in summary["diagnostic_answers"])
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")


def write_reproduce_commands(path, args):
    content = """# Phase 7.4B Random Woods Robustness Reproduce Commands

## Runtime

Terminal 1:

```bash
source /opt/ros/noetic/setup.bash
source /home/whk/vf_ws/devel/setup.bash
source /home/whk/vf_ws/src/drone_chase/scripts/source_px4_gazebo_env.sh
roslaunch drone_chase phase2_chase_world.launch world:=random_woods gui:=false interactive:=false
```

Terminal 2:

```bash
source /opt/ros/noetic/setup.bash
source /home/whk/vf_ws/devel/setup.bash
roslaunch drone_chase phase3_perception.launch debug:=false
```

Terminal 3:

```bash
source /opt/ros/noetic/setup.bash
source /home/whk/vf_ws/devel/setup.bash
roslaunch drone_chase phase6_env_runtime.launch
```

Terminal 4:

```bash
source /opt/ros/noetic/setup.bash
source /home/whk/vf_ws/devel/setup.bash
rosrun drone_chase phase7_random_woods_robustness_eval.py \\
  --model {model} \\
  --vecnormalize {vecnormalize} \\
  --world {world} \\
  --deterministic \\
  --output-dir {output_dir}
```
""".format(
        model=args.model,
        vecnormalize=args.vecnormalize,
        world=args.world,
        output_dir=args.output_dir,
    )
    with open(path, "w") as handle:
        handle.write(content)


def output_paths(args):
    return {
        "rollout_csv": os.path.join(args.output_dir, "random_woods_robustness_rollouts.csv"),
        "episode_csv": os.path.join(args.output_dir, "random_woods_robustness_episodes.csv"),
        "safety_csv": os.path.join(args.output_dir, "safety_events.csv"),
        "failure_csv": os.path.join(args.output_dir, "failure_cases.csv"),
        "summary_json": os.path.join(args.output_dir, "random_woods_robustness_summary.json"),
        "report_md": os.path.join(args.output_dir, "phase7_4b_report.md"),
        "reproduce_md": os.path.join(args.output_dir, "reproduce_commands.md"),
    }


def _optional_bool(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return bool(value)
    text = str(value).strip().lower()
    if text == "":
        return None
    if text in ("1", "true", "yes", "y", "是"):
        return True
    if text in ("0", "false", "no", "n", "否"):
        return False
    return bool(text)


def _as_int_value(value, default=0):
    if value is None or str(value).strip() == "":
        return int(default)
    return int(float(value))


def _as_float_value(value, default=math.nan):
    if value is None or str(value).strip() == "":
        return float(default)
    return float(value)


def load_episode_rows_from_csv(path):
    bool_fields = {
        "success",
        "timeout",
        "collision",
        "emergency_failure",
        "out_of_bounds",
        "height_violation",
        "reset_recovery_used",
        "reset_pollution_detected",
        "front_obstacle_spawn_success",
        "front_obstacle_delete_success",
        "intervention_danger_seen",
        "intervention_safety_triggered",
        "intervention_filtered_vx_nonpositive",
        "intervention_recovered_or_safe",
    }
    int_fields = {
        "episode",
        "global_episode",
        "seed",
        "episode_length",
        "emergency_count",
        "depth_stop_count",
        "raw_timeout_count",
        "obstacle_danger_steps",
        "offboard_drop",
    }
    float_fields = {
        "final_distance",
        "min_distance",
        "target_visible_ratio",
        "front_q05_min",
        "reward",
        "target_lateral",
    }
    rows = []
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            row = {}
            for key in episode_fieldnames():
                value = raw.get(key, "")
                if key in bool_fields:
                    row[key] = _optional_bool(value)
                elif key in int_fields:
                    row[key] = _as_int_value(value)
                elif key in float_fields:
                    row[key] = _as_float_value(value)
                else:
                    row[key] = value
            # Collision is an episode outcome. A transient collision_condition in the
            # initial forced-obstacle window is not a terminal collision.
            row["collision"] = bool(str(row.get("done_reason", "")) == "collision_or_too_close")
            rows.append(row)
    return rows


def count_csv_data_rows(path):
    if not os.path.exists(path):
        return 0
    with open(path, newline="") as handle:
        return max(0, sum(1 for _line in handle) - 1)


def write_episode_csv(path, episode_rows):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=episode_fieldnames())
        writer.writeheader()
        writer.writerows(episode_rows)


def summarize_existing_outputs(args):
    paths = output_paths(args)
    if not os.path.exists(paths["episode_csv"]):
        raise FileNotFoundError(paths["episode_csv"])
    if not os.path.exists(paths["summary_json"]):
        raise FileNotFoundError(paths["summary_json"])

    _manifest, model_sha, vec_sha = verify_registry_policy(args.model, args.vecnormalize, args.manifest)
    with open(paths["summary_json"]) as handle:
        previous_summary = json.load(handle)
    episode_rows = load_episode_rows_from_csv(paths["episode_csv"])
    rollout_count = count_csv_data_rows(paths["rollout_csv"])
    safety_count = count_csv_data_rows(paths["safety_csv"])
    reset_rows = previous_summary.get("reset_rows", [])
    summary = build_summary(
        args,
        model_sha,
        vec_sha,
        episode_rows,
        [None] * int(rollout_count),
        reset_rows,
        [None] * int(safety_count),
    )

    write_episode_csv(paths["episode_csv"], episode_rows)
    with open(paths["failure_csv"], "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=episode_fieldnames() + ["failure_category"])
        writer.writeheader()
        writer.writerows(build_failure_rows(episode_rows))
    with open(paths["summary_json"], "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    write_report(paths["report_md"], summary)
    write_reproduce_commands(paths["reproduce_md"], args)
    print("rewrote {}".format(paths["episode_csv"]))
    print("rewrote {}".format(paths["failure_csv"]))
    print("rewrote {}".format(paths["summary_json"]))
    print("rewrote {}".format(paths["report_md"]))
    print("phase7_4b_passed={}".format(summary["phase7_4b_passed"]))


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Phase 7.4B random_woods robustness/stress eval.")
    parser.add_argument("--model", default=REGISTERED_MODEL)
    parser.add_argument("--vecnormalize", default=REGISTERED_VECNORMALIZE)
    parser.add_argument("--manifest", default=MANIFEST_PATH)
    parser.add_argument("--world", default="random_woods", choices=["random_woods"])
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed-base", type=int, default=74200)
    parser.add_argument("--reset-retries", type=int, default=2)
    parser.add_argument("--front-face-distance", type=float, default=0.45)
    parser.add_argument("--intervention-delete-after-steps", type=int, default=12)
    parser.add_argument("--deterministic", dest="deterministic", action="store_true", default=True)
    parser.add_argument("--stochastic", dest="deterministic", action="store_false")
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="Rebuild summary/report from existing CSV artifacts without running simulation.",
    )
    return parser


def main():
    args = build_arg_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    paths = output_paths(args)
    if args.summarize_only:
        summarize_existing_outputs(args)
        return

    require_eval_deps()

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from gazebo_chase_env import GazeboChaseEnv

    _manifest, model_sha, vec_sha = verify_registry_policy(args.model, args.vecnormalize, args.manifest)
    config = load_yaml(args.config)
    env_cfg = base_env_kwargs(config, args.seed_base, args.world)

    raw_env = GazeboChaseEnv(**env_cfg)
    vec_env = DummyVecEnv([lambda: raw_env])
    vec_env = VecNormalize.load(args.vecnormalize, vec_env)
    vec_env.training = False
    vec_env.norm_reward = False
    model = PPO.load(args.model)
    monitor = CommandMonitor()

    rollout_rows = []
    episode_rows = []
    reset_rows = []
    safety_events = []
    global_episode = 0
    try:
        for spec in GROUP_SPECS:
            for local_episode in range(int(spec["episodes"])):
                seed = int(args.seed_base) + int(global_episode)
                episode_row, step_rows, event_rows, reset_attempts = run_episode(
                    raw_env,
                    vec_env,
                    model,
                    monitor,
                    env_cfg,
                    spec,
                    local_episode,
                    global_episode,
                    seed,
                    args.deterministic,
                    args.reset_retries,
                    args.front_face_distance,
                    args.intervention_delete_after_steps,
                )
                rollout_rows.extend(step_rows)
                episode_rows.append(episode_row)
                safety_events.extend(event_rows)
                reset_rows.extend(reset_attempts)
                print(
                    "group={} episode={} seed={} success={} reason={} final={:.3f} min={:.3f} "
                    "visible={:.3f} danger_steps={} emergency={} depth_stop={} offboard_drop={}".format(
                        spec["group"],
                        local_episode,
                        seed,
                        episode_row["success"],
                        episode_row["done_reason"],
                        episode_row["final_distance"],
                        episode_row["min_distance"],
                        episode_row["target_visible_ratio"],
                        episode_row["obstacle_danger_steps"],
                        episode_row["emergency_count"],
                        episode_row["depth_stop_count"],
                        episode_row["offboard_drop"],
                    )
                )
                global_episode += 1
    finally:
        try:
            delete_front_obstacle()
        except Exception:
            pass
        monitor.close()
        vec_env.close()

    summary = build_summary(args, model_sha, vec_sha, episode_rows, rollout_rows, reset_rows, safety_events)
    with open(paths["rollout_csv"], "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rollout_fieldnames())
        writer.writeheader()
        writer.writerows(rollout_rows)
    write_episode_csv(paths["episode_csv"], episode_rows)
    with open(paths["safety_csv"], "w", newline="") as handle:
        fieldnames = [
            "group",
            "episode",
            "global_episode",
            "seed",
            "step",
            "event_type",
            "safety_mode",
            "obstacle_danger",
            "front_q05_depth",
            "obstacle_area_ratio",
            "raw_vx_body",
            "filtered_vx_body",
            "drone_z",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(safety_events)
    with open(paths["failure_csv"], "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=episode_fieldnames() + ["failure_category"])
        writer.writeheader()
        writer.writerows(build_failure_rows(episode_rows))
    with open(paths["summary_json"], "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    write_report(paths["report_md"], summary)
    write_reproduce_commands(paths["reproduce_md"], args)

    print("wrote {}".format(paths["rollout_csv"]))
    print("wrote {}".format(paths["episode_csv"]))
    print("wrote {}".format(paths["summary_json"]))
    print("wrote {}".format(paths["report_md"]))
    print("wrote {}".format(paths["safety_csv"]))
    print("wrote {}".format(paths["failure_csv"]))
    print("wrote {}".format(paths["reproduce_md"]))
    print("phase7_4b_passed={}".format(summary["phase7_4b_passed"]))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("phase7_random_woods_robustness_eval failed: {}".format(exc), file=sys.stderr)
        sys.exit(1)
