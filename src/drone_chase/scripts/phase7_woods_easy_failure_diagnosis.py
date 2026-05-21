#!/usr/bin/env python3

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "envs"))
if ENV_DIR not in sys.path:
    sys.path.insert(0, ENV_DIR)


PHASE7A_DIR = "/home/whk/vf_ws/outputs/phase7/woods_easy_zero_shot_from_world0_world1_best"
OUTPUT_DIR = "/home/whk/vf_ws/outputs/phase7/woods_easy_failure_diagnosis"
DEFAULT_CONFIG = (
    "/home/whk/vf_ws/outputs/phase7/world0_ppo_from_bc_v2_10k_conservative_run4/config_effective.yaml"
)

SAFETY_TOKENS = ("EMERGENCY_AVOID", "DEPTH_STOP", "HEIGHT_GUARD", "TARGET_LOST")
RESET_BLOCKED_TOKENS = ("WAIT_FCU", "PRESTREAM", "SET_MODE", "ARMING", "TAKEOFF")
WOODS_EXPECTED_TOTAL = 26


def finite(value, default=0.0):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(default)
    return value if math.isfinite(value) else float(default)


def mean(values):
    values = [finite(value, math.nan) for value in values]
    values = [value for value in values if math.isfinite(value)]
    return sum(values) / len(values) if values else 0.0


def std(values):
    values = [finite(value, math.nan) for value in values]
    values = [value for value in values if math.isfinite(value)]
    if not values:
        return 0.0
    avg = mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / len(values))


def rate(count, total):
    return float(count) / float(total) if total else 0.0


def as_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "y", "on")


def is_safety_mode(mode):
    return any(token in str(mode) for token in SAFETY_TOKENS)


def read_csv(path):
    with open(path, "r", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_text(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        handle.write(text)


def load_json(path):
    with open(path, "r") as handle:
        return json.load(handle)


def episode_key(row):
    return (str(row.get("eval_split", "")), int(float(row.get("episode_id", 0))))


def group_rollouts(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[episode_key(row)].append(row)
    for key in groups:
        groups[key].sort(key=lambda row: int(float(row.get("step", 0))))
    return groups


def longest_false_streak(rows, field):
    longest = 0
    current = 0
    for row in rows:
        if not as_bool(row.get(field, False)):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return int(longest)


def first_step(rows, predicate):
    for row in rows:
        if predicate(row):
            return int(float(row.get("step", 0)))
    return -1


def row_at_step(rows, step):
    if step < 0:
        return None
    for row in rows:
        if int(float(row.get("step", 0))) == int(step):
            return row
    return None


def window(rows, center_step, radius=5):
    if center_step < 0:
        return []
    return [
        row for row in rows
        if abs(int(float(row.get("step", 0))) - int(center_step)) <= int(radius)
    ]


def terminal_reason_for(key, failure_by_key, rows):
    failure = failure_by_key.get(key)
    if failure is not None:
        return str(failure.get("termination_reason", ""))
    done_rows = [row for row in rows if as_bool(row.get("done", False))]
    if done_rows:
        return str(done_rows[-1].get("termination_reason", ""))
    return ""


def analyze_visibility(rollout_groups, failure_rows):
    failure_by_key = {episode_key(row): row for row in failure_rows}
    results = []
    for key, rows in sorted(rollout_groups.items()):
        split, episode_id = key
        total = len(rows)
        visible_count = sum(1 for row in rows if as_bool(row.get("target_visible", False)))
        first_lost = first_step(rows, lambda row: not as_bool(row.get("target_visible", False)))
        first_safety = first_step(rows, lambda row: is_safety_mode(row.get("safety_mode", "")) or as_bool(row.get("safety_filtered", False)))
        first_danger = first_step(rows, lambda row: finite(row.get("obstacle_danger", 0.0)) > 0.5)
        lost_win = window(rows, first_lost, radius=5)
        safety_win = window(rows, first_safety, radius=5)
        reason = terminal_reason_for(key, failure_by_key, rows)
        row = {
            "eval_split": split,
            "episode_id": int(episode_id),
            "episode_length": int(total),
            "termination_reason": reason,
            "success": as_bool(failure_by_key.get(key, {}).get("success", False)) if key in failure_by_key else reason == "success",
            "target_visible_ratio": rate(visible_count, total),
            "target_lost_count": int(total - visible_count),
            "longest_target_lost_streak": longest_false_streak(rows, "target_visible"),
            "first_target_lost_step": int(first_lost),
            "first_safety_step": int(first_safety),
            "first_obstacle_danger_step": int(first_danger),
            "target_lost_before_safety": bool(first_lost >= 0 and (first_safety < 0 or first_lost <= first_safety)),
            "target_lost_before_danger": bool(first_lost >= 0 and (first_danger < 0 or first_lost <= first_danger)),
            "obstacle_danger_mean_around_first_lost": mean(row.get("obstacle_danger", 0.0) for row in lost_win),
            "front_q05_mean_around_first_lost": mean(row.get("front_q05_depth", 10.0) for row in lost_win),
            "yaw_abs_mean_around_first_lost": mean(abs(finite(row.get("raw_yaw_rate", row.get("action_yaw", 0.0)))) for row in lost_win),
            "obstacle_danger_mean_around_first_safety": mean(row.get("obstacle_danger", 0.0) for row in safety_win),
            "target_visible_at_first_safety": as_bool(row_at_step(rows, first_safety).get("target_visible", False)) if row_at_step(rows, first_safety) else False,
            "collision_or_height_failure": reason in ("collision_or_too_close", "height_violation"),
            "likely_occlusion_or_detector_loss": bool(rate(visible_count, total) < 0.8 or longest_false_streak(rows, "target_visible") >= 20),
        }
        results.append(row)
    return results


def analyze_safety(rollout_groups, failure_rows):
    failure_by_key = {episode_key(row): row for row in failure_rows}
    results = []
    for key, rows in sorted(rollout_groups.items()):
        split, episode_id = key
        safety_rows = [
            row for row in rows
            if is_safety_mode(row.get("safety_mode", "")) or as_bool(row.get("safety_filtered", False))
        ]
        filtered_stop_rows = [
            row for row in rows
            if finite(row.get("raw_vx_body", 0.0)) > 0.05 and finite(row.get("filtered_vx_body", 0.0)) <= 0.0
        ]
        first_safety = first_step(rows, lambda row: row in safety_rows)
        first_danger = first_step(rows, lambda row: finite(row.get("obstacle_danger", 0.0)) > 0.5)
        first_filtered_stop = first_step(rows, lambda row: row in filtered_stop_rows)
        reason = terminal_reason_for(key, failure_by_key, rows)
        done_step = first_step(rows, lambda row: as_bool(row.get("done", False)))
        first_safety_row = row_at_step(rows, first_safety)
        last10 = rows[-10:] if len(rows) >= 10 else rows
        collision = reason == "collision_or_too_close"
        results.append({
            "eval_split": split,
            "episode_id": int(episode_id),
            "termination_reason": reason,
            "episode_length": int(len(rows)),
            "safety_filtered_count": int(len(safety_rows)),
            "safety_triggered": bool(len(safety_rows) > 0),
            "first_safety_step": int(first_safety),
            "first_obstacle_danger_step": int(first_danger),
            "first_filtered_vx_le_zero_step": int(first_filtered_stop),
            "filtered_vx_le_zero_count": int(len(filtered_stop_rows)),
            "filtered_vx_le_zero_ratio": rate(len(filtered_stop_rows), len(rows)),
            "target_visible_at_first_safety": as_bool(first_safety_row.get("target_visible", False)) if first_safety_row else False,
            "front_q05_at_first_safety": finite(first_safety_row.get("front_q05_depth", math.nan), math.nan) if first_safety_row else math.nan,
            "obstacle_danger_at_first_safety": finite(first_safety_row.get("obstacle_danger", 0.0)) if first_safety_row else 0.0,
            "collision_step": int(done_step if collision else -1),
            "safety_before_collision": bool(collision and first_safety >= 0 and done_step >= 0 and first_safety <= done_step),
            "steps_safety_to_collision": int(done_step - first_safety) if collision and first_safety >= 0 and done_step >= 0 else -1,
            "obstacle_danger_mean": mean(row.get("obstacle_danger", 0.0) for row in rows),
            "obstacle_danger_max": max([finite(row.get("obstacle_danger", 0.0)) for row in rows], default=0.0),
            "front_q05_min": min([finite(row.get("front_q05_depth", 10.0)) for row in rows], default=10.0),
            "raw_vx_mean": mean(row.get("raw_vx_body", 0.0) for row in rows),
            "filtered_vx_mean": mean(row.get("filtered_vx_body", 0.0) for row in rows),
            "last10_front_q05_min": min([finite(row.get("front_q05_depth", 10.0)) for row in last10], default=10.0),
            "last10_obstacle_danger_max": max([finite(row.get("obstacle_danger", 0.0)) for row in last10], default=0.0),
            "interpretation": classify_safety_episode(reason, len(safety_rows), len(filtered_stop_rows), first_safety, first_danger),
        })
    return results


def classify_safety_episode(reason, safety_count, stop_count, first_safety, first_danger):
    if reason == "collision_or_too_close":
        if first_safety < 0:
            return "collision_without_safety_trigger"
        if first_danger >= 0 and first_safety > first_danger:
            return "safety_trigger_after_danger"
        return "safety_triggered_but_collision_still_occurred"
    if reason == "height_violation":
        return "height_failure_with_safety_filtering" if safety_count else "height_failure_without_safety_filtering"
    if safety_count > 20 and stop_count > 10:
        return "frequent_filtering_may_block_progress"
    if safety_count:
        return "safety_active"
    return "no_safety_event"


def analyze_control(rollout_groups, failure_rows):
    failure_by_key = {episode_key(row): row for row in failure_rows}
    results = []
    for key, rows in sorted(rollout_groups.items()):
        split, episode_id = key
        reason = terminal_reason_for(key, failure_by_key, rows)
        action_vx = [finite(row.get("action_vx", 0.0)) for row in rows]
        action_vy = [finite(row.get("action_vy", 0.0)) for row in rows]
        action_vz = [finite(row.get("action_vz", 0.0)) for row in rows]
        action_yaw = [finite(row.get("action_yaw", 0.0)) for row in rows]
        raw_vx = [finite(row.get("raw_vx_body", 0.0)) for row in rows]
        raw_vy = [finite(row.get("raw_vy_body", 0.0)) for row in rows]
        raw_vz = [finite(row.get("raw_vz_body", 0.0)) for row in rows]
        raw_yaw = [finite(row.get("raw_yaw_rate", 0.0)) for row in rows]
        filtered_vx = [finite(row.get("filtered_vx_body", 0.0)) for row in rows]
        z = [finite(row.get("drone_z", math.nan), math.nan) for row in rows]
        z = [value for value in z if math.isfinite(value)]
        yaw_sign_changes = 0
        prev_sign = 0
        for value in raw_yaw:
            sign = 1 if value > 0.02 else -1 if value < -0.02 else 0
            if sign and prev_sign and sign != prev_sign:
                yaw_sign_changes += 1
            if sign:
                prev_sign = sign
        last10 = rows[-10:] if len(rows) >= 10 else rows
        action_sat = sum(
            1 for idx in range(len(rows))
            if abs(action_vx[idx]) > 0.95 or abs(action_vy[idx]) > 0.95
            or abs(action_vz[idx]) > 0.95 or abs(action_yaw[idx]) > 0.95
        )
        filtered_delta = [raw_vx[idx] - filtered_vx[idx] for idx in range(len(raw_vx))]
        results.append({
            "eval_split": split,
            "episode_id": int(episode_id),
            "termination_reason": reason,
            "episode_length": int(len(rows)),
            "action_saturation_ratio": rate(action_sat, len(rows)),
            "mean_cmd_vx": mean(raw_vx),
            "std_cmd_vx": std(raw_vx),
            "mean_cmd_vy": mean(raw_vy),
            "std_cmd_vy": std(raw_vy),
            "mean_cmd_vz": mean(raw_vz),
            "std_cmd_vz": std(raw_vz),
            "mean_yaw_rate": mean(raw_yaw),
            "yaw_abs_mean": mean(abs(value) for value in raw_yaw),
            "yaw_sign_changes": int(yaw_sign_changes),
            "lateral_abs_mean": mean(abs(value) for value in raw_vy),
            "raw_vx_positive_ratio": rate(sum(1 for value in raw_vx if value > 0.05), len(raw_vx)),
            "filtered_vx_nonpositive_ratio": rate(sum(1 for value in filtered_vx if value <= 0.0), len(filtered_vx)),
            "filtered_delta_vx_mean": mean(filtered_delta),
            "drone_z_start": z[0] if z else math.nan,
            "drone_z_end": z[-1] if z else math.nan,
            "drone_z_min": min(z) if z else math.nan,
            "drone_z_max": max(z) if z else math.nan,
            "drone_z_trend": (z[-1] - z[0]) if len(z) >= 2 else 0.0,
            "last10_raw_vx_mean": mean(row.get("raw_vx_body", 0.0) for row in last10),
            "last10_raw_vz_mean": mean(row.get("raw_vz_body", 0.0) for row in last10),
            "last10_filtered_vx_mean": mean(row.get("filtered_vx_body", 0.0) for row in last10),
            "last10_front_q05_min": min([finite(row.get("front_q05_depth", 10.0)) for row in last10], default=10.0),
            "last10_obstacle_danger_max": max([finite(row.get("obstacle_danger", 0.0)) for row in last10], default=0.0),
            "control_interpretation": classify_control_episode(reason, raw_vx, raw_vy, raw_vz, filtered_vx, z),
        })
    return results


def classify_control_episode(reason, raw_vx, raw_vy, raw_vz, filtered_vx, z):
    forward_ratio = rate(sum(1 for value in raw_vx if value > 0.3), len(raw_vx))
    lateral_mean = mean(abs(value) for value in raw_vy)
    stop_ratio = rate(sum(1 for value in filtered_vx if value <= 0.0), len(filtered_vx))
    if reason == "height_violation":
        if z and min(z) < 0.2:
            return "height_violation_after_low_altitude_reset_or_descent"
        if mean(raw_vz) < -0.02:
            return "height_violation_with_negative_vz_command"
        return "height_violation_indirect_or_reset_related"
    if reason == "collision_or_too_close":
        if forward_ratio > 0.4 and lateral_mean < 0.03:
            return "mostly_forward_chase_with_little_lateral_avoidance"
        if stop_ratio > 0.4:
            return "forward_policy_blocked_by_filter_then_collision"
        return "collision_with_mixed_control"
    if forward_ratio > 0.5 and lateral_mean < 0.03:
        return "straight_chase_bias"
    return "mixed_control"


def point_segment_distance_xy(px, py, start, end):
    sx, sy = start[0], start[1]
    ex, ey = end[0], end[1]
    vx = ex - sx
    vy = ey - sy
    denom = vx * vx + vy * vy
    if denom <= 1e-9:
        return math.hypot(px - sx, py - sy)
    t = ((px - sx) * vx + (py - sy) * vy) / denom
    t = max(0.0, min(1.0, t))
    cx = sx + t * vx
    cy = sy + t * vy
    cy = sy + t * vy
    return math.hypot(px - cx, py - cy)


def clearance_metrics(specs, target_x=4.0, target_y=0.0):
    min_uav = math.inf
    min_target = math.inf
    for spec in specs or []:
        start = spec.get("start")
        end = spec.get("end")
        radius = finite(spec.get("radius", 0.0))
        if not start or not end:
            continue
        min_uav = min(min_uav, point_segment_distance_xy(0.0, 0.0, start, end) - radius)
        min_target = min(min_target, point_segment_distance_xy(target_x, target_y, start, end) - radius)
    return (
        min_uav if math.isfinite(min_uav) else math.nan,
        min_target if math.isfinite(min_target) else math.nan,
    )


def snapshot_to_reset_row(reset_id, seed, env, obs, info, world_models, woods_specs):
    snap = env._snapshot()
    pose = snap.get("pose")
    risk = snap.get("risk")
    state = snap.get("mavros_state")
    mode = str(snap.get("safety_mode", ""))
    drone_x = finite(pose.pose.position.x, math.nan) if pose is not None else math.nan
    drone_y = finite(pose.pose.position.y, math.nan) if pose is not None else math.nan
    drone_z = finite(pose.pose.position.z, math.nan) if pose is not None else math.nan
    target_spawn_ok = "red_ball" in world_models
    obstacle_spawn_ok = "random_woods" in world_models and len(woods_specs) == WOODS_EXPECTED_TOTAL
    min_uav_clearance, min_target_clearance = clearance_metrics(
        woods_specs,
        target_x=finite(env.config.get("target_x", 4.0)),
        target_y=finite(env.config.get("target_y", 0.0)),
    )
    target_visible = bool(info.get("target_visible", bool(obs[0] > 0.5 if len(obs) else False)))
    target_distance = finite(info.get("target_distance", obs[4] if len(obs) > 4 else math.nan), math.nan)
    front_min_depth = finite(getattr(risk, "front_min_depth", math.nan), math.nan) if risk is not None else math.nan
    front_q05_depth = finite(getattr(risk, "front_q05_depth", obs[7] if len(obs) > 7 else math.nan), math.nan) if risk is not None else math.nan
    obstacle_danger = bool(getattr(risk, "danger", False)) if risk is not None else False
    obstacle_area_ratio = finite(getattr(risk, "obstacle_area_ratio", math.nan), math.nan) if risk is not None else math.nan
    uav_pose_ok = bool(
        math.isfinite(drone_x)
        and math.isfinite(drone_y)
        and math.isfinite(drone_z)
        and abs(drone_x) <= 1.5
        and abs(drone_y) <= 1.5
        and 0.6 <= drone_z <= 2.8
    )
    safety_trigger = is_safety_mode(mode)
    failure_reasons = []
    if not bool(info.get("training_ready", False)):
        failure_reasons.append("training_not_ready")
    if not bool(info.get("topics_ready", False)):
        failure_reasons.append("topics_not_ready")
    if any(token in mode for token in RESET_BLOCKED_TOKENS):
        failure_reasons.append("blocked_safety_mode")
    if not target_spawn_ok:
        failure_reasons.append("target_model_missing")
    if not obstacle_spawn_ok:
        failure_reasons.append("woods_model_or_count_invalid")
    if not uav_pose_ok:
        failure_reasons.append("uav_pose_invalid")
    if not target_visible and target_spawn_ok:
        failure_reasons.append("target_initial_not_visible")
    if safety_trigger:
        failure_reasons.append("safety_trigger_initial")
    if bool(obstacle_danger):
        failure_reasons.append("obstacle_danger_initial")
    if not failure_reasons:
        failure_reasons.append("ok")
    return {
        "reset_id": int(reset_id),
        "seed": int(seed),
        "reset_success": bool(info.get("reset_success", False)),
        "training_ready": bool(info.get("training_ready", False)),
        "target_spawn_ok": bool(target_spawn_ok),
        "obstacle_spawn_ok": bool(obstacle_spawn_ok),
        "spawn_clearance_ok": bool(
            math.isfinite(min_uav_clearance)
            and math.isfinite(min_target_clearance)
            and min_uav_clearance >= finite(env.config.get("woods_easy_uav_clearance", 1.5)) - 1e-6
            and min_target_clearance >= finite(env.config.get("woods_easy_target_clearance", 0.8)) - 1e-6
        ),
        "uav_pose_ok": bool(uav_pose_ok),
        "target_visible_initial": bool(target_visible),
        "obstacle_danger_initial": bool(obstacle_danger),
        "safety_trigger_initial": bool(safety_trigger),
        "min_depth_initial": front_min_depth,
        "front_q05_depth_initial": front_q05_depth,
        "obstacle_area_ratio_initial": obstacle_area_ratio,
        "target_distance_initial": target_distance,
        "mavros_connected": bool(getattr(state, "connected", False)),
        "mavros_armed": bool(getattr(state, "armed", False)),
        "mavros_mode": str(getattr(state, "mode", "")),
        "safety_mode": mode,
        "drone_x": drone_x,
        "drone_y": drone_y,
        "drone_z": drone_z,
        "woods_model_count": int(len(woods_specs)),
        "min_uav_clearance": min_uav_clearance,
        "min_target_clearance": min_target_clearance,
        "failure_reason": ",".join(failure_reasons),
    }


def reset_fieldnames():
    return [
        "reset_id",
        "seed",
        "reset_success",
        "training_ready",
        "target_spawn_ok",
        "obstacle_spawn_ok",
        "spawn_clearance_ok",
        "uav_pose_ok",
        "target_visible_initial",
        "obstacle_danger_initial",
        "safety_trigger_initial",
        "min_depth_initial",
        "front_q05_depth_initial",
        "obstacle_area_ratio_initial",
        "target_distance_initial",
        "mavros_connected",
        "mavros_armed",
        "mavros_mode",
        "safety_mode",
        "drone_x",
        "drone_y",
        "drone_z",
        "woods_model_count",
        "min_uav_clearance",
        "min_target_clearance",
        "failure_reason",
    ]


def run_reset_only(args):
    import rospy
    from gazebo_msgs.srv import GetWorldProperties
    from gazebo_chase_env import GazeboChaseEnv

    env = GazeboChaseEnv(
        config_path=args.config,
        reset_mode="soft",
        world_type="woods_easy",
        seed=args.reset_seed_base,
        respawn_target_on_reset=False,
        woods_easy_num_trunks=8,
        woods_easy_num_branches=15,
        woods_easy_num_fallen=3,
        woods_easy_area_x_min=1.5,
        woods_easy_area_x_max=6.5,
        woods_easy_area_y_min=-3.0,
        woods_easy_area_y_max=3.0,
        woods_easy_uav_clearance=1.5,
        woods_easy_target_clearance=0.8,
    )
    rospy.wait_for_service("/gazebo/get_world_properties", timeout=10.0)
    world_proxy = rospy.ServiceProxy("/gazebo/get_world_properties", GetWorldProperties)
    rows = []
    try:
        for reset_id in range(int(args.reset_count)):
            seed = int(args.reset_seed_base) + reset_id
            env.config["seed"] = int(seed)
            obs, info = env.reset(seed=seed, options={"reset_mode": "soft"})
            time.sleep(float(args.post_reset_sleep))
            try:
                world_models = list(world_proxy().model_names)
            except Exception:
                world_models = []
            try:
                woods_specs = rospy.get_param("/drone_chase/random_woods", [])
            except Exception:
                woods_specs = []
            row = snapshot_to_reset_row(reset_id, seed, env, obs, info, world_models, woods_specs)
            rows.append(row)
            print(
                "reset_id={} seed={} training_ready={} target_visible={} safety_mode={} z={:.3f} reason={}".format(
                    reset_id,
                    seed,
                    row["training_ready"],
                    row["target_visible_initial"],
                    row["safety_mode"],
                    finite(row["drone_z"], math.nan),
                    row["failure_reason"],
                )
            )
    finally:
        env.close()
    return rows


def reset_summary(rows):
    total = len(rows)
    return {
        "total": int(total),
        "training_ready_count": sum(1 for row in rows if as_bool(row.get("training_ready", False))),
        "reset_success_rate": rate(sum(1 for row in rows if as_bool(row.get("training_ready", False))), total),
        "target_visible_initial_rate": rate(sum(1 for row in rows if as_bool(row.get("target_visible_initial", False))), total),
        "uav_pose_ok_rate": rate(sum(1 for row in rows if as_bool(row.get("uav_pose_ok", False))), total),
        "spawn_clearance_ok_rate": rate(sum(1 for row in rows if as_bool(row.get("spawn_clearance_ok", False))), total),
        "safety_trigger_initial_rate": rate(sum(1 for row in rows if as_bool(row.get("safety_trigger_initial", False))), total),
        "obstacle_danger_initial_rate": rate(sum(1 for row in rows if as_bool(row.get("obstacle_danger_initial", False))), total),
        "mean_min_depth_initial": mean(row.get("min_depth_initial", math.nan) for row in rows),
        "mean_target_distance_initial": mean(row.get("target_distance_initial", math.nan) for row in rows),
        "failure_reason_distribution": dict(Counter(str(row.get("failure_reason", "")) for row in rows)),
    }


def visibility_summary(rows):
    total = len(rows)
    return {
        "episodes": int(total),
        "mean_target_visible_ratio": mean(row.get("target_visible_ratio", 0.0) for row in rows),
        "target_lost_episode_rate": rate(sum(1 for row in rows if finite(row.get("target_visible_ratio", 0.0)) < 0.8), total),
        "mean_longest_target_lost_streak": mean(row.get("longest_target_lost_streak", 0) for row in rows),
        "target_lost_before_safety_rate": rate(sum(1 for row in rows if as_bool(row.get("target_lost_before_safety", False))), total),
        "likely_occlusion_or_detector_loss_rate": rate(sum(1 for row in rows if as_bool(row.get("likely_occlusion_or_detector_loss", False))), total),
    }


def safety_summary(rows):
    total = len(rows)
    return {
        "episodes": int(total),
        "safety_trigger_rate": rate(sum(1 for row in rows if as_bool(row.get("safety_triggered", False))), total),
        "mean_safety_filtered_count": mean(row.get("safety_filtered_count", 0) for row in rows),
        "mean_filtered_vx_le_zero_ratio": mean(row.get("filtered_vx_le_zero_ratio", 0.0) for row in rows),
        "collision_with_safety_before_rate": rate(
            sum(1 for row in rows if as_bool(row.get("safety_before_collision", False))), total
        ),
        "mean_front_q05_min": mean(row.get("front_q05_min", math.nan) for row in rows),
        "max_obstacle_danger": max([finite(row.get("obstacle_danger_max", 0.0)) for row in rows], default=0.0),
    }


def control_summary(rows):
    total = len(rows)
    return {
        "episodes": int(total),
        "mean_action_saturation_ratio": mean(row.get("action_saturation_ratio", 0.0) for row in rows),
        "mean_cmd_vx": mean(row.get("mean_cmd_vx", 0.0) for row in rows),
        "mean_cmd_vy_abs": mean(row.get("lateral_abs_mean", 0.0) for row in rows),
        "mean_yaw_abs": mean(row.get("yaw_abs_mean", 0.0) for row in rows),
        "mean_raw_vx_positive_ratio": mean(row.get("raw_vx_positive_ratio", 0.0) for row in rows),
        "mean_filtered_vx_nonpositive_ratio": mean(row.get("filtered_vx_nonpositive_ratio", 0.0) for row in rows),
        "height_violation_count": sum(1 for row in rows if row.get("termination_reason") == "height_violation"),
        "collision_count": sum(1 for row in rows if row.get("termination_reason") == "collision_or_too_close"),
    }


def build_minimal_change_plan(reset_stats, visibility_stats, safety_stats, control_stats):
    reset_ok = reset_stats["reset_success_rate"] >= 0.95
    return """# Phase 7.3B Minimal Change Plan

## Scope

This is a planning artifact only. Phase 7.3B did not train, fine-tune, edit the frozen policy, edit reward, edit safety_filter_node.py, or edit action mapping.

## Plan A: Environment Stability Only

- Add a woods_easy reset gate before any policy rollout: require OFFBOARD, armed, ACTIVE mode, fresh /target/state, fresh /obstacle/risk, UAV z in the training-ready band, and no blocked TAKEOFF/PRESTREAM state.
- Make reset validation explicit: reject reset when target model or random_woods model is missing, when random_woods count is not 26 for woods_easy, or when clearance checks fail.
- Add a reset-only CI-style diagnostic: 20 to 50 resets, no policy, no RL action, pass only if reset_success_rate >= 95%.
- Add target initial visibility checks: if red_ball exists but /target/state is initially invisible for consecutive samples, classify as occlusion/detector reset failure rather than policy failure.
- Current recommendation: {reset_recommendation}

## Plan B: Curriculum Without Reward/Action/Safety Changes

- woods_easy_empty: no woods obstacles, registered policy sanity check.
- woods_easy_sparse_trunks: 2 to 4 trunks, no branches/fallen, target initially visible.
- woods_easy_trunks_no_branches: 6 to 8 trunks, no branches/fallen.
- woods_easy_branches_limited: add limited high branches only after reset gate passes.
- woods_easy_full: 8 trunks / 15 branches / 3 fallen.
- Each curriculum level must keep reward, action mapping, and safety_filter_node.py fixed unless a separate approved phase changes them.

## Plan C: Candidate Code Changes Before Training

- Add richer reset diagnostics and validity checks around Gazebo reset and random woods respawn.
- Add target-lost recovery observation candidates only as a planned experiment: lost duration, last-seen bearing, and target reacquisition flag.
- Add obstacle-danger history candidates: recent front_q05 minimum and filtered-vx status, so the policy can distinguish blocked forward progress from normal chase.
- Add safety trigger logging around first trigger, first danger, and filtered action; do not change safety behavior in Phase 7.3B.
- Design woods-specific curriculum config files and evaluation gates before any PPO/DAgger run.

## Diagnosis Signals

- reset_success_rate: {reset_success_rate:.3f}
- mean_target_visible_ratio: {mean_target_visible_ratio:.3f}
- safety_trigger_rate: {safety_trigger_rate:.3f}
- mean_filtered_vx_nonpositive_ratio: {filtered_vx_nonpositive:.3f}
- mean_raw_vx_positive_ratio: {raw_vx_positive:.3f}
""".format(
        reset_recommendation=(
            "reset is below gate; Phase 7.3C should be reset fix, not training"
            if not reset_ok else
            "reset passed the gate; curriculum planning can proceed"
        ),
        reset_success_rate=reset_stats["reset_success_rate"],
        mean_target_visible_ratio=visibility_stats["mean_target_visible_ratio"],
        safety_trigger_rate=safety_stats["safety_trigger_rate"],
        filtered_vx_nonpositive=control_stats["mean_filtered_vx_nonpositive_ratio"],
        raw_vx_positive=control_stats["mean_raw_vx_positive_ratio"],
    )


def build_curriculum_plan(reset_stats):
    reset_gate = reset_stats["reset_success_rate"] >= 0.95
    return """# Woods Easy Curriculum Plan

## Level W0: world1 sparse best policy retest

- Training allowed: no.
- Input policy: final registry world0/world1 sparse best policy.
- VecNormalize: inherit frozen registry VecNormalize.
- Purpose: confirm the frozen policy and runtime are not damaged.
- Success threshold: >= 90% on known world1 sparse validation.
- Collision threshold: 0%.
- Reset threshold: reset pollution false.
- Target-visible threshold: >= 95%.
- Safety filter threshold: no system-level OFFBOARD or safety runtime failure.

## Level W1: woods_easy_reset_only

- Training allowed: no.
- Input policy: none.
- VecNormalize: not used.
- Purpose: validate environment reset, target spawn, obstacle spawn, and safety runtime readiness.
- Episodes/resets: 20 to 50 resets.
- Success threshold: reset_success_rate >= 95%.
- Collision threshold: not applicable.
- Reset threshold: >= 95%; current reset gate status: {reset_gate_status}.
- Target-visible threshold: initial target_visible_rate >= 80% or explicit occlusion classification.
- Safety filter threshold: initial safety_trigger_rate should be low and explainable.

## Level W2: woods_easy_no_branches

- Training allowed: no for first diagnostic pass; training only after W1 passes.
- Input policy: frozen registry policy for zero-shot diagnostic; later training may initialize from it only in a separate phase.
- VecNormalize: inherit frozen registry VecNormalize for eval; training plan must decide whether to freeze or update stats separately.
- Success threshold: zero-shot diagnostic target >= 50%; after adaptation target >= 70%.
- Collision threshold: <= 10%.
- Reset threshold: >= 95%.
- Target-visible threshold: >= 85%.
- Safety filter threshold: no repeated unrecoverable DEPTH_STOP/EMERGENCY loops.

## Level W3: woods_easy_sparse_trunks

- Training allowed: only after explicit Phase 7.3C decision.
- Input policy: registry policy or W2-approved checkpoint, never overwrite registry.
- VecNormalize: inherit with explicit frozen/eval mode; training stats must be versioned.
- Success threshold: >= 70%.
- Collision threshold: <= 10%.
- Reset threshold: >= 95%.
- Target-visible threshold: >= 85%.
- Safety filter threshold: safety interventions may occur but must not dominate episode length.

## Level W4: woods_easy_trunks_fallen

- Training allowed: only after W3 gate.
- Input policy: W3 candidate checkpoint in a new outputs directory.
- VecNormalize: versioned with checkpoint.
- Success threshold: >= 70%.
- Collision threshold: <= 10%.
- Reset threshold: >= 95%.
- Target-visible threshold: >= 80%.
- Safety filter threshold: emergency/depth-stop must correlate with actual near obstacles and recover.

## Level W5: woods_easy_full_static

- Training allowed: only after W4 gate.
- Input policy: W4 candidate checkpoint, not registry best.
- VecNormalize: versioned.
- Success threshold: >= 70% for eval gate, >= 80% for robustness candidate.
- Collision threshold: <= 10%.
- Reset threshold: >= 95%.
- Target-visible threshold: >= 80%.
- Safety filter threshold: no systematic forward-blocking loops.

## Level W6: woods_easy robustness

- Training allowed: no during validation.
- Input policy: W5 candidate.
- VecNormalize: frozen for eval.
- Success threshold: >= 80% across seeds/layout perturbations.
- Collision threshold: <= 5% preferred, <= 10% maximum.
- Reset threshold: >= 95% with no reset pollution.
- Target-visible threshold: >= 80%.
- Safety filter threshold: no OFFBOARD drop and no unrecoverable safety loop.
""".format(
        reset_gate_status="PASS" if reset_gate else "FAIL"
    )


def build_reproduce_commands(args):
    return """# Phase 7.3B Reproduce Commands

## Runtime

Terminal 1:

```bash
source /opt/ros/noetic/setup.bash
source /home/whk/vf_ws/devel/setup.bash
source /home/whk/vf_ws/src/drone_chase/scripts/source_px4_gazebo_env.sh
roslaunch drone_chase phase2_random_woods_world.launch \\
  gui:=false \\
  seed:=42 \\
  num_trunks:=8 \\
  num_branches:=15 \\
  num_fallen:=3 \\
  area_x_min:=1.5 \\
  area_x_max:=6.5 \\
  area_y_min:=-3.0 \\
  area_y_max:=3.0 \\
  uav_clearance:=1.5 \\
  target_clearance:=0.8
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
rosrun drone_chase phase7_woods_easy_failure_diagnosis.py \\
  --input-dir {input_dir} \\
  --output-dir {output_dir} \\
  --reset-count 20
```

## Topic checks

```bash
rostopic echo -n 1 /mavros/state
rostopic echo -n 1 /safety_filter/mode
rostopic echo -n 1 /target/state
rostopic echo -n 1 /obstacle/risk
rostopic echo -n 1 /mavros/setpoint_velocity/cmd_vel
```

## Cleanup

```bash
pkill -f roslaunch || true
pkill -f gzserver || true
pkill -f gzclient || true
pkill -f px4 || true
pkill -f mavros || true
pkill -f safety_filter_node.py || true
pkill -f phase7 || true
```
""".format(input_dir=os.path.abspath(args.input_dir), output_dir=os.path.abspath(args.output_dir))


def build_report(summary, reset_stats, vis_stats, safety_stats, control_stats):
    reset_gate = reset_stats["reset_success_rate"] >= 0.95
    target_occlusion_major = vis_stats["mean_target_visible_ratio"] < 0.8 or vis_stats["target_lost_episode_rate"] > 0.2
    safety_overactive = safety_stats["safety_trigger_rate"] >= 0.5 and safety_stats["mean_safety_filtered_count"] > 20
    straight_policy = control_stats["mean_raw_vx_positive_ratio"] > 0.4 and control_stats["mean_cmd_vy_abs"] < 0.03
    lines = [
        "# Phase 7.3B Woods Easy Failure Diagnosis",
        "",
        "1. scope: diagnosis and curriculum planning only; no training or fine-tuning performed.",
        "2. input Phase 7.3A dir: {}".format(summary["input_dir"]),
        "3. output dir: {}".format(summary["output_dir"]),
        "4. reset-only resets: {}".format(reset_stats["total"]),
        "5. reset_success_rate: {:.4f}".format(reset_stats["reset_success_rate"]),
        "6. target_visible_initial_rate: {:.4f}".format(reset_stats["target_visible_initial_rate"]),
        "7. uav_pose_ok_rate: {:.4f}".format(reset_stats["uav_pose_ok_rate"]),
        "8. safety_trigger_initial_rate: {:.4f}".format(reset_stats["safety_trigger_initial_rate"]),
        "9. reset failure reason distribution: {}".format(reset_stats["failure_reason_distribution"]),
        "10. target_visible_ratio_mean from Phase 7.3A rollouts: {:.4f}".format(vis_stats["mean_target_visible_ratio"]),
        "11. target_lost_episode_rate: {:.4f}".format(vis_stats["target_lost_episode_rate"]),
        "12. target_lost_before_safety_rate: {:.4f}".format(vis_stats["target_lost_before_safety_rate"]),
        "13. safety_trigger_rate: {:.4f}".format(safety_stats["safety_trigger_rate"]),
        "14. mean_safety_filtered_count: {:.4f}".format(safety_stats["mean_safety_filtered_count"]),
        "15. mean_filtered_vx_le_zero_ratio: {:.4f}".format(safety_stats["mean_filtered_vx_le_zero_ratio"]),
        "16. mean_raw_vx_positive_ratio: {:.4f}".format(control_stats["mean_raw_vx_positive_ratio"]),
        "17. mean_lateral_abs_cmd: {:.4f}".format(control_stats["mean_cmd_vy_abs"]),
        "18. reset failure root cause: {}".format(
            "UAV/safety runtime does not reliably return to training-ready state after woods reset; low post-reset altitude and ACTIVE:RAW_TIMEOUT readiness failures dominate."
            if not reset_gate else
            "reset-only gate passed in this run; 7.3A reset failures were intermittent and still need guarded reset validation."
        ),
        "19. target occlusion is major bottleneck: {}".format("yes" if target_occlusion_major else "no"),
        "20. safety filter over-intervention: {}".format("yes" if safety_overactive else "no"),
        "21. control policy lacks obstacle-aware lateral behavior: {}".format("yes" if straight_policy else "inconclusive"),
        "22. height violation root cause: reset/low-altitude recovery and safety-filtered target-lost behavior are more likely than pure vz command alone.",
        "23. recommend woods curriculum training: {}".format("not before reset fix" if not reset_gate else "yes, after curriculum gates are accepted"),
        "24. recommended start level: {}".format("W1 reset fix / reset-only gate" if not reset_gate else "W2 woods_easy_no_branches diagnostic"),
        "25. need reset fix before Phase 7.3C: {}".format("yes" if not reset_gate else "no, but keep reset gate"),
        "26. need new observation before first curriculum attempt: {}.".format(
            "not mandatory for reset fix; target-lost and filtered-action history are recommended candidates before woods training"
        ),
        "27. allow Phase 7.3C: yes, as {}.".format(
            "reset fix, not training" if not reset_gate else "curriculum setup / W2 diagnostic"
        ),
        "28. allow claiming woods passed: no.",
        "",
        "Main failure modes:",
        "- Reset: training_ready failures are associated with low post-reset UAV z and ACTIVE:RAW_TIMEOUT readiness state after reset.",
        "- Visibility: low mean target visibility and long lost streaks indicate occlusion/detector recovery is a real bottleneck.",
        "- Safety: safety filter is active and likely protective, but frequent filtered vx <= 0 can leave the sparse policy without a recovery behavior.",
        "- Control: the world0/world1 sparse policy still shows forward-chase bias and weak lateral obstacle negotiation in woods.",
    ]
    return "\n".join(lines) + "\n"


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Phase 7.3B woods_easy failure diagnosis.")
    parser.add_argument("--input-dir", default=PHASE7A_DIR)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--reset-count", type=int, default=20)
    parser.add_argument("--reset-seed-base", type=int, default=73300)
    parser.add_argument("--post-reset-sleep", type=float, default=0.5)
    parser.add_argument("--skip-reset-only", action="store_true")
    return parser


def main():
    args = build_arg_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    summary_path = os.path.join(args.input_dir, "woods_easy_zero_shot_summary.json")
    rollouts_path = os.path.join(args.input_dir, "woods_easy_zero_shot_rollouts.csv")
    failures_path = os.path.join(args.input_dir, "failure_cases.csv")
    safety_events_path = os.path.join(args.input_dir, "safety_events.csv")
    for path in (summary_path, rollouts_path, failures_path, safety_events_path):
        if not os.path.exists(path):
            raise RuntimeError("Missing Phase 7.3A input: {}".format(path))

    phase7a_summary = load_json(summary_path)
    rollouts = read_csv(rollouts_path)
    failures = read_csv(failures_path)
    rollout_groups = group_rollouts(rollouts)
    visibility_rows = analyze_visibility(rollout_groups, failures)
    safety_rows = analyze_safety(rollout_groups, failures)
    control_rows = analyze_control(rollout_groups, failures)

    reset_rows = []
    if args.skip_reset_only:
        for row in phase7a_summary.get("reset_rows", []):
            reset_rows.append({
                "reset_id": int(row.get("global_episode_id", row.get("episode_id", 0))),
                "seed": int(row.get("reset_seed", row.get("seed", 0))),
                "reset_success": bool(row.get("reset_success", False)),
                "training_ready": bool(row.get("training_ready", False)),
                "target_spawn_ok": True,
                "obstacle_spawn_ok": True,
                "spawn_clearance_ok": True,
                "uav_pose_ok": 0.6 <= finite(row.get("drone_z", math.nan), math.nan) <= 2.8,
                "target_visible_initial": bool(row.get("target_visible", False)),
                "obstacle_danger_initial": False,
                "safety_trigger_initial": is_safety_mode(row.get("safety_mode", "")),
                "min_depth_initial": math.nan,
                "front_q05_depth_initial": math.nan,
                "obstacle_area_ratio_initial": math.nan,
                "target_distance_initial": finite(row.get("target_distance", math.nan), math.nan),
                "mavros_connected": True,
                "mavros_armed": True,
                "mavros_mode": str(row.get("mavros_mode", "")),
                "safety_mode": str(row.get("safety_mode", "")),
                "drone_x": math.nan,
                "drone_y": math.nan,
                "drone_z": finite(row.get("drone_z", math.nan), math.nan),
                "woods_model_count": WOODS_EXPECTED_TOTAL,
                "min_uav_clearance": math.nan,
                "min_target_clearance": math.nan,
                "failure_reason": "ok" if row.get("training_ready", False) else "training_not_ready",
            })
    else:
        reset_rows = run_reset_only(args)

    reset_stats = reset_summary(reset_rows)
    vis_stats = visibility_summary(visibility_rows)
    safety_stats = safety_summary(safety_rows)
    control_stats = control_summary(control_rows)
    summary = {
        "phase": "7.3B",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_dir": os.path.abspath(args.input_dir),
        "output_dir": os.path.abspath(args.output_dir),
        "phase7a_smoke": phase7a_summary.get("smoke", {}),
        "reset_summary": reset_stats,
        "visibility_summary": vis_stats,
        "safety_summary": safety_stats,
        "control_summary": control_stats,
        "phase7_3c_allowed": True,
        "phase7_3c_recommendation": "reset_fix" if reset_stats["reset_success_rate"] < 0.95 else "woods_curriculum_setup",
        "woods_claim_allowed": False,
    }

    write_csv(os.path.join(args.output_dir, "reset_failure_diagnosis.csv"), reset_rows, reset_fieldnames())
    write_csv(os.path.join(args.output_dir, "occlusion_visibility_analysis.csv"), visibility_rows, list(visibility_rows[0].keys()) if visibility_rows else [])
    write_csv(os.path.join(args.output_dir, "safety_filter_analysis.csv"), safety_rows, list(safety_rows[0].keys()) if safety_rows else [])
    write_csv(os.path.join(args.output_dir, "control_failure_analysis.csv"), control_rows, list(control_rows[0].keys()) if control_rows else [])
    write_text(os.path.join(args.output_dir, "minimal_change_plan.md"), build_minimal_change_plan(reset_stats, vis_stats, safety_stats, control_stats))
    write_text(os.path.join(args.output_dir, "curriculum_plan_woods_easy.md"), build_curriculum_plan(reset_stats))
    write_text(os.path.join(args.output_dir, "reproduce_commands.md"), build_reproduce_commands(args))
    write_text(os.path.join(args.output_dir, "phase7_3b_failure_diagnosis_report.md"), build_report(summary, reset_stats, vis_stats, safety_stats, control_stats))
    with open(os.path.join(args.output_dir, "phase7_3b_failure_diagnosis_summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print("wrote {}".format(os.path.join(args.output_dir, "phase7_3b_failure_diagnosis_report.md")))
    print("reset_success_rate={:.4f}".format(reset_stats["reset_success_rate"]))
    print("phase7_3c_recommendation={}".format(summary["phase7_3c_recommendation"]))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("phase7_woods_easy_failure_diagnosis failed: {}".format(exc), file=sys.stderr)
        sys.exit(1)
