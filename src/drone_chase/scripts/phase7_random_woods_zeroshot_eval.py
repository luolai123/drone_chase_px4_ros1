#!/usr/bin/env python3

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import threading
import time
from collections import Counter
from datetime import datetime

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "envs"))
if ENV_DIR not in sys.path:
    sys.path.insert(0, ENV_DIR)


REGISTRY_DIR = "/home/whk/vf_ws/outputs/final_policy_registry"
REGISTERED_MODEL = os.path.join(REGISTRY_DIR, "best_world0_world1_policy.zip")
REGISTERED_VECNORMALIZE = os.path.join(REGISTRY_DIR, "best_world0_world1_vecnormalize.pkl")
MANIFEST_PATH = os.path.join(REGISTRY_DIR, "policy_manifest.json")
DEFAULT_OUTPUT_DIR = "/home/whk/vf_ws/outputs/phase7/random_woods_zeroshot_from_final_policy"
DEFAULT_CONFIG = (
    "/home/whk/vf_ws/outputs/phase7/world0_ppo_from_bc_v2_10k_conservative_run4/config_effective.yaml"
)

SAFETY_TOKENS = ("EMERGENCY_AVOID", "DEPTH_STOP", "HEIGHT_GUARD", "TARGET_LOST")
BLOCKED_RESET_TOKENS = ("WAIT_FCU", "PRESTREAM", "SET_MODE", "ARMING", "TAKEOFF")


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
        raise RuntimeError("Missing Phase 7 eval dependencies: {}".format(", ".join(missing)))


def load_json(path):
    with open(path, "r") as handle:
        return json.load(handle)


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


def verify_registry_policy(model_path, vecnormalize_path, manifest_path):
    if os.path.abspath(model_path) != os.path.abspath(REGISTERED_MODEL):
        raise RuntimeError("Phase 7.4A must load registered policy only; got model={}".format(model_path))
    if os.path.abspath(vecnormalize_path) != os.path.abspath(REGISTERED_VECNORMALIZE):
        raise RuntimeError(
            "Phase 7.4A must load registered VecNormalize only; got vecnormalize={}".format(vecnormalize_path)
        )
    if not os.path.exists(model_path):
        raise RuntimeError("registered policy missing: {}".format(model_path))
    if not os.path.exists(vecnormalize_path):
        raise RuntimeError("registered VecNormalize missing: {}".format(vecnormalize_path))
    if not os.path.exists(manifest_path):
        raise RuntimeError("policy manifest missing: {}".format(manifest_path))
    manifest = load_json(manifest_path)
    model_sha = sha256(model_path)
    vec_sha = sha256(vecnormalize_path)
    if model_sha != manifest.get("registered_model_sha256"):
        raise RuntimeError("registered model SHA mismatch: {}".format(model_sha))
    if vec_sha != manifest.get("registered_vecnormalize_sha256"):
        raise RuntimeError("registered VecNormalize SHA mismatch: {}".format(vec_sha))
    return manifest, model_sha, vec_sha


def finite(value, default=math.nan):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(default)
    return value if math.isfinite(value) else float(default)


def mean(values):
    values = [float(v) for v in values if math.isfinite(float(v))]
    return float(sum(values)) / float(len(values)) if values else 0.0


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
        "raw_vx_body": ax * max_vx if ax >= 0.0 else ax * abs(min_vx),
        "raw_vy_body": float(action[1]) * max_vy,
        "raw_vz_body": float(action[2]) * max_vz,
        "raw_yaw_rate": float(action[3]) * max_yaw_rate,
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
            "raw_vx_body": finite(raw["vx"]) if raw is not None else finite(fallback_raw["raw_vx_body"]),
            "raw_vy_body": finite(raw["vy"]) if raw is not None else finite(fallback_raw["raw_vy_body"]),
            "raw_vz_body": finite(raw["vz"]) if raw is not None else finite(fallback_raw["raw_vz_body"]),
            "raw_yaw_rate": finite(raw["yaw"]) if raw is not None else finite(fallback_raw["raw_yaw_rate"]),
            "filtered_vx_body": finite(filtered["vx"]) if filtered is not None else math.nan,
            "filtered_vy_body": finite(filtered["vy"]) if filtered is not None else math.nan,
            "filtered_vz_body": finite(filtered["vz"]) if filtered is not None else math.nan,
            "filtered_yaw_rate": finite(filtered["yaw"]) if filtered is not None else math.nan,
            "published_vx_world": finite(published["vx"]) if published is not None else math.nan,
            "published_vy_world": finite(published["vy"]) if published is not None else math.nan,
            "published_vz_world": finite(published["vz"]) if published is not None else math.nan,
            "published_yaw_rate": finite(published["yaw"]) if published is not None else math.nan,
            "debug_raw_available": bool(raw is not None),
            "debug_filtered_available": bool(filtered is not None),
            "published_available": bool(published is not None),
        }

    def close(self):
        for subscriber in self.subscribers:
            try:
                subscriber.unregister()
            except Exception:
                pass


def env_kwargs_from_config(config, seed, world):
    env_cfg = dict(config.get("env", {}))
    env_cfg["world_type"] = str(world)
    env_cfg["reset_mode"] = "soft"
    env_cfg["respawn_target_on_reset"] = False
    env_cfg["seed"] = int(seed)
    env_cfg.setdefault("random_woods_num_trunks", 18)
    env_cfg.setdefault("random_woods_num_branches", 45)
    env_cfg.setdefault("random_woods_num_fallen", 10)
    env_cfg.setdefault("random_woods_area_x_min", 1.0)
    env_cfg.setdefault("random_woods_area_x_max", 6.5)
    env_cfg.setdefault("random_woods_area_y_min", -3.5)
    env_cfg.setdefault("random_woods_area_y_max", 3.5)
    env_cfg.setdefault("random_woods_uav_clearance", 1.0)
    env_cfg.setdefault("random_woods_target_clearance", 0.6)
    env_cfg.setdefault("target_x", 4.0)
    env_cfg.setdefault("target_y", 0.0)
    env_cfg.setdefault("target_z", 1.0)
    env_cfg.setdefault("reset_ready_timeout", 20.0)
    env_cfg.setdefault("reset_zero_cmd_duration", 1.0)
    return {key: value for key, value in env_cfg.items() if value is not None}


def reset_ready_enough(reset_info):
    safety_mode = str(reset_info.get("safety_mode", ""))
    return bool(
        reset_info.get("topics_ready", False)
        and reset_info.get("training_ready", False)
        and reset_info.get("mavros_mode", "") == "OFFBOARD"
        and not any(token in safety_mode for token in BLOCKED_RESET_TOKENS)
    )


def reset_with_recovery(env, seed, retries):
    attempts = []
    last_obs = None
    last_info = {}
    for attempt in range(int(retries) + 1):
        attempt_seed = int(seed) + attempt * 100000
        env.config["seed"] = int(attempt_seed)
        obs, info = env.reset(seed=attempt_seed, options={"reset_mode": "soft"})
        row = {
            "attempt": int(attempt),
            "seed": int(seed),
            "reset_seed": int(attempt_seed),
            "reset_success": bool(info.get("reset_success", False)),
            "topics_ready": bool(info.get("topics_ready", False)),
            "training_ready": bool(info.get("training_ready", False)),
            "reset_mode_used": str(info.get("reset_mode_used", info.get("reset_mode", ""))),
            "mavros_mode": str(info.get("mavros_mode", "")),
            "safety_mode": str(info.get("safety_mode", "")),
            "target_visible": bool(info.get("target_visible", False)),
            "target_distance": finite(info.get("target_distance", math.nan)),
            "drone_z": finite(info.get("drone_z", math.nan)),
            "accepted_for_episode": False,
            "accepted_with_warning": False,
        }
        attempts.append(row)
        last_obs = obs
        last_info = info
        if row["reset_success"]:
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
        "eval_split",
        "episode",
        "episode_id",
        "global_episode_id",
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
        "action_vy",
        "action_vz",
        "action_yaw",
        "raw_vx_body",
        "raw_vy_body",
        "raw_vz_body",
        "raw_yaw_rate",
        "filtered_vx_body",
        "filtered_vy_body",
        "filtered_vz_body",
        "filtered_yaw_rate",
        "published_vx_world",
        "published_vy_world",
        "published_vz_world",
        "published_yaw_rate",
        "drone_z",
        "drone_x",
        "drone_y",
        "reward",
        "done",
        "done_reason",
        "termination_reason",
        "mavros_mode",
        "mavros_armed",
        "success_count",
        "safety_filtered",
        "filtered_vx_limited",
    ]


def episode_fieldnames():
    return [
        "eval_split",
        "episode",
        "episode_id",
        "global_episode_id",
        "seed",
        "success",
        "done_reason",
        "termination_reason",
        "capture_step",
        "final_distance",
        "min_distance",
        "target_visible_ratio",
        "target_lost_count",
        "collision",
        "terminal_collision",
        "safety_triggered",
        "safety_filtered_count",
        "obstacle_danger_mean",
        "obstacle_danger_max",
        "min_depth",
        "altitude_invalid",
        "height_violation",
        "out_of_bounds",
        "timeout",
        "offboard_loss",
        "offboard_drop",
        "emergency_count",
        "depth_stop_count",
        "raw_timeout_count",
        "front_q05_min",
        "obstacle_danger_steps",
        "mean_cmd_vx",
        "mean_cmd_vy",
        "mean_cmd_vz",
        "mean_yaw_rate",
        "filtered_vx_count",
        "episode_return",
        "episode_length",
        "reset_recovery_used",
        "reset_pollution_detected",
    ]


def safety_event_fieldnames():
    return [
        "eval_split",
        "episode_id",
        "global_episode_id",
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


def is_safety_mode(mode):
    return any(token in str(mode) for token in SAFETY_TOKENS)


def classify_failure(row):
    if row["success"]:
        return "none"
    if row["termination_reason"] == "reset_failure":
        return "environment_reset_failure"
    if row["terminal_collision"] or row["collision"]:
        return "collision_or_too_close"
    if row["altitude_invalid"]:
        return "altitude_invalid"
    if row["out_of_bounds"]:
        return "out_of_bounds"
    if row["offboard_loss"]:
        return "offboard_loss"
    if row["timeout"]:
        if row["target_visible_ratio"] < 0.8:
            return "timeout_target_lost_or_occluded"
        if row["safety_filtered_count"] > 0:
            return "timeout_safety_filtered"
        return "timeout_control_or_generalization"
    if row["target_visible_ratio"] < 0.8:
        return "target_lost_or_low_visibility"
    return "unknown"


def make_reset_failure_episode(split, episode_id, global_episode_id, seed, reset_row):
    target_visible = bool(reset_row.get("target_visible", False))
    target_distance = finite(reset_row.get("target_distance", math.nan))
    drone_z = finite(reset_row.get("drone_z", math.nan))
    mavros_mode = str(reset_row.get("mavros_mode", ""))
    safety_mode = str(reset_row.get("safety_mode", ""))
    return {
        "eval_split": split,
        "episode": int(episode_id),
        "episode_id": int(episode_id),
        "global_episode_id": int(global_episode_id),
        "seed": int(seed),
        "success": False,
        "done_reason": "reset_failure",
        "termination_reason": "reset_failure",
        "capture_step": -1,
        "final_distance": target_distance,
        "min_distance": target_distance,
        "target_visible_ratio": 1.0 if target_visible else 0.0,
        "target_lost_count": 0 if target_visible else 1,
        "collision": False,
        "terminal_collision": False,
        "safety_triggered": is_safety_mode(safety_mode),
        "safety_filtered_count": 0,
        "obstacle_danger_mean": 0.0,
        "obstacle_danger_max": 0.0,
        "min_depth": math.nan,
        "altitude_invalid": bool(math.isfinite(drone_z) and drone_z < 0.6),
        "height_violation": bool(math.isfinite(drone_z) and drone_z < 0.6),
        "out_of_bounds": False,
        "timeout": False,
        "offboard_loss": bool(mavros_mode != "OFFBOARD"),
        "offboard_drop": int(mavros_mode != "OFFBOARD"),
        "emergency_count": 0,
        "depth_stop_count": 0,
        "raw_timeout_count": int("RAW_TIMEOUT" in safety_mode),
        "front_q05_min": math.nan,
        "obstacle_danger_steps": 0,
        "mean_cmd_vx": 0.0,
        "mean_cmd_vy": 0.0,
        "mean_cmd_vz": 0.0,
        "mean_yaw_rate": 0.0,
        "filtered_vx_count": 0,
        "episode_return": 0.0,
        "episode_length": 0,
        "reset_recovery_used": bool(int(reset_row.get("attempt", 0)) > 0),
        "reset_pollution_detected": False,
    }


def run_episode(raw_env, vec_env, model, monitor, env_cfg, split, episode_id, global_episode_id, seed, deterministic, reset_retries):
    obs, reset_info, reset_row, reset_attempts = reset_with_recovery(raw_env, seed, reset_retries)
    if not reset_row["accepted_for_episode"]:
        episode_row = make_reset_failure_episode(split, episode_id, global_episode_id, seed, reset_row)
        return episode_row, [], [], reset_attempts

    rollout_rows = []
    safety_events = []
    visible_count = 0
    target_lost_count = 0
    min_distance = float("inf")
    final_distance = float("inf")
    front_depths = []
    obstacle_dangers = []
    raw_vx_values = []
    raw_vy_values = []
    raw_vz_values = []
    raw_yaw_values = []
    episode_return = 0.0
    safety_filtered_count = 0
    filtered_vx_count = 0
    raw_timeout_count = 0
    obstacle_danger_steps = 0
    safety_triggered = False
    collision = False
    terminal_collision = False
    altitude_invalid = False
    out_of_bounds = False
    timeout = False
    offboard_loss = False
    capture_step = -1
    termination_reason = ""
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
        termination_reason = str(info.get("terminal_reason", "")) if done else ""
        target_visible = bool(info.get("target_visible", False))
        target_depth = finite(info.get("target_distance", math.nan))
        front_q05_depth = finite(info.get("front_q05_depth", obs[7] if len(obs) > 7 else math.nan))
        left_q05_depth = finite(obs[8] if len(obs) > 8 else math.nan)
        right_q05_depth = finite(obs[9] if len(obs) > 9 else math.nan)
        obstacle_area_ratio = finite(info.get("obstacle_area_ratio", obs[10] if len(obs) > 10 else math.nan))
        obstacle_danger = float(info.get("obstacle_danger", obs[11] if len(obs) > 11 else 0.0))
        obstacle_danger_bool = bool(obstacle_danger > 0.5)
        drone_z = finite(info.get("drone_z", math.nan))
        raw_vx = snapshot["raw_vx_body"]
        filtered_vx = snapshot["filtered_vx_body"]
        filtered_vx_limited = bool(
            math.isfinite(filtered_vx)
            and math.isfinite(raw_vx)
            and raw_vx > 0.05
            and filtered_vx < raw_vx - 0.05
        )
        safety_filtered = bool(is_safety_mode(mode) or filtered_vx_limited)
        raw_timeout = bool("RAW_TIMEOUT" in mode)

        visible_count += int(target_visible)
        target_lost_count += int(not target_visible)
        min_distance = min(min_distance, target_depth)
        final_distance = target_depth
        front_depths.append(front_q05_depth)
        obstacle_dangers.append(obstacle_danger)
        raw_vx_values.append(snapshot["raw_vx_body"])
        raw_vy_values.append(snapshot["raw_vy_body"])
        raw_vz_values.append(snapshot["raw_vz_body"])
        raw_yaw_values.append(snapshot["raw_yaw_rate"])
        episode_return += float(reward)
        safety_filtered_count += int(safety_filtered)
        filtered_vx_count += int(filtered_vx_limited)
        raw_timeout_count += int(raw_timeout)
        obstacle_danger_steps += int(obstacle_danger_bool)
        safety_triggered = bool(safety_triggered or is_safety_mode(mode))
        collision = bool(collision or info.get("collision_condition", False) or info.get("collision", False))
        terminal_collision = bool(terminal_collision or termination_reason == "collision_or_too_close")
        altitude_invalid = bool(altitude_invalid or info.get("height_violation_condition", False) or info.get("height_violation", False))
        out_of_bounds = bool(out_of_bounds or info.get("out_of_bounds", False))
        timeout = bool(timeout or info.get("timeout", False) or termination_reason == "timeout")
        offboard_loss = bool(offboard_loss or str(info.get("mavros_mode", "")) != "OFFBOARD")
        if capture_step < 0 and int(info.get("success_count", 0)) > 0:
            capture_step = int(step)

        row = {
            "eval_split": split,
            "episode": int(episode_id),
            "episode_id": int(episode_id),
            "global_episode_id": int(global_episode_id),
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
            "obstacle_danger": obstacle_danger,
            "safety_mode": mode,
            "action_vx": finite(action_row[0]),
            "action_vy": finite(action_row[1]),
            "action_vz": finite(action_row[2]),
            "action_yaw": finite(action_row[3]),
            "raw_vx_body": snapshot["raw_vx_body"],
            "raw_vy_body": snapshot["raw_vy_body"],
            "raw_vz_body": snapshot["raw_vz_body"],
            "raw_yaw_rate": snapshot["raw_yaw_rate"],
            "filtered_vx_body": snapshot["filtered_vx_body"],
            "filtered_vy_body": snapshot["filtered_vy_body"],
            "filtered_vz_body": snapshot["filtered_vz_body"],
            "filtered_yaw_rate": snapshot["filtered_yaw_rate"],
            "published_vx_world": snapshot["published_vx_world"],
            "published_vy_world": snapshot["published_vy_world"],
            "published_vz_world": snapshot["published_vz_world"],
            "published_yaw_rate": snapshot["published_yaw_rate"],
            "drone_z": drone_z,
            "drone_x": finite(info.get("drone_x", math.nan)),
            "drone_y": finite(info.get("drone_y", math.nan)),
            "reward": float(reward),
            "done": bool(done),
            "done_reason": termination_reason if done else "",
            "termination_reason": termination_reason,
            "mavros_mode": str(info.get("mavros_mode", "")),
            "mavros_armed": bool(info.get("mavros_armed", False)),
            "success_count": int(info.get("success_count", 0)),
            "safety_filtered": bool(safety_filtered),
            "filtered_vx_limited": bool(filtered_vx_limited),
        }
        rollout_rows.append(row)
        if is_safety_mode(mode) or obstacle_danger_bool or filtered_vx_limited:
            event_type = []
            if is_safety_mode(mode):
                event_type.append("safety_mode")
            if obstacle_danger_bool:
                event_type.append("obstacle_danger")
            if filtered_vx_limited:
                event_type.append("filtered_vx_limited")
            safety_events.append(
                {
                    "eval_split": split,
                    "episode_id": int(episode_id),
                    "global_episode_id": int(global_episode_id),
                    "seed": int(seed),
                    "step": int(step),
                    "event_type": "+".join(event_type),
                    "safety_mode": mode,
                    "obstacle_danger": obstacle_danger,
                    "front_q05_depth": front_q05_depth,
                    "obstacle_area_ratio": obstacle_area_ratio,
                    "raw_vx_body": snapshot["raw_vx_body"],
                    "filtered_vx_body": snapshot["filtered_vx_body"],
                    "drone_z": drone_z,
                }
            )
        step += 1
        if done:
            break

    episode_length = int(step)
    episode_row = {
        "eval_split": split,
        "episode": int(episode_id),
        "episode_id": int(episode_id),
        "global_episode_id": int(global_episode_id),
        "seed": int(seed),
        "success": bool(termination_reason == "success"),
        "done_reason": termination_reason or "unknown",
        "termination_reason": termination_reason or "unknown",
        "capture_step": int(capture_step),
        "final_distance": final_distance,
        "min_distance": min_distance,
        "target_visible_ratio": rate(visible_count, episode_length),
        "target_lost_count": int(target_lost_count),
        "collision": bool(collision),
        "terminal_collision": bool(terminal_collision),
        "safety_triggered": bool(safety_triggered),
        "safety_filtered_count": int(safety_filtered_count),
        "obstacle_danger_mean": mean(obstacle_dangers),
        "obstacle_danger_max": max(obstacle_dangers) if obstacle_dangers else 0.0,
        "min_depth": min(front_depths) if front_depths else math.nan,
        "altitude_invalid": bool(altitude_invalid),
        "height_violation": bool(altitude_invalid),
        "out_of_bounds": bool(out_of_bounds),
        "timeout": bool(timeout),
        "offboard_loss": bool(offboard_loss),
        "offboard_drop": int(offboard_loss),
        "emergency_count": int(info.get("emergency_count", 0)) if "info" in locals() else 0,
        "depth_stop_count": int(info.get("depth_stop_count", 0)) if "info" in locals() else 0,
        "raw_timeout_count": int(raw_timeout_count),
        "front_q05_min": min(front_depths) if front_depths else math.nan,
        "obstacle_danger_steps": int(obstacle_danger_steps),
        "mean_cmd_vx": mean(raw_vx_values),
        "mean_cmd_vy": mean(raw_vy_values),
        "mean_cmd_vz": mean(raw_vz_values),
        "mean_yaw_rate": mean(raw_yaw_values),
        "filtered_vx_count": int(filtered_vx_count),
        "episode_return": float(episode_return),
        "episode_length": int(episode_length),
        "reset_recovery_used": bool(int(reset_row.get("attempt", 0)) > 0),
        "reset_pollution_detected": bool(
            reset_row.get("accepted_for_episode", False)
            and (not reset_row.get("reset_success", False) or reset_row.get("accepted_with_warning", False))
        ),
    }
    return episode_row, rollout_rows, safety_events, reset_attempts


def summarize(rows):
    total = len(rows)
    success_count = sum(1 for row in rows if row["success"])
    termination_distribution = Counter(str(row["termination_reason"]) for row in rows)
    target_lost_episode_count = sum(1 for row in rows if float(row["target_visible_ratio"]) < 0.8)
    collision_count = sum(1 for row in rows if row["collision"])
    terminal_collision_count = sum(1 for row in rows if row["terminal_collision"])
    height_violation_count = sum(1 for row in rows if row["altitude_invalid"])
    out_of_bounds_count = sum(1 for row in rows if row["out_of_bounds"])
    timeout_count = sum(1 for row in rows if row["timeout"])
    offboard_drop_count = sum(int(row.get("offboard_drop", int(row["offboard_loss"]))) for row in rows)
    emergency_count = sum(int(row.get("emergency_count", 0)) for row in rows)
    depth_stop_count = sum(int(row.get("depth_stop_count", 0)) for row in rows)
    raw_timeout_count = sum(int(row.get("raw_timeout_count", 0)) for row in rows)
    collision_emergency_failure_count = sum(
        1
        for row in rows
        if (not row["success"])
        and (
            row["collision"]
            or row["terminal_collision"]
            or bool(row.get("safety_triggered", False))
            or int(row.get("emergency_count", 0)) > 0
            or int(row.get("depth_stop_count", 0)) > 0
        )
    )
    return {
        "total_episodes": int(total),
        "success_count": int(success_count),
        "success_rate": rate(success_count, total),
        "collision_count": int(collision_count),
        "collision_rate": rate(collision_count, total),
        "terminal_collision_count": int(terminal_collision_count),
        "terminal_collision_rate": rate(terminal_collision_count, total),
        "height_violation_count": int(height_violation_count),
        "height_violation_rate": rate(height_violation_count, total),
        "altitude_invalid_rate": rate(height_violation_count, total),
        "target_lost_rate": rate(target_lost_episode_count, total),
        "target_lost_episode_count": int(target_lost_episode_count),
        "timeout_count": int(timeout_count),
        "timeout_rate": rate(timeout_count, total),
        "out_of_bounds_count": int(out_of_bounds_count),
        "out_of_bounds_rate": rate(out_of_bounds_count, total),
        "offboard_drop_count": int(offboard_drop_count),
        "offboard_loss_rate": rate(offboard_drop_count, total),
        "mean_final_distance": mean(row["final_distance"] for row in rows),
        "mean_min_distance": mean(row["min_distance"] for row in rows),
        "mean_target_visible_ratio": mean(row["target_visible_ratio"] for row in rows),
        "mean_episode_length": mean(row["episode_length"] for row in rows),
        "raw_timeout_count": int(raw_timeout_count),
        "emergency_count": int(emergency_count),
        "depth_stop_count": int(depth_stop_count),
        "front_q05_min_mean": mean(row.get("front_q05_min", math.nan) for row in rows),
        "obstacle_danger_steps_mean": mean(row.get("obstacle_danger_steps", 0) for row in rows),
        "collision_emergency_failure_count": int(collision_emergency_failure_count),
        "collision_emergency_failure_rate": rate(collision_emergency_failure_count, total),
        "mean_safety_filtered_count": mean(row["safety_filtered_count"] for row in rows),
        "mean_obstacle_danger": mean(row["obstacle_danger_mean"] for row in rows),
        "max_obstacle_danger": max((row["obstacle_danger_max"] for row in rows), default=0.0),
        "safety_trigger_rate": rate(sum(1 for row in rows if row["safety_triggered"]), total),
        "termination_reason_distribution": dict(sorted(termination_distribution.items())),
    }


def ideal_gate_pass(summary, reset_pollution_detected, safety_filter_normal, expected_episodes):
    return bool(
        summary["total_episodes"] == int(expected_episodes)
        and summary["success_rate"] >= 0.70
        and summary["mean_target_visible_ratio"] > 0.80
        and summary["mean_final_distance"] <= 1.50
        and summary["mean_min_distance"] <= 1.20
        and summary["raw_timeout_count"] == 0
        and summary["out_of_bounds_count"] == 0
        and summary["height_violation_count"] == 0
        and summary["offboard_loss_rate"] <= 0.05
        and summary["collision_emergency_failure_rate"] <= 0.10
        and safety_filter_normal
        and not reset_pollution_detected
    )


def minimum_gate_pass(summary, reset_pollution_detected, safety_filter_normal, expected_episodes):
    return bool(
        summary["total_episodes"] == int(expected_episodes)
        and summary["success_rate"] >= 0.50
        and summary["mean_target_visible_ratio"] > 0.70
        and summary["mean_min_distance"] <= 1.80
        and summary["raw_timeout_count"] == 0
        and summary["collision_emergency_failure_rate"] <= 0.20
        and summary["offboard_loss_rate"] <= 0.10
        and safety_filter_normal
        and not reset_pollution_detected
    )


def gate_pass(summary, reset_pollution_detected, safety_filter_normal, expected_episodes):
    return minimum_gate_pass(summary, reset_pollution_detected, safety_filter_normal, expected_episodes)


def smoke_gate_pass(summary, expected_episodes, reset_pollution_detected):
    return bool(
        summary["total_episodes"] == int(expected_episodes)
        and gate_pass(summary, reset_pollution_detected, True, expected_episodes)
    )


def analyze_failures(failures, summary):
    if not failures:
        return ["none"]
    categories = Counter(classify_failure(row) for row in failures)
    lines = ["{}={}".format(key, value) for key, value in sorted(categories.items())]
    if summary["mean_target_visible_ratio"] < 0.8:
        lines.append("perception: low mean target visibility suggests random_woods occlusion or detector limits")
    if summary["mean_safety_filtered_count"] > 20.0:
        lines.append("safety_filter: frequent filtering may prevent approach through random_woods")
    if summary["terminal_collision_rate"] > 0.0:
        lines.append("control/safety: terminal collision observed")
    if summary["offboard_loss_rate"] > 0.0:
        lines.append("environment: OFFBOARD loss observed")
    return lines


def build_diagnostic_answers(summary):
    low_visibility = bool(summary["mean_target_visible_ratio"] <= 0.80 or summary["target_lost_episode_count"] > 0)
    frequent_depth_risk = bool(summary["emergency_count"] > 0 or summary["depth_stop_count"] > 0)
    straight_line_risk = bool(
        summary["success_rate"] < 0.50
        and (summary["collision_rate"] > 0.0 or summary["collision_emergency_failure_rate"] > 0.0)
    )
    spawn_layout_issue = bool(
        "reset_failure" in summary["termination_reason_distribution"]
        or summary["mean_target_visible_ratio"] <= 0.70
    )
    roi_sensitive = bool(
        summary["obstacle_danger_steps_mean"] > 20.0
        or summary["mean_safety_filtered_count"] > 20.0
        or summary["depth_stop_count"] > 0
    )
    adaptation_needed = bool(summary["success_rate"] < 0.50 or summary["mean_min_distance"] > 1.80)
    return [
        "目标遮挡导致 target_visible_ratio 下降：{}".format("是" if low_visibility else "否"),
        "depth risk 频繁触发 emergency/depth stop：{}".format("是" if frequent_depth_risk else "否"),
        "policy 疑似偏直线追球、难处理 random_woods 树木遮挡：{}".format("是" if straight_line_risk else "否"),
        "red_ball spawn 与 random_woods trees/obstacles 可能形成不可见或不可达布局：{}".format(
            "是" if spawn_layout_issue else "否"
        ),
        "depth ROI 对 random_woods 场景可能过于敏感：{}".format("是" if roi_sensitive else "否"),
        "是否需要 random_woods DAgger demos：{}".format("是，建议先采集诊断/示范再训练" if adaptation_needed else "暂不需要"),
        "是否需要 random_woods BC adaptation：{}".format("是，若后续允许训练应先走 BC/DAgger adaptation" if adaptation_needed else "暂不需要"),
        "是否需要调整 random_woods spawn 规则，而不是直接改策略：{}".format(
            "是，先排查可见性/可达性与 reset 布局" if spawn_layout_issue else "暂不需要"
        ),
    ]


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path, data):
    with open(path, "w") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def fmt(value, digits=4):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return "{:.{}f}".format(value, digits)
    return str(value)


def write_report(path, summary):
    formal = summary["formal"]
    smoke = summary["smoke"]
    eval_summary = formal if formal["total_episodes"] > 0 else smoke
    passed_text = "是" if summary["phase7_4a_passed"] else "否"
    phase7_4b_text = "是" if summary["phase7_4b_allowed"] else "否"
    random_woods_claim_text = "是" if summary["random_woods_claim_allowed"] else "否"
    all_woods_claim_text = "是" if summary["all_woods_claim_allowed"] else "否"
    lines = [
        "# Phase 7.4A Random Woods Zero-Shot Validation",
        "",
        "1. evaluated model：{}".format(summary["policy_path"]),
        "2. evaluated vecnormalize：{}".format(summary["vecnormalize_path"]),
        "3. world：{}".format(summary["world"]),
        "4. episodes：{}".format(eval_summary["total_episodes"]),
        "5. success rate：{} ({}/{})".format(
            fmt(eval_summary["success_rate"]),
            eval_summary["success_count"],
            eval_summary["total_episodes"],
        ),
        "6. timeout rate：{}".format(fmt(eval_summary["timeout_rate"])),
        "7. collision/emergency failure rate：{} ({}/{})".format(
            fmt(eval_summary["collision_emergency_failure_rate"]),
            eval_summary["collision_emergency_failure_count"],
            eval_summary["total_episodes"],
        ),
        "8. out_of_bounds rate：{}".format(fmt(eval_summary["out_of_bounds_rate"])),
        "9. height_violation rate：{}".format(fmt(eval_summary["height_violation_rate"])),
        "10. target_visible_ratio_mean：{}".format(fmt(eval_summary["mean_target_visible_ratio"])),
        "11. final_distance_mean：{} m".format(fmt(eval_summary["mean_final_distance"])),
        "12. min_distance_mean：{} m".format(fmt(eval_summary["mean_min_distance"])),
        "13. mean episode length：{}".format(fmt(eval_summary["mean_episode_length"])),
        "14. RAW_TIMEOUT count：{}".format(eval_summary["raw_timeout_count"]),
        "15. emergency_count：{}".format(eval_summary["emergency_count"]),
        "16. depth_stop_count：{}".format(eval_summary["depth_stop_count"]),
        "17. front_q05_min_mean：{} m".format(fmt(eval_summary["front_q05_min_mean"])),
        "18. obstacle_danger_steps_mean：{}".format(fmt(eval_summary["obstacle_danger_steps_mean"])),
        "19. OFFBOARD drop count：{}".format(eval_summary["offboard_drop_count"]),
        "20. reset pollution detected：{}".format("是" if summary["reset_pollution_detected"] else "否"),
        "21. main failure modes：{}".format("; ".join(summary["failure_analysis"])),
        "22. 是否通过 random_woods zero-shot gate：{}".format(passed_text),
        "23. 是否允许进入 Phase 7.4B：{}".format(phase7_4b_text),
        "24. 是否允许声称 random_woods 已通过：{}".format(random_woods_claim_text),
        "25. 是否允许声称全部 woods 已通过：{}".format(all_woods_claim_text),
        "",
        "Gate detail:",
        "- ideal_gate_passed: {}".format("是" if summary["ideal_gate_passed"] else "否"),
        "- minimum_gate_passed: {}".format("是" if summary["minimum_gate_passed"] else "否"),
        "- termination reason distribution: {}".format(eval_summary["termination_reason_distribution"]),
        "",
        "Failure diagnosis:",
    ]
    lines.extend("- {}".format(item) for item in summary["diagnostic_answers"])
    lines.extend([
        "",
        "Notes:",
        "- target_lost_rate is measured as the fraction of eval episodes with target_visible_ratio < 0.8.",
        "- This is zero-shot evaluation of the frozen final registered policy; no training or fine-tuning was performed.",
        "- Passing Phase 7.4A only permits Phase 7.4B random_woods robustness/stress validation.",
        "- random_woods cannot be claimed complete from zero-shot alone, and all woods cannot be claimed complete from this phase.",
    ])
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")


def write_reproduce_commands(path, args):
    content = """# Phase 7.4A Random Woods Zero-Shot Reproduce Commands

## Runtime

Terminal 1:

```bash
source /opt/ros/noetic/setup.bash
source /home/whk/vf_ws/devel/setup.bash
source /home/whk/vf_ws/src/drone_chase/scripts/source_px4_gazebo_env.sh
roslaunch drone_chase phase2_chase_world.launch \\
  world:=random_woods \\
  gui:=false \\
  interactive:=false \\
  seed:=42
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
rosrun drone_chase phase7_random_woods_zeroshot_eval.py \\
  --model {model} \\
  --vecnormalize {vecnormalize} \\
  --manifest {manifest} \\
  --world {world} \\
  --episodes {episodes} \\
  --deterministic \\
  --output-dir {output_dir}
```

## Checks

```bash
source /opt/ros/noetic/setup.bash
source /home/whk/vf_ws/devel/setup.bash
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
""".format(
        model=args.model,
        vecnormalize=args.vecnormalize,
        manifest=args.manifest,
        world=args.world,
        episodes=args.episodes,
        output_dir=args.output_dir,
    )
    with open(path, "w") as handle:
        handle.write(content)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Phase 7.4A random_woods zero-shot eval.")
    parser.add_argument("--model", default=REGISTERED_MODEL)
    parser.add_argument("--vecnormalize", default=REGISTERED_VECNORMALIZE)
    parser.add_argument("--manifest", default=MANIFEST_PATH)
    parser.add_argument("--world", default="random_woods", choices=["random_woods"])
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--smoke-episodes", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed-base", type=int, default=73000)
    parser.add_argument("--reset-retries", type=int, default=2)
    parser.add_argument("--deterministic", dest="deterministic", action="store_true", default=True)
    parser.add_argument("--stochastic", dest="deterministic", action="store_false")
    return parser


def main():
    args = build_arg_parser().parse_args()
    require_eval_deps()

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from gazebo_chase_env import GazeboChaseEnv

    manifest, model_sha, vec_sha = verify_registry_policy(args.model, args.vecnormalize, args.manifest)
    config = load_yaml(args.config)
    env_cfg = env_kwargs_from_config(config, args.seed_base, args.world)
    os.makedirs(args.output_dir, exist_ok=True)

    raw_env = GazeboChaseEnv(**env_cfg)
    vec_env = DummyVecEnv([lambda: raw_env])
    vec_env = VecNormalize.load(args.vecnormalize, vec_env)
    vec_env.training = False
    vec_env.norm_reward = False
    model = PPO.load(args.model)
    monitor = CommandMonitor()

    all_episode_rows = []
    all_rollout_rows = []
    all_safety_events = []
    all_reset_rows = []
    global_episode = 0

    try:
        eval_splits = []
        if int(args.smoke_episodes) > 0:
            eval_splits.append(("smoke", int(args.smoke_episodes)))
        eval_splits.append(("formal", int(args.episodes)))
        for split, count in eval_splits:
            if split == "formal" and int(args.smoke_episodes) > 0:
                smoke_rows = [row for row in all_episode_rows if row["eval_split"] == "smoke"]
                smoke_summary = summarize(smoke_rows)
                smoke_reset_pollution = any(
                    row.get("accepted_for_episode", False)
                    and (not row.get("reset_success", False) or row.get("accepted_with_warning", False))
                    for row in all_reset_rows
                    if row.get("eval_split") == "smoke"
                )
                smoke_pass = smoke_gate_pass(smoke_summary, int(args.smoke_episodes), smoke_reset_pollution)
                if not smoke_pass:
                    print("smoke_failed; skipping formal eval")
                    break

            for episode_id in range(count):
                seed = int(args.seed_base) + int(global_episode)
                episode_row, rollout_rows, safety_events, reset_attempts = run_episode(
                    raw_env,
                    vec_env,
                    model,
                    monitor,
                    env_cfg,
                    split,
                    episode_id,
                    global_episode,
                    seed,
                    args.deterministic,
                    args.reset_retries,
                )
                for reset_row in reset_attempts:
                    reset_copy = dict(reset_row)
                    reset_copy["eval_split"] = split
                    reset_copy["episode_id"] = int(episode_id)
                    reset_copy["global_episode_id"] = int(global_episode)
                    all_reset_rows.append(reset_copy)
                all_episode_rows.append(episode_row)
                all_rollout_rows.extend(rollout_rows)
                all_safety_events.extend(safety_events)
                print(
                    "split={} episode={} seed={} success={} reason={} final={:.3f} min={:.3f} "
                    "visible={:.3f} lost={} safety_filtered={} offboard_loss={}".format(
                        split,
                        episode_id,
                        seed,
                        episode_row["success"],
                        episode_row["termination_reason"],
                        episode_row["final_distance"],
                        episode_row["min_distance"],
                        episode_row["target_visible_ratio"],
                        episode_row["target_lost_count"],
                        episode_row["safety_filtered_count"],
                        episode_row["offboard_loss"],
                    )
                )
                global_episode += 1
    finally:
        monitor.close()
        vec_env.close()

    rollout_csv = os.path.join(args.output_dir, "random_woods_zeroshot_rollouts.csv")
    episode_csv = os.path.join(args.output_dir, "random_woods_zeroshot_episodes.csv")
    failure_csv = os.path.join(args.output_dir, "failure_cases.csv")
    safety_csv = os.path.join(args.output_dir, "safety_events.csv")
    summary_json = os.path.join(args.output_dir, "random_woods_zeroshot_summary.json")
    report_md = os.path.join(args.output_dir, "phase7_4a_report.md")
    reproduce_md = os.path.join(args.output_dir, "reproduce_commands.md")

    smoke_rows = [row for row in all_episode_rows if row["eval_split"] == "smoke"]
    formal_rows = [row for row in all_episode_rows if row["eval_split"] == "formal"]
    smoke_summary = summarize(smoke_rows)
    formal_summary = summarize(formal_rows)
    reset_pollution_detected = any(
        row.get("accepted_for_episode", False)
        and (not row.get("reset_success", False) or row.get("accepted_with_warning", False))
        for row in all_reset_rows
    )
    safety_filter_normal = True
    reset_failure_detected = any(row["termination_reason"] == "reset_failure" for row in all_episode_rows)
    smoke_reset_pollution = any(
        row.get("accepted_for_episode", False)
        and (not row.get("reset_success", False) or row.get("accepted_with_warning", False))
        for row in all_reset_rows
        if row.get("eval_split") == "smoke"
    )
    smoke_passed = (
        smoke_gate_pass(smoke_summary, int(args.smoke_episodes), smoke_reset_pollution)
        if int(args.smoke_episodes) > 0
        else True
    )
    ideal_passed = ideal_gate_pass(formal_summary, reset_pollution_detected, safety_filter_normal, int(args.episodes))
    minimum_passed = minimum_gate_pass(
        formal_summary, reset_pollution_detected, safety_filter_normal, int(args.episodes)
    )
    phase_passed = bool(
        len(formal_rows) == int(args.episodes)
        and gate_pass(formal_summary, reset_pollution_detected, safety_filter_normal, int(args.episodes))
        and not reset_failure_detected
    )
    failure_rows = []
    for row in all_episode_rows:
        if not row["success"] or row["collision"] or row["offboard_loss"] or row["altitude_invalid"]:
            item = dict(row)
            item["failure_category"] = classify_failure(row)
            failure_rows.append(item)
    analysis_rows = [row for row in formal_rows if not row["success"]]
    analysis_summary = formal_summary
    if not analysis_rows and not formal_rows:
        analysis_rows = [row for row in smoke_rows if not row["success"]]
        analysis_summary = smoke_summary
    failure_analysis = analyze_failures(analysis_rows, analysis_summary)
    if reset_failure_detected:
        failure_analysis.append("environment: reset failed before a valid episode in at least one case")
    if not smoke_passed and not formal_rows:
        failure_analysis.append("smoke gate failed; formal 30-episode eval skipped")
    diagnostic_answers = build_diagnostic_answers(formal_summary if formal_rows else smoke_summary)

    woods_cfg = {
        "num_trunks": env_cfg.get("random_woods_num_trunks"),
        "num_branches": env_cfg.get("random_woods_num_branches"),
        "num_fallen": env_cfg.get("random_woods_num_fallen"),
        "area_x_min": env_cfg.get("random_woods_area_x_min"),
        "area_x_max": env_cfg.get("random_woods_area_x_max"),
        "area_y_min": env_cfg.get("random_woods_area_y_min"),
        "area_y_max": env_cfg.get("random_woods_area_y_max"),
        "uav_clearance": env_cfg.get("random_woods_uav_clearance"),
        "target_clearance": env_cfg.get("random_woods_target_clearance"),
    }
    summary = {
        "phase": "7.4A",
        "report_date": datetime.now().isoformat(timespec="seconds"),
        "policy_path": os.path.abspath(args.model),
        "vecnormalize_path": os.path.abspath(args.vecnormalize),
        "manifest_path": os.path.abspath(args.manifest),
        "model_sha256": model_sha,
        "vecnormalize_sha256": vec_sha,
        "sha256_check_passed": True,
        "manifest_policy_name": manifest.get("policy_name"),
        "world": args.world,
        "random_woods_config": woods_cfg,
        "deterministic": bool(args.deterministic),
        "smoke": smoke_summary,
        "formal": formal_summary,
        "smoke_passed": bool(smoke_passed),
        "formal_skipped_due_to_smoke_failure": bool(not smoke_passed and not formal_rows),
        "ideal_gate_passed": bool(ideal_passed),
        "minimum_gate_passed": bool(minimum_passed),
        "reset_failure_detected": bool(reset_failure_detected),
        "reset_pollution_detected": bool(reset_pollution_detected),
        "safety_filter_normal": bool(safety_filter_normal),
        "phase7_4a_passed": bool(phase_passed),
        "phase7_4b_allowed": bool(phase_passed),
        "random_woods_claim_allowed": False,
        "all_woods_claim_allowed": False,
        "failure_analysis": failure_analysis,
        "diagnostic_answers": diagnostic_answers,
        "episode_rows": all_episode_rows,
        "reset_rows": all_reset_rows,
    }
    eval_summary = formal_summary if formal_rows else smoke_summary
    summary.update(
        {
            "episodes": eval_summary["total_episodes"],
            "success_rate": eval_summary["success_rate"],
            "timeout_rate": eval_summary["timeout_rate"],
            "collision_emergency_failure_rate": eval_summary["collision_emergency_failure_rate"],
            "out_of_bounds_rate": eval_summary["out_of_bounds_rate"],
            "height_violation_rate": eval_summary["height_violation_rate"],
            "target_visible_ratio_mean": eval_summary["mean_target_visible_ratio"],
            "final_distance_mean": eval_summary["mean_final_distance"],
            "min_distance_mean": eval_summary["mean_min_distance"],
            "mean_episode_length": eval_summary["mean_episode_length"],
            "raw_timeout_count": eval_summary["raw_timeout_count"],
            "emergency_count": eval_summary["emergency_count"],
            "depth_stop_count": eval_summary["depth_stop_count"],
            "front_q05_min_mean": eval_summary["front_q05_min_mean"],
            "obstacle_danger_steps_mean": eval_summary["obstacle_danger_steps_mean"],
            "offboard_drop_count": eval_summary["offboard_drop_count"],
        }
    )

    write_csv(rollout_csv, all_rollout_rows, rollout_fieldnames())
    write_csv(episode_csv, all_episode_rows, episode_fieldnames())
    write_csv(failure_csv, failure_rows, episode_fieldnames() + ["failure_category"])
    write_csv(safety_csv, all_safety_events, safety_event_fieldnames())
    write_summary(summary_json, summary)
    write_report(report_md, summary)
    write_reproduce_commands(reproduce_md, args)

    print("wrote {}".format(rollout_csv))
    print("wrote {}".format(episode_csv))
    print("wrote {}".format(summary_json))
    print("wrote {}".format(report_md))
    print("wrote {}".format(failure_csv))
    print("wrote {}".format(safety_csv))
    print("wrote {}".format(reproduce_md))
    print("phase7_4a_passed={}".format(summary["phase7_4a_passed"]))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("phase7_random_woods_zeroshot_eval failed: {}".format(exc), file=sys.stderr)
        sys.exit(1)
