#!/usr/bin/env python3

import argparse
import csv
import glob
import json
import os
import re
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "envs"))
if ENV_DIR not in sys.path:
    sys.path.insert(0, ENV_DIR)

DEFAULT_INPUT_DIR = "/home/whk/vf_ws/outputs/phase7/world0_30k_curriculum_v1"


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


def mean(values):
    values = list(values)
    return float(sum(values)) / float(len(values)) if values else 0.0


def stats(values):
    values = [float(v) for v in values]
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    mu = mean(values)
    var = mean((v - mu) ** 2.0 for v in values)
    return {
        "mean": mu,
        "std": var ** 0.5,
        "min": min(values),
        "max": max(values),
    }


def env_kwargs_from_config(config):
    env_cfg = dict(config.get("env", {}))
    kwargs = dict(env_cfg)
    kwargs.setdefault("world_type", "world_0")
    kwargs.setdefault("reset_mode", "episode_soft")
    return {key: value for key, value in kwargs.items() if value is not None}


def checkpoint_step(path):
    match = re.search(r"ppo_step_(\d+)\.zip$", os.path.basename(path))
    return int(match.group(1)) if match else None


def discover_checkpoints(input_dir):
    candidates = []
    checkpoint_dir = os.path.join(input_dir, "checkpoints")
    for model_path in sorted(glob.glob(os.path.join(checkpoint_dir, "ppo_step_*.zip")), key=checkpoint_step):
        step = checkpoint_step(model_path)
        vec_path = os.path.join(checkpoint_dir, "vecnormalize_step_{}.pkl".format(step))
        if os.path.exists(vec_path):
            candidates.append(
                {
                    "checkpoint_step": str(step),
                    "checkpoint_name": "step_{}".format(step),
                    "model_path": model_path,
                    "vecnormalize_path": vec_path,
                }
            )
    final_model = os.path.join(input_dir, "model.zip")
    final_vec = os.path.join(input_dir, "vecnormalize.pkl")
    if os.path.exists(final_model) and os.path.exists(final_vec):
        candidates.append(
            {
                "checkpoint_step": "final",
                "checkpoint_name": "final",
                "model_path": final_model,
                "vecnormalize_path": final_vec,
            }
        )
    return candidates


def action_to_raw_cmd(action, env_cfg):
    max_vx = float(env_cfg.get("max_vx", 0.5))
    min_vx = float(env_cfg.get("min_vx", -0.2))
    max_vy = float(env_cfg.get("max_vy", 0.3))
    max_vz = float(env_cfg.get("max_vz", 0.25))
    max_yaw_rate = float(env_cfg.get("max_yaw_rate", 0.6))
    ax = float(action[0])
    return {
        "raw_vx": ax * max_vx if ax >= 0.0 else ax * abs(min_vx),
        "raw_vy": float(action[1]) * max_vy,
        "raw_vz": float(action[2]) * max_vz,
        "raw_yaw": float(action[3]) * max_yaw_rate,
    }


class FilteredCmdTracker:
    def __init__(self):
        import threading

        self.lock = threading.Lock()
        self.msg = None

    def callback(self, msg):
        with self.lock:
            self.msg = msg

    def snapshot(self):
        with self.lock:
            msg = self.msg
        if msg is None:
            return {"filtered_vx": 0.0, "filtered_vy": 0.0, "filtered_vz": 0.0, "filtered_yaw": 0.0}
        return {
            "filtered_vx": float(msg.twist.linear.x),
            "filtered_vy": float(msg.twist.linear.y),
            "filtered_vz": float(msg.twist.linear.z),
            "filtered_yaw": float(msg.twist.angular.z),
        }


def evaluate_candidate(candidate, config, episodes, deterministic):
    import rospy
    from geometry_msgs.msg import TwistStamped
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from gazebo_chase_env import GazeboChaseEnv

    env_cfg = dict(config.get("env", {}))

    def make_env():
        return Monitor(GazeboChaseEnv(**env_kwargs_from_config(config)))

    vec_env = DummyVecEnv([make_env])
    vec_env = VecNormalize.load(candidate["vecnormalize_path"], vec_env)
    vec_env.training = False
    vec_env.norm_reward = False
    tracker = FilteredCmdTracker()
    subscriber = rospy.Subscriber("/mavros/setpoint_velocity/cmd_vel", TwistStamped, tracker.callback, queue_size=1)
    model = PPO.load(candidate["model_path"], env=vec_env)

    episode_rows = []
    action_vx = []
    action_yaw = []
    filtered_vx = []
    yaw_abs = []
    target_depth_lt_1_count = 0
    emergency_count = 0
    depth_stop_count = 0
    raw_timeout_count = 0
    height_guard_count = 0
    target_lost_steps = 0

    try:
        for episode in range(int(episodes)):
            obs = vec_env.reset()
            total_reward = 0.0
            visible_count = 0
            min_distance = float("inf")
            first_distance = None
            final_distance = float("inf")
            terminated_reason = ""
            step = 0
            while True:
                action, _state = model.predict(obs, deterministic=deterministic)
                action_row = action[0] if len(action.shape) > 1 else action
                obs, rewards, dones, infos = vec_env.step(action)
                info = infos[0]
                mode = str(info.get("safety_mode", ""))
                reward = float(rewards[0])
                distance = float(info.get("target_distance", final_distance))
                if first_distance is None:
                    first_distance = distance
                final_distance = distance
                min_distance = min(min_distance, distance)
                total_reward += reward
                visible_count += int(bool(info.get("target_visible", False)))
                target_depth_lt_1_count += int(distance < 1.0)
                emergency_count += int("EMERGENCY_AVOID" in mode)
                depth_stop_count += int("DEPTH_STOP" in mode)
                raw_timeout_count += int("RAW_TIMEOUT" in mode)
                height_guard_count += int("HEIGHT_GUARD" in mode)
                target_lost_steps += int("TARGET_LOST" in mode)
                filtered = tracker.snapshot()
                action_vx.append(float(action_row[0]))
                action_yaw.append(float(action_row[3]))
                filtered_vx.append(float(filtered["filtered_vx"]))
                yaw_abs.append(abs(float(action_row[3])))
                terminated_reason = str(info.get("terminal_reason", ""))
                step += 1
                if bool(dones[0]):
                    break
            episode_rows.append(
                {
                    "success": terminated_reason == "success",
                    "timeout": terminated_reason == "timeout",
                    "reward": total_reward,
                    "length": step,
                    "first_distance": first_distance if first_distance is not None else 0.0,
                    "final_distance": final_distance,
                    "min_distance": min_distance,
                    "target_visible_ratio": float(visible_count) / float(max(1, step)),
                }
            )
    finally:
        try:
            subscriber.unregister()
        except Exception:
            pass
        vec_env.close()

    vx_stats = stats(action_vx)
    yaw_stats = stats(action_yaw)
    filtered_vx_stats = stats(filtered_vx)
    return {
        **candidate,
        "deterministic": bool(deterministic),
        "episodes": int(episodes),
        "success_rate": mean(1.0 if row["success"] else 0.0 for row in episode_rows),
        "timeout_rate": mean(1.0 if row["timeout"] else 0.0 for row in episode_rows),
        "target_visible_ratio_mean": mean(row["target_visible_ratio"] for row in episode_rows),
        "final_distance_mean": mean(row["final_distance"] for row in episode_rows),
        "min_distance_mean": mean(row["min_distance"] for row in episode_rows),
        "first_distance_mean": mean(row["first_distance"] for row in episode_rows),
        "target_depth_lt_1_count": int(target_depth_lt_1_count),
        "mean_reward": mean(row["reward"] for row in episode_rows),
        "mean_episode_len": mean(row["length"] for row in episode_rows),
        "emergency_count": int(emergency_count),
        "depth_stop_count": int(depth_stop_count),
        "raw_timeout_count": int(raw_timeout_count),
        "height_guard_count": int(height_guard_count),
        "action_vx_mean": vx_stats["mean"],
        "action_vx_std": vx_stats["std"],
        "action_vx_min": vx_stats["min"],
        "action_vx_max": vx_stats["max"],
        "action_yaw_mean": yaw_stats["mean"],
        "action_yaw_std": yaw_stats["std"],
        "action_yaw_min": yaw_stats["min"],
        "action_yaw_max": yaw_stats["max"],
        "filtered_vx_mean": filtered_vx_stats["mean"],
        "filtered_vx_std": filtered_vx_stats["std"],
        "filtered_vx_min": filtered_vx_stats["min"],
        "filtered_vx_max": filtered_vx_stats["max"],
        "yaw_abs_mean": mean(yaw_abs),
        "target_lost_steps": int(target_lost_steps),
    }


def best_row(rows):
    def key(row):
        return (
            float(row["success_rate"]),
            -float(row["min_distance_mean"]),
            -float(row["final_distance_mean"]),
            float(row["target_visible_ratio_mean"]),
            float(row["target_depth_lt_1_count"]),
        )

    return max(rows, key=key)


def fieldnames():
    return [
        "checkpoint_step",
        "checkpoint_name",
        "model_path",
        "vecnormalize_path",
        "deterministic",
        "episodes",
        "success_rate",
        "timeout_rate",
        "target_visible_ratio_mean",
        "final_distance_mean",
        "min_distance_mean",
        "first_distance_mean",
        "target_depth_lt_1_count",
        "mean_reward",
        "mean_episode_len",
        "emergency_count",
        "depth_stop_count",
        "raw_timeout_count",
        "height_guard_count",
        "action_vx_mean",
        "action_vx_std",
        "action_vx_min",
        "action_vx_max",
        "action_yaw_mean",
        "action_yaw_std",
        "action_yaw_min",
        "action_yaw_max",
        "filtered_vx_mean",
        "filtered_vx_std",
        "filtered_vx_min",
        "filtered_vx_max",
        "yaw_abs_mean",
        "target_lost_steps",
    ]


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Sweep Phase 7 PPO checkpoints with deterministic eval.")
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--config", default=None)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--stochastic-episodes", type=int, default=0)
    return parser


def main():
    args = build_arg_parser().parse_args()
    require_training_deps()
    input_dir = os.path.abspath(args.input_dir)
    config_path = args.config or os.path.join(input_dir, "config_effective.yaml")
    if not os.path.exists(config_path):
        config_path = os.path.join(input_dir, "config_used.yaml")
    config = load_yaml(config_path)
    candidates = discover_checkpoints(input_dir)
    if not candidates:
        raise RuntimeError("No checkpoints found in {}".format(input_dir))

    rows = []
    for candidate in candidates:
        row = evaluate_candidate(candidate, config, args.episodes, deterministic=True)
        rows.append(row)
        print("deterministic checkpoint={} success_rate={:.3f} min_distance={:.3f}".format(
            row["checkpoint_step"], row["success_rate"], row["min_distance_mean"]
        ))
        if args.stochastic_episodes > 0:
            stochastic = evaluate_candidate(candidate, config, args.stochastic_episodes, deterministic=False)
            stochastic["checkpoint_name"] = "{}_stochastic".format(stochastic["checkpoint_name"])
            rows.append(stochastic)

    csv_path = os.path.join(input_dir, "checkpoint_sweep.csv")
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames())
        writer.writeheader()
        writer.writerows(rows)

    deterministic_rows = [row for row in rows if bool(row["deterministic"])]
    best = best_row(deterministic_rows)
    summary = {
        "input_dir": input_dir,
        "config": config_path,
        "best_checkpoint_step": best["checkpoint_step"],
        "best_model_path": best["model_path"],
        "best_vecnormalize_path": best["vecnormalize_path"],
        "best_success_rate": best["success_rate"],
        "best_target_visible_ratio_mean": best["target_visible_ratio_mean"],
        "best_final_distance_mean": best["final_distance_mean"],
        "best_min_distance_mean": best["min_distance_mean"],
        "rows": deterministic_rows,
    }
    summary_path = os.path.join(input_dir, "checkpoint_sweep_summary.json")
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    best_path = os.path.join(input_dir, "best_checkpoint.json")
    with open(best_path, "w") as handle:
        json.dump(
            {
                "checkpoint_step": best["checkpoint_step"],
                "model_path": best["model_path"],
                "vecnormalize_path": best["vecnormalize_path"],
                "success_rate": best["success_rate"],
                "target_visible_ratio_mean": best["target_visible_ratio_mean"],
                "final_distance_mean": best["final_distance_mean"],
                "min_distance_mean": best["min_distance_mean"],
            },
            handle,
            indent=2,
            sort_keys=True,
        )
    print("wrote {}".format(csv_path))
    print("wrote {}".format(summary_path))
    print("wrote {}".format(best_path))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("phase7_checkpoint_sweep failed: {}".format(exc), file=sys.stderr)
        sys.exit(1)
