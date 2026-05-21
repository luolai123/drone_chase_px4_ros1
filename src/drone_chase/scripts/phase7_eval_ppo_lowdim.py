#!/usr/bin/env python3

import argparse
import csv
import os
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "envs"))
if ENV_DIR not in sys.path:
    sys.path.insert(0, ENV_DIR)

INSTALL_CMD = (
    "/usr/bin/python3 -m pip install --user "
    "stable-baselines3==2.3.2 gymnasium==0.29.1 'torch<2.5' tensorboard pandas"
)


def require_training_deps():
    missing = []
    try:
        import gymnasium  # noqa: F401
    except ImportError:
        missing.append("gymnasium")
    try:
        import stable_baselines3  # noqa: F401
    except ImportError:
        missing.append("stable-baselines3")
    try:
        import torch  # noqa: F401
    except ImportError:
        missing.append("torch")
    if missing:
        raise RuntimeError(
            "Missing Phase 7 dependencies: {}. Suggested install command: {}".format(
                ", ".join(missing),
                INSTALL_CMD,
            )
        )


def load_yaml(path):
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read config files") from exc
    with open(path, "r") as handle:
        return yaml.safe_load(handle) or {}


def env_kwargs_from_config(config):
    env_cfg = dict(config.get("env", {}))
    woods_cfg = config.get("woods", {})
    kwargs = dict(env_cfg)
    kwargs.setdefault("reset_mode", "episode_soft")
    if woods_cfg:
        mapping = {
            "seed": "seed",
            "num_trunks": "woods_easy_num_trunks",
            "num_branches": "woods_easy_num_branches",
            "num_fallen": "woods_easy_num_fallen",
            "area_x_min": "woods_easy_area_x_min",
            "area_x_max": "woods_easy_area_x_max",
            "area_y_min": "woods_easy_area_y_min",
            "area_y_max": "woods_easy_area_y_max",
            "uav_clearance": "woods_easy_uav_clearance",
            "target_clearance": "woods_easy_target_clearance",
        }
        for src, dst in mapping.items():
            if src in woods_cfg and dst not in kwargs:
                kwargs[dst] = woods_cfg[src]
    return {key: value for key, value in kwargs.items() if value is not None}


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).lower() in ("1", "true", "yes", "y")


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Evaluate a low-dimensional PPO policy on GazeboChaseEnv.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--vecnormalize", required=True)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--deterministic", default=None)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--output-rollout-csv", default=None)
    return parser


def action_to_raw_cmd(action, env_cfg):
    max_vx = float(env_cfg.get("max_vx", 0.5))
    min_vx = float(env_cfg.get("min_vx", -0.2))
    max_vy = float(env_cfg.get("max_vy", 0.3))
    max_vz = float(env_cfg.get("max_vz", 0.25))
    max_yaw_rate = float(env_cfg.get("max_yaw_rate", 0.6))
    ax = float(action[0])
    vx = ax * max_vx if ax >= 0.0 else ax * abs(min_vx)
    return {
        "raw_vx": vx,
        "raw_vy": float(action[1]) * max_vy,
        "raw_vz": float(action[2]) * max_vz,
        "raw_yaw": float(action[3]) * max_yaw_rate,
    }


class FilteredCmdTracker:
    def __init__(self):
        import threading

        self._lock = threading.Lock()
        self._msg = None

    def callback(self, msg):
        with self._lock:
            self._msg = msg

    def snapshot(self):
        with self._lock:
            msg = self._msg
        if msg is None:
            return {
                "filtered_vx": 0.0,
                "filtered_vy": 0.0,
                "filtered_vz": 0.0,
                "filtered_yaw": 0.0,
            }
        return {
            "filtered_vx": float(msg.twist.linear.x),
            "filtered_vy": float(msg.twist.linear.y),
            "filtered_vz": float(msg.twist.linear.z),
            "filtered_yaw": float(msg.twist.angular.z),
        }


def reward_component_row(info):
    components = info.get("reward_components", {}) or {}
    return {
        "r_approach": float(components.get("r_approach", 0.0)),
        "r_distance": float(components.get("r_distance", 0.0)),
        "r_visibility": float(components.get("r_visibility", 0.0)),
        "r_center": float(components.get("r_center", 0.0)),
        "r_obstacle": float(components.get("r_obstacle", 0.0)),
        "r_smooth": float(components.get("r_smooth", 0.0)),
        "r_safety_mode": float(components.get("r_safety_mode", 0.0)),
        "r_lost_extra": float(components.get("r_lost_extra", 0.0)),
        "r_yaw": float(components.get("r_yaw", 0.0)),
        "r_forward": float(components.get("r_forward", 0.0)),
        "r_terminal": float(components.get("r_terminal", 0.0)),
        "delta_distance": float(components.get("delta_distance", 0.0)),
        "delta_distance_clipped": float(components.get("delta_distance_clipped", 0.0)),
        "approach_valid": float(components.get("approach_valid", 0.0)),
    }


def main():
    args = build_arg_parser().parse_args()
    require_training_deps()

    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    import rospy
    from geometry_msgs.msg import TwistStamped
    from gazebo_chase_env import GazeboChaseEnv

    config = load_yaml(args.config)
    env_cfg = dict(config.get("env", {}))
    episodes = args.episodes or int(config.get("eval", {}).get("eval_episodes", 10))
    deterministic = (
        str_to_bool(args.deterministic)
        if args.deterministic is not None
        else bool(config.get("eval", {}).get("deterministic", True))
    )
    output_csv = args.output_csv or os.path.join(os.path.dirname(args.model), "eval_results.csv")
    output_rollout_csv = args.output_rollout_csv
    tracker = FilteredCmdTracker()

    def make_env():
        return Monitor(GazeboChaseEnv(**env_kwargs_from_config(config)))

    vec_env = DummyVecEnv([make_env])
    vec_env = VecNormalize.load(args.vecnormalize, vec_env)
    vec_env.training = False
    vec_env.norm_reward = False
    rospy.Subscriber("/mavros/setpoint_velocity/cmd_vel", TwistStamped, tracker.callback, queue_size=1)
    model = PPO.load(args.model, env=vec_env)

    rows = []
    rollout_rows = []
    for episode in range(episodes):
        obs = vec_env.reset()
        total_reward = 0.0
        length = 0
        visible_count = 0
        emergency_count = 0
        depth_stop_count = 0
        target_lost_count = 0
        min_target_distance = float("inf")
        final_target_distance = float("inf")
        terminated_reason = ""
        terminated = False
        truncated = False

        while True:
            action, _state = model.predict(obs, deterministic=deterministic)
            obs, rewards, dones, infos = vec_env.step(action)
            info = infos[0]
            action_row = action[0] if len(action.shape) > 1 else action
            raw_cmd = action_to_raw_cmd(action_row, env_cfg)
            filtered_cmd = tracker.snapshot()
            reward_terms = reward_component_row(info)
            total_reward += float(rewards[0])
            length += 1
            visible_count += int(bool(info.get("target_visible", False)))
            final_target_distance = float(info.get("target_distance", final_target_distance))
            min_target_distance = min(min_target_distance, final_target_distance)
            mode = str(info.get("safety_mode", ""))
            emergency_count += int("EMERGENCY_AVOID" in mode)
            depth_stop_count += int("DEPTH_STOP" in mode)
            target_lost_count += int("TARGET_LOST" in mode)
            terminated_reason = str(info.get("terminal_reason", ""))
            truncated = bool(info.get("timeout", False))
            terminated = bool(dones[0]) and not truncated
            rollout_rows.append(
                {
                    "episode": episode,
                    "step": length - 1,
                    "reward": float(rewards[0]),
                    "done": bool(dones[0]),
                    "success": bool(info.get("success", False)),
                    "timeout": bool(info.get("timeout", False)),
                    "terminated_reason": terminated_reason,
                    "target_visible": bool(info.get("target_visible", False)),
                    "target_distance": final_target_distance,
                    "min_target_distance_so_far": min_target_distance,
                    "target_u": float(info.get("target_u", 0.0)),
                    "target_v": float(info.get("target_v", 0.0)),
                    "front_q05_depth": float(info.get("front_q05_depth", 0.0)),
                    "safety_mode": mode,
                    "mavros_mode": str(info.get("mavros_mode", "")),
                    "mavros_armed": bool(info.get("mavros_armed", False)),
                    "action_vx": float(action_row[0]),
                    "action_vy": float(action_row[1]),
                    "action_vz": float(action_row[2]),
                    "action_yaw": float(action_row[3]),
                    "drone_z": float(info.get("drone_z", 0.0)),
                    **raw_cmd,
                    **filtered_cmd,
                    **reward_terms,
                }
            )
            if bool(dones[0]):
                break

        rows.append(
            {
                "episode": episode,
                "success": terminated_reason == "success",
                "reward": total_reward,
                "length": length,
                "terminated": terminated,
                "truncated": truncated,
                "terminated_reason": terminated_reason or "unknown",
                "final_target_distance": final_target_distance,
                "min_target_distance": min_target_distance,
                "target_visible_ratio": float(visible_count) / float(max(1, length)),
                "emergency_count": emergency_count,
                "depth_stop_count": depth_stop_count,
                "target_lost_count": target_lost_count,
            }
        )
        print("episode={} result={}".format(episode, rows[-1]))

    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    fieldnames = [
        "episode",
        "success",
        "reward",
        "length",
        "terminated",
        "truncated",
        "terminated_reason",
        "final_target_distance",
        "min_target_distance",
        "target_visible_ratio",
        "emergency_count",
        "depth_stop_count",
        "target_lost_count",
    ]
    with open(output_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    if output_rollout_csv:
        rollout_fieldnames = [
            "episode",
            "step",
            "reward",
            "done",
            "success",
            "timeout",
            "terminated_reason",
            "target_visible",
            "target_distance",
            "min_target_distance_so_far",
            "target_u",
            "target_v",
            "front_q05_depth",
            "safety_mode",
            "mavros_mode",
            "mavros_armed",
            "action_vx",
            "action_vy",
            "action_vz",
            "action_yaw",
            "drone_z",
            "raw_vx",
            "raw_vy",
            "raw_vz",
            "raw_yaw",
            "filtered_vx",
            "filtered_vy",
            "filtered_vz",
            "filtered_yaw",
            "r_approach",
            "r_distance",
            "r_visibility",
            "r_center",
            "r_obstacle",
            "r_smooth",
            "r_safety_mode",
            "r_lost_extra",
            "r_yaw",
            "r_forward",
            "r_terminal",
            "delta_distance",
            "delta_distance_clipped",
            "approach_valid",
        ]
        os.makedirs(os.path.dirname(os.path.abspath(output_rollout_csv)), exist_ok=True)
        with open(output_rollout_csv, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rollout_fieldnames)
            writer.writeheader()
            writer.writerows(rollout_rows)
    vec_env.close()
    print("wrote {}".format(output_csv))
    if output_rollout_csv:
        print("wrote {}".format(output_rollout_csv))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("phase7_eval_ppo_lowdim failed: {}".format(exc), file=sys.stderr)
        sys.exit(1)
