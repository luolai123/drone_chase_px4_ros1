#!/usr/bin/env python3

import argparse
import csv
import json
import os
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


def env_kwargs_from_config(config):
    env_cfg = dict(config.get("env", {}))
    env_cfg.setdefault("world_type", "world_0")
    env_cfg.setdefault("reset_mode", "episode_soft")
    return {key: value for key, value in env_cfg.items() if value is not None}


def mean(values):
    values = list(values)
    return float(sum(values)) / float(len(values)) if values else 0.0


def component(info, key, default=0.0):
    value = (info.get("reward_components", {}) or {}).get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


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
            return {"filtered_vx": 0.0, "filtered_vz": 0.0, "filtered_yaw": 0.0}
        return {
            "filtered_vx": float(msg.twist.linear.x),
            "filtered_vz": float(msg.twist.linear.z),
            "filtered_yaw": float(msg.twist.angular.z),
        }


def rollout(model_path, vecnormalize_path, config, episodes, output_csv):
    import rospy
    from geometry_msgs.msg import TwistStamped
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from gazebo_chase_env import GazeboChaseEnv

    def make_env():
        return Monitor(GazeboChaseEnv(**env_kwargs_from_config(config)))

    vec_env = DummyVecEnv([make_env])
    vec_env = VecNormalize.load(vecnormalize_path, vec_env)
    vec_env.training = False
    vec_env.norm_reward = False
    tracker = FilteredCmdTracker()
    subscriber = rospy.Subscriber("/mavros/setpoint_velocity/cmd_vel", TwistStamped, tracker.callback, queue_size=1)
    model = PPO.load(model_path, env=vec_env)

    rows = []
    try:
        for episode in range(int(episodes)):
            obs = vec_env.reset()
            prev_target_visible = None
            prev_target_depth = None
            prev_distance_used = None
            step = 0
            while True:
                action, _state = model.predict(obs, deterministic=True)
                action_row = action[0] if len(action.shape) > 1 else action
                obs, rewards, dones, infos = vec_env.step(action)
                info = infos[0]
                target_visible = bool(info.get("target_visible", False))
                target_depth = float(info.get("target_distance", 0.0))
                distance_used = component(info, "distance_used_for_reward", target_depth)
                if prev_distance_used is None:
                    prev_distance_used = component(info, "prev_distance_used_for_reward", target_depth)
                delta_distance = component(info, "delta_distance", prev_distance_used - distance_used)
                filtered = tracker.snapshot()
                done_reason = str(info.get("terminal_reason", ""))
                rows.append(
                    {
                        "episode": episode,
                        "step": step,
                        "target_visible": target_visible,
                        "prev_target_visible": "" if prev_target_visible is None else bool(prev_target_visible),
                        "target_depth": target_depth,
                        "prev_target_depth": "" if prev_target_depth is None else float(prev_target_depth),
                        "distance_used_for_reward": distance_used,
                        "prev_distance_used_for_reward": prev_distance_used,
                        "delta_distance": delta_distance,
                        "delta_distance_clipped": component(info, "delta_distance_clipped", delta_distance),
                        "approach_valid": bool(component(info, "approach_valid", 0.0) > 0.5),
                        "r_approach": component(info, "r_approach"),
                        "r_distance": component(info, "r_distance"),
                        "r_visibility": component(info, "r_visibility"),
                        "r_center": component(info, "r_center"),
                        "r_obstacle": component(info, "r_obstacle"),
                        "r_smooth": component(info, "r_smooth"),
                        "r_safety_mode": component(info, "r_safety_mode"),
                        "r_lost_extra": component(info, "r_lost_extra"),
                        "r_yaw": component(info, "r_yaw"),
                        "r_forward": component(info, "r_forward"),
                        "r_terminal": component(info, "r_terminal"),
                        "total_reward": float(rewards[0]),
                        "target_u": float(info.get("target_u", 0.0)),
                        "target_v": float(info.get("target_v", 0.0)),
                        "action_vx": float(action_row[0]),
                        "action_vy": float(action_row[1]),
                        "action_vz": float(action_row[2]),
                        "action_yaw": float(action_row[3]),
                        "filtered_vx": filtered["filtered_vx"],
                        "filtered_vz": filtered["filtered_vz"],
                        "filtered_yaw": filtered["filtered_yaw"],
                        "safety_mode": str(info.get("safety_mode", "")),
                        "drone_z": float(info.get("drone_z", 0.0)),
                        "done": bool(dones[0]),
                        "done_reason": done_reason,
                    }
                )
                prev_target_visible = target_visible
                prev_target_depth = target_depth
                prev_distance_used = distance_used
                step += 1
                if bool(dones[0]):
                    break
    finally:
        try:
            subscriber.unregister()
        except Exception:
            pass
        vec_env.close()

    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    with open(output_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames())
        writer.writeheader()
        writer.writerows(rows)
    return rows


def fieldnames():
    return [
        "episode",
        "step",
        "target_visible",
        "prev_target_visible",
        "target_depth",
        "prev_target_depth",
        "distance_used_for_reward",
        "prev_distance_used_for_reward",
        "delta_distance",
        "delta_distance_clipped",
        "approach_valid",
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
        "total_reward",
        "target_u",
        "target_v",
        "action_vx",
        "action_vy",
        "action_vz",
        "action_yaw",
        "filtered_vx",
        "filtered_vz",
        "filtered_yaw",
        "safety_mode",
        "drone_z",
        "done",
        "done_reason",
    ]


def truthy(value):
    return str(value).lower() == "true"


def audit_rows(name, rows):
    false_positive_approach = [
        row for row in rows if (not truthy(row["target_visible"])) and float(row["r_approach"]) > 1e-6
    ]
    reacquire_positive = [
        row
        for row in rows
        if str(row["prev_target_visible"]).lower() == "false"
        and truthy(row["target_visible"])
        and float(row["r_approach"]) > 1.0
    ]
    max_depth_reacquire = [
        row
        for row in rows
        if str(row["prev_target_depth"]) not in ("", "nan")
        and float(row["prev_target_depth"]) >= 9.5
        and float(row["target_depth"]) < 9.5
        and float(row["r_approach"]) > 1.0
    ]
    yaw_positive = [
        row
        for row in rows
        if abs(float(row["action_yaw"])) > 0.9 and float(row["total_reward"]) > 0.0
    ]
    off_center_positive = [
        row
        for row in rows
        if (float(row["target_u"]) ** 2 + float(row["target_v"]) ** 2) ** 0.5 > 0.7
        and float(row["r_center"]) > 0.0
    ]
    target_lost_rows = [row for row in rows if "TARGET_LOST" in str(row["safety_mode"])]
    height_guard_rows = [row for row in rows if "HEIGHT_GUARD" in str(row["safety_mode"])]
    return {
        "name": name,
        "rows": len(rows),
        "episodes": len({row["episode"] for row in rows}),
        "success_steps": sum(1 for row in rows if truthy(row["done"]) and str(row["done_reason"]) == "success"),
        "target_visible_ratio": mean(1.0 if truthy(row["target_visible"]) else 0.0 for row in rows),
        "false_visible_positive_approach_steps": len(false_positive_approach),
        "lost_to_visible_large_approach_steps": len(reacquire_positive),
        "max_depth_reacquire_large_approach_steps": len(max_depth_reacquire),
        "yaw_saturated_positive_reward_steps": len(yaw_positive),
        "off_center_positive_center_reward_steps": len(off_center_positive),
        "target_lost_steps": len(target_lost_rows),
        "height_guard_steps": len(height_guard_rows),
        "r_approach_mean": mean(float(row["r_approach"]) for row in rows),
        "total_reward_mean": mean(float(row["total_reward"]) for row in rows),
    }


def write_markdown(path, final_summary, best_summary):
    lines = [
        "# Phase 7.1D Reward Hacking Audit",
        "",
        "## Final model",
        "",
    ]
    for key, value in final_summary.items():
        if key != "name":
            lines.append("- {}: {}".format(key, value))
    lines.extend(["", "## Best checkpoint", ""])
    for key, value in best_summary.items():
        if key != "name":
            lines.append("- {}: {}".format(key, value))
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            "- target_visible=False positive r_approach indicates direct lost-step reward leakage.",
            "- lost_to_visible/max_depth reacquire positive r_approach indicates reward can be harvested by losing and reacquiring the target.",
            "- yaw_saturated_positive_reward_steps indicates saturated yaw is not sufficiently penalized by the old reward.",
            "- off_center_positive_center_reward_steps indicates the old center reward still pays positive reward when the target is near the image edge.",
        ]
    )
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Audit failed PPO rollouts for reward hacking.")
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--config", default=None)
    parser.add_argument("--episodes", type=int, default=3)
    return parser


def main():
    args = build_arg_parser().parse_args()
    require_training_deps()
    input_dir = os.path.abspath(args.input_dir)
    config_path = args.config or os.path.join(input_dir, "config_effective.yaml")
    if not os.path.exists(config_path):
        config_path = os.path.join(input_dir, "config_used.yaml")
    config = load_yaml(config_path)

    best_json = os.path.join(input_dir, "best_checkpoint.json")
    if not os.path.exists(best_json):
        raise RuntimeError("Missing {}; run phase7_checkpoint_sweep.py first".format(best_json))
    with open(best_json, "r") as handle:
        best = json.load(handle)

    final_rows = rollout(
        os.path.join(input_dir, "model.zip"),
        os.path.join(input_dir, "vecnormalize.pkl"),
        config,
        args.episodes,
        os.path.join(input_dir, "final_failed_rollout_audit.csv"),
    )
    best_rows = rollout(
        best["model_path"],
        best["vecnormalize_path"],
        config,
        args.episodes,
        os.path.join(input_dir, "best_failed_rollout_audit.csv"),
    )

    final_summary = audit_rows("final", final_rows)
    best_summary = audit_rows("best", best_rows)
    md_path = os.path.join(input_dir, "reward_hacking_audit.md")
    write_markdown(md_path, final_summary, best_summary)
    print("final_summary={}".format(final_summary))
    print("best_summary={}".format(best_summary))
    print("wrote {}".format(md_path))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("phase7_failed_rollout_audit failed: {}".format(exc), file=sys.stderr)
        sys.exit(1)
