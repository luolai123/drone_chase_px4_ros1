#!/usr/bin/env python3

import argparse
import csv
import os
import sys
import time

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_ENV_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "envs"))
if SOURCE_ENV_DIR not in sys.path:
    sys.path.insert(0, SOURCE_ENV_DIR)

try:
    import rospkg

    PACKAGE_ENV_DIR = os.path.join(rospkg.RosPack().get_path("drone_chase"), "envs")
    if PACKAGE_ENV_DIR not in sys.path:
        sys.path.insert(0, PACKAGE_ENV_DIR)
except ImportError:
    pass

from gazebo_chase_env import GazeboChaseEnv


DEFAULT_OUTPUT = "/home/whk/vf_ws/outputs/phase6/random_policy_rollout.csv"


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Phase 6 random-policy rollout for GazeboChaseEnv.")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reset-mode", default="none", choices=["none", "soft", "hard", "episode_soft"])
    parser.add_argument("--world-type", default="woods_easy")
    parser.add_argument("--action-scale", type=float, default=0.5)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--config", default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no-settle-start-height", action="store_true")
    parser.add_argument("--settle-min-height", type=float, default=0.8)
    parser.add_argument("--settle-max-height", type=float, default=2.2)
    parser.add_argument("--settle-timeout", type=float, default=25.0)
    return parser


def load_env_config(path):
    if not path:
        return {}
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read config files") from exc
    with open(path, "r") as handle:
        data = yaml.safe_load(handle) or {}
    return dict(data.get("env", {}))


def publish_normalized_action(env, action):
    _, body_cmd = env._map_action(action)
    env._publish_body_cmd(body_cmd)
    env.last_body_cmd = body_cmd.copy()


def settle_start_height(env, min_height, max_height, timeout):
    deadline = time.monotonic() + float(timeout)
    zero = np.zeros(4, dtype=np.float32)
    while time.monotonic() < deadline:
        obs = env._get_obs()
        drone_z = float(obs[14])
        if min_height <= drone_z <= max_height:
            for _ in range(5):
                publish_normalized_action(env, zero)
                time.sleep(env.step_dt)
            return True, drone_z

        action = zero.copy()
        action[2] = 0.8 if drone_z < min_height else -0.8
        publish_normalized_action(env, action)
        time.sleep(env.step_dt)

    return False, float(env._get_obs()[14])


def main():
    args = build_arg_parser().parse_args()
    rng = np.random.default_rng(args.seed)
    env_kwargs = load_env_config(args.config)
    env_kwargs.update(
        {
            "reset_mode": args.reset_mode,
            "world_type": args.world_type,
            "seed": args.seed,
        }
    )
    env = GazeboChaseEnv(**env_kwargs)
    rows = []
    try:
        total_step = 0
        episode = 0
        while total_step < args.steps:
            obs, info = env.reset(options={"reset_mode": args.reset_mode})
            print(
                "episode={} reset obs_shape={} reset_success={} topics_ready={} safety_mode={}".format(
                    episode,
                    obs.shape,
                    info.get("reset_success"),
                    info.get("topics_ready"),
                    info.get("safety_mode"),
                )
            )
            if not info.get("reset_success", False):
                raise RuntimeError("Reset failed for episode {}: {}".format(episode, info))
            if not args.no_settle_start_height:
                settled, drone_z = settle_start_height(
                    env,
                    args.settle_min_height,
                    args.settle_max_height,
                    args.settle_timeout,
                )
                print("episode={} settle_start_height settled={} drone_z={:.3f}".format(episode, settled, drone_z))

            episode_step = 0
            while total_step < args.steps:
                action = float(args.action_scale) * rng.uniform(-1.0, 1.0, size=4).astype(np.float32)
                obs, reward, terminated, truncated, info = env.step(action)
                if obs.shape != (20,) or not np.all(np.isfinite(obs)) or not np.isfinite(reward):
                    raise RuntimeError("Invalid env output at step {}".format(total_step))
                rows.append(
                    {
                        "step": total_step,
                        "episode": episode,
                        "episode_step": episode_step,
                        "reward": reward,
                        "terminated": terminated,
                        "truncated": truncated,
                        "terminal_reason": info["terminal_reason"],
                        "success": info["success"],
                        "target_visible": info["target_visible"],
                        "target_distance": info["target_distance"],
                        "target_u": info["target_u"],
                        "target_v": info["target_v"],
                        "front_q05_depth": info["front_q05_depth"],
                        "obstacle_area_ratio": info["obstacle_area_ratio"],
                        "drone_z": info["drone_z"],
                        "safety_mode": info["safety_mode"],
                        "emergency_count": info["emergency_count"],
                        "depth_stop_count": info["depth_stop_count"],
                        "target_lost_count": info["target_lost_count"],
                        "action_vx": action[0],
                        "action_vy": action[1],
                        "action_vz": action[2],
                        "action_yaw": action[3],
                    }
                )
                if not args.quiet:
                    print(
                        "step={} episode={} episode_step={} reward={:.3f} terminated={} truncated={} visible={} "
                        "distance={:.3f} front_q05={:.3f} drone_z={:.3f} mode={} terminal_reason={}".format(
                            total_step,
                            episode,
                            episode_step,
                            reward,
                            terminated,
                            truncated,
                            info["target_visible"],
                            info["target_distance"],
                            info["front_q05_depth"],
                            info["drone_z"],
                            info["safety_mode"],
                            info["terminal_reason"],
                        )
                    )
                total_step += 1
                episode_step += 1
                if terminated or truncated:
                    break
            episode += 1
    finally:
        env.close()

    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fieldnames = [
        "step",
        "episode",
        "episode_step",
        "reward",
        "terminated",
        "truncated",
        "terminal_reason",
        "success",
        "target_visible",
        "target_distance",
        "target_u",
        "target_v",
        "front_q05_depth",
        "obstacle_area_ratio",
        "drone_z",
        "safety_mode",
        "emergency_count",
        "depth_stop_count",
        "target_lost_count",
        "action_vx",
        "action_vy",
        "action_vz",
        "action_yaw",
    ]
    with open(output_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print("wrote {}".format(output_path))


if __name__ == "__main__":
    main()
