#!/usr/bin/env python3

import argparse
import os
import sys

import numpy as np


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


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Check GazeboChaseEnv compatibility before PPO training.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--action-scale", type=float, default=0.2)
    parser.add_argument("--random-actions", action="store_true")
    parser.add_argument("--run-sb3-check-env", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main():
    args = build_arg_parser().parse_args()
    require_training_deps()

    from gymnasium import spaces
    from gazebo_chase_env import GazeboChaseEnv

    config = load_yaml(args.config)
    rng = np.random.default_rng(args.seed)
    env = GazeboChaseEnv(**env_kwargs_from_config(config))
    try:
        if not isinstance(env.action_space, spaces.Box):
            raise RuntimeError("action_space must be gymnasium.spaces.Box for SB3")
        if not isinstance(env.observation_space, spaces.Box):
            raise RuntimeError("observation_space must be gymnasium.spaces.Box for SB3")
        print("action_space={} observation_space={}".format(env.action_space, env.observation_space))

        if args.run_sb3_check_env:
            from stable_baselines3.common.env_checker import check_env

            check_env(env, warn=True, skip_render_check=True)

        obs, info = env.reset()
        print("reset obs_shape={} info={}".format(obs.shape, info))
        if not bool(info.get("reset_success", False)):
            raise RuntimeError("reset_success=False info={}".format(info))
        if obs.shape != (20,):
            raise RuntimeError("Expected obs shape (20,), got {}".format(obs.shape))
        if not np.all(np.isfinite(obs)):
            raise RuntimeError("Reset observation contains NaN or inf")

        for step in range(args.steps):
            if args.random_actions:
                action = args.action_scale * rng.uniform(-1.0, 1.0, size=4).astype(np.float32)
            else:
                action = np.zeros(4, dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            if obs.shape != (20,):
                raise RuntimeError("Expected obs shape (20,), got {}".format(obs.shape))
            if not np.all(np.isfinite(obs)) or not np.isfinite(reward):
                raise RuntimeError("Non-finite env output at step {}".format(step))
            if bool(info.get("height_violation", False)):
                raise RuntimeError("height_violation at step {} info={}".format(step, info))
            print(
                "step={} z={:.3f} safety_mode={} terminated_reason={} "
                "reward={:.3f} terminated={} truncated={} target_visible={} "
                "target_distance={:.3f} front_q05_depth={:.3f} info={}".format(
                    step,
                    info["drone_z"],
                    info["safety_mode"],
                    info["terminal_reason"],
                    reward,
                    terminated,
                    truncated,
                    info["target_visible"],
                    info["target_distance"],
                    info["front_q05_depth"],
                    info,
                )
            )
            if terminated or truncated:
                break
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("phase7_check_training_env failed: {}".format(exc), file=sys.stderr)
        sys.exit(1)
