#!/usr/bin/env python3

import argparse
import math
import os
import sys

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


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Phase 6 zero-action GazeboChaseEnv smoke test.")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--reset-mode", default="none", choices=["none", "soft", "hard"])
    parser.add_argument("--world-type", default="world_0")
    return parser


def main():
    args = build_arg_parser().parse_args()
    env = GazeboChaseEnv(reset_mode=args.reset_mode, world_type=args.world_type)
    try:
        obs, info = env.reset(options={"reset_mode": args.reset_mode} if args.reset_mode else None)
        print(
            "reset obs_shape={} reset_success={} topics_ready={} safety_mode={}".format(
                obs.shape,
                info.get("reset_success"),
                info.get("topics_ready"),
                info.get("safety_mode"),
            )
        )
        if obs.shape != (20,):
            raise RuntimeError("Expected obs shape (20,), got {}".format(obs.shape))
        if not np.all(np.isfinite(obs)):
            raise RuntimeError("Observation contains NaN or inf after reset")

        action = np.zeros(4, dtype=np.float32)
        for step in range(args.steps):
            obs, reward, terminated, truncated, info = env.step(action)
            if obs.shape != (20,):
                raise RuntimeError("Expected obs shape (20,), got {}".format(obs.shape))
            if not np.all(np.isfinite(obs)):
                raise RuntimeError("Observation contains NaN or inf at step {}".format(step))
            if not math.isfinite(float(reward)):
                raise RuntimeError("Reward is not finite at step {}".format(step))
            print(
                "step={} obs_shape={} reward={:.3f} terminated={} truncated={} "
                "target_visible={} target_distance={:.3f} front_q05_depth={:.3f} "
                "drone_z={:.3f} safety_mode={} terminal_reason={}".format(
                    step,
                    obs.shape,
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
            if terminated or truncated:
                break
    finally:
        env.close()


if __name__ == "__main__":
    main()
