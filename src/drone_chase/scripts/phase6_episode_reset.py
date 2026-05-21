#!/usr/bin/env python3

import argparse
import os
import sys


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
    parser = argparse.ArgumentParser(description="Run one GazeboChaseEnv reset and print the result.")
    parser.add_argument("--reset-mode", default="soft", choices=["none", "soft", "hard"])
    parser.add_argument("--world-type", default="woods_easy")
    parser.add_argument("--seed", type=int, default=None)
    return parser


def main():
    args = build_arg_parser().parse_args()
    env = GazeboChaseEnv(reset_mode=args.reset_mode, world_type=args.world_type, seed=args.seed)
    try:
        obs, info = env.reset(options={"reset_mode": args.reset_mode})
        print(
            "reset_success={} topics_ready={} obs_shape={} target_visible={} "
            "target_distance={:.3f} front_q05_depth={:.3f} drone_z={:.3f} safety_mode={}".format(
                info.get("reset_success"),
                info.get("topics_ready"),
                obs.shape,
                info["target_visible"],
                info["target_distance"],
                info["front_q05_depth"],
                info["drone_z"],
                info["safety_mode"],
            )
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
