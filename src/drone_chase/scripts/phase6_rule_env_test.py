#!/usr/bin/env python3
"""Phase 6 rule-based environment test.

Runs 3 episodes with simple rule-based actions derived from obs (no
rule_based_chaser.py). Saves results to outputs/phase6/rule_env_test.csv.
"""

import csv
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

CSV_PATH = "/home/whk/vf_ws/outputs/phase6/rule_env_test.csv"
NUM_EPISODES = 3
MAX_STEPS = 150


def rule_action(obs):
    """Generate rule-based action from observation vector.

    obs layout (20-dim):
      [0]  target_visible
      [4]  target_distance
      [5]  target_u
      [6]  target_v
    """
    visible = float(obs[0]) > 0.5
    if visible:
        target_u = float(obs[5])
        target_v = float(obs[6])
        target_distance = float(obs[4])
        a_yaw = np.clip(-0.8 * target_u / 0.6, -1.0, 1.0)
        a_vz = np.clip(-0.35 * target_v / 0.25, -1.0, 1.0)
        a_vx = np.clip(0.35 * (target_distance - 0.8) / 0.5, -1.0, 1.0)
        a_vy = 0.0
    else:
        a_vx = 0.0
        a_vy = 0.0
        a_vz = 0.0
        a_yaw = np.clip(0.3 / 0.6, -1.0, 1.0)
    return np.array([a_vx, a_vy, a_vz, a_yaw], dtype=np.float32)


def main():
    env = GazeboChaseEnv(reset_mode="soft", world_type="world_0")
    rows = []

    try:
        for ep in range(NUM_EPISODES):
            print("\n=== Episode {} ===".format(ep))
            obs, info = env.reset()
            print(
                "reset: shape={} success={} topics_ready={} safety_mode={}".format(
                    obs.shape,
                    info.get("reset_success"),
                    info.get("topics_ready"),
                    info.get("safety_mode"),
                )
            )

            total_reward = 0.0
            ep_rows = []
            for step in range(MAX_STEPS):
                action = rule_action(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward

                row = {
                    "episode": ep,
                    "step": step,
                    "reward": reward,
                    "total_reward": total_reward,
                    "terminated": terminated,
                    "truncated": truncated,
                    "success": info["success"],
                    "collision": info["collision"],
                    "timeout": info["timeout"],
                    "target_visible": info["target_visible"],
                    "target_depth": info["target_distance"],
                    "front_q05": info["front_q05_depth"],
                    "drone_z": info["drone_z"],
                    "safety_mode": info["safety_mode"],
                    "terminal_reason": info.get("terminal_reason", ""),
                }
                ep_rows.append(row)

                print(
                    "  step={} reward={:.3f} done={} visible={} depth={:.3f} "
                    "front_q05={:.3f} z={:.3f} mode={}".format(
                        step,
                        reward,
                        terminated or truncated,
                        info["target_visible"],
                        info["target_distance"],
                        info["front_q05_depth"],
                        info["drone_z"],
                        info["safety_mode"],
                    )
                )

                if terminated or truncated:
                    break

            rows.extend(ep_rows)
            last = ep_rows[-1] if ep_rows else {}
            print(
                "Episode {} summary: total_reward={:.3f} steps={} success={} "
                "collision={} timeout={}".format(
                    ep,
                    total_reward,
                    len(ep_rows),
                    last.get("success", False),
                    last.get("collision", False),
                    last.get("timeout", False),
                )
            )
    finally:
        env.close()

    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    fieldnames = [
        "episode", "step", "reward", "total_reward",
        "terminated", "truncated", "success", "collision", "timeout",
        "target_visible", "target_depth", "front_q05",
        "drone_z", "safety_mode", "terminal_reason",
    ]
    with open(CSV_PATH, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print("\nCSV saved: {}".format(CSV_PATH))


if __name__ == "__main__":
    main()
