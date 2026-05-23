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
DEFAULT_MODEL = os.path.join(REGISTRY_DIR, "best_world0_world1_policy.zip")
DEFAULT_VECNORMALIZE = os.path.join(REGISTRY_DIR, "best_world0_world1_vecnormalize.pkl")
DEFAULT_OUTPUT_DIR = "/home/whk/vf_ws/outputs/phase8/perception_feature_regression"
DEFAULT_CONFIG = (
    "/home/whk/vf_ws/outputs/phase7/world0_ppo_from_bc_v2_10k_conservative_run4/config_effective.yaml"
)


WORLD_MAP = {
    "world0": "world_0",
    "world1": "world_1",
    "woods_easy": "woods_easy",
    "random_woods": "random_woods",
}


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
        raise RuntimeError("Missing eval dependencies: {}".format(", ".join(missing)))


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


def env_kwargs_from_config(config, seed, world_type):
    env_cfg = dict(config.get("env", {}))
    env_cfg["world_type"] = str(world_type)
    env_cfg["reset_mode"] = "soft"
    env_cfg["respawn_target_on_reset"] = False
    env_cfg["seed"] = int(seed)
    env_cfg.setdefault("reset_ready_timeout", 20.0)
    env_cfg.setdefault("reset_zero_cmd_duration", 1.0)
    return {key: value for key, value in env_cfg.items() if value is not None}


class DebugTopicMonitor:
    def __init__(self):
        import rospy
        from std_msgs.msg import String

        self.lock = threading.Lock()
        self.depth_debug_count = 0
        self.target_debug_count = 0
        self.depth_debug_last = None
        self.target_debug_last = None

        self.subs = [
            rospy.Subscriber("/debug/depth_risk_extended", String, self._depth_cb, queue_size=10),
            rospy.Subscriber("/debug/target_detection_quality", String, self._target_cb, queue_size=10),
        ]

    def _depth_cb(self, msg):
        with self.lock:
            self.depth_debug_count += 1
            self.depth_debug_last = msg

    def _target_cb(self, msg):
        with self.lock:
            self.target_debug_count += 1
            self.target_debug_last = msg

    def snapshot(self):
        with self.lock:
            return {
                "depth_debug_count": int(self.depth_debug_count),
                "target_debug_count": int(self.target_debug_count),
            }

    def close(self):
        for sub in self.subs:
            try:
                sub.unregister()
            except Exception:
                pass


def rollout_fieldnames():
    return [
        "world",
        "episode",
        "seed",
        "step",
        "target_visible",
        "target_distance",
        "target_u",
        "target_v",
        "front_q05_depth",
        "obstacle_area_ratio",
        "obstacle_danger",
        "drone_z",
        "safety_mode",
        "mavros_mode",
        "mavros_armed",
        "raw_timeout_step",
        "offboard_step",
        "reward",
        "done",
        "terminal_reason",
        "debug_depth_count",
        "debug_target_count",
    ]


def episode_fieldnames():
    return [
        "world",
        "episode",
        "seed",
        "success",
        "timeout",
        "collision",
        "out_of_bounds",
        "height_violation",
        "terminal_reason",
        "episode_length",
        "final_distance",
        "min_distance",
        "target_visible_ratio",
        "raw_timeout_steps",
        "offboard_steps",
        "debug_depth_msgs",
        "debug_target_msgs",
    ]


def run_eval_for_world(raw_env, vec_env, model, monitor, world_name, world_type, episodes, seed_base, deterministic):
    episode_rows = []
    rollout_rows = []

    for episode in range(int(episodes)):
        debug_counts_start = monitor.snapshot()
        seed = int(seed_base) + int(episode)
        raw_env.config["seed"] = int(seed)
        raw_env.config["world_type"] = str(world_type)
        raw_env.world_type = str(world_type)
        obs, info = raw_env.reset(seed=seed, options={"reset_mode": "soft"})

        total_steps = 0
        visible_steps = 0
        raw_timeout_steps = 0
        offboard_steps = 0
        min_distance = float("inf")
        final_distance = float("inf")
        terminal_reason = ""

        while True:
            norm_obs = vec_env.normalize_obs(np.asarray([obs], dtype=np.float32))
            action, _ = model.predict(norm_obs, deterministic=bool(deterministic))
            action_arr = np.asarray(action)
            action_row = action_arr[0] if action_arr.ndim > 1 else action_arr
            obs, reward, terminated, truncated, info = raw_env.step(action_row)
            reward = float(reward)
            done = bool(terminated or truncated)

            target_visible = bool(info.get("target_visible", False))
            visible_steps += int(target_visible)
            distance = finite(info.get("target_distance", math.nan))
            if math.isfinite(distance):
                final_distance = distance
                min_distance = min(min_distance, distance)

            safety_mode = str(info.get("safety_mode", "") or "")
            if "RAW_TIMEOUT" in safety_mode:
                raw_timeout_steps += 1
            mavros_mode = str(info.get("mavros_mode", "") or "")
            mavros_armed = bool(info.get("mavros_armed", False))
            if mavros_armed and mavros_mode != "OFFBOARD":
                offboard_steps += 1

            if done:
                terminal_reason = str(info.get("terminal_reason", "") or "")

            snap = monitor.snapshot()
            rollout_rows.append(
                {
                    "world": world_name,
                    "episode": int(episode),
                    "seed": int(seed),
                    "step": int(total_steps),
                    "target_visible": int(target_visible),
                    "target_distance": distance,
                    "target_u": finite(info.get("target_u", math.nan)),
                    "target_v": finite(info.get("target_v", math.nan)),
                    "front_q05_depth": finite(info.get("front_q05_depth", math.nan)),
                    "obstacle_area_ratio": finite(info.get("obstacle_area_ratio", math.nan)),
                    "obstacle_danger": int(bool(info.get("obstacle_danger", False))),
                    "drone_z": finite(info.get("drone_z", math.nan)),
                    "safety_mode": safety_mode,
                    "mavros_mode": mavros_mode,
                    "mavros_armed": int(bool(mavros_armed)),
                    "raw_timeout_step": int("RAW_TIMEOUT" in safety_mode),
                    "offboard_step": int(mavros_armed and mavros_mode != "OFFBOARD"),
                    "reward": reward,
                    "done": int(done),
                    "terminal_reason": terminal_reason,
                    "debug_depth_count": int(snap["depth_debug_count"]),
                    "debug_target_count": int(snap["target_debug_count"]),
                }
            )

            total_steps += 1
            if done:
                break

        snap_end = monitor.snapshot()
        debug_depth_msgs = snap_end["depth_debug_count"] - debug_counts_start["depth_debug_count"]
        debug_target_msgs = snap_end["target_debug_count"] - debug_counts_start["target_debug_count"]
        episode_rows.append(
            {
                "world": world_name,
                "episode": int(episode),
                "seed": int(seed),
                "success": int(bool(info.get("success", False))),
                "timeout": int(bool(info.get("timeout", False))),
                "collision": int(bool(info.get("collision", False))),
                "out_of_bounds": int(bool(info.get("out_of_bounds", False))),
                "height_violation": int(bool(info.get("height_violation", False))),
                "terminal_reason": terminal_reason,
                "episode_length": int(total_steps),
                "final_distance": finite(final_distance),
                "min_distance": finite(min_distance),
                "target_visible_ratio": float(visible_steps) / float(max(1, total_steps)),
                "raw_timeout_steps": int(raw_timeout_steps),
                "offboard_steps": int(offboard_steps),
                "debug_depth_msgs": int(debug_depth_msgs),
                "debug_target_msgs": int(debug_target_msgs),
            }
        )

    return episode_rows, rollout_rows


def summarize(episode_rows):
    if not episode_rows:
        return {}
    total = len(episode_rows)
    success = sum(int(r.get("success", 0)) for r in episode_rows)
    timeout = sum(int(r.get("timeout", 0)) for r in episode_rows)
    collision = sum(int(r.get("collision", 0)) for r in episode_rows)
    out_of_bounds = sum(int(r.get("out_of_bounds", 0)) for r in episode_rows)
    height_violation = sum(int(r.get("height_violation", 0)) for r in episode_rows)
    raw_timeout_steps = sum(int(r.get("raw_timeout_steps", 0)) for r in episode_rows)
    offboard_steps = sum(int(r.get("offboard_steps", 0)) for r in episode_rows)
    visible_ratio = mean([r.get("target_visible_ratio", 0.0) for r in episode_rows])
    final_distance = mean([r.get("final_distance", math.nan) for r in episode_rows])
    min_distance = mean([r.get("min_distance", math.nan) for r in episode_rows])
    debug_depth_msgs = sum(int(r.get("debug_depth_msgs", 0)) for r in episode_rows)
    debug_target_msgs = sum(int(r.get("debug_target_msgs", 0)) for r in episode_rows)
    reasons = Counter(str(r.get("terminal_reason", "")) for r in episode_rows)
    return {
        "episodes": int(total),
        "success_rate": rate(success, total),
        "timeout_rate": rate(timeout, total),
        "collision_rate": rate(collision, total),
        "out_of_bounds_rate": rate(out_of_bounds, total),
        "height_violation_rate": rate(height_violation, total),
        "target_visible_ratio_mean": float(visible_ratio),
        "final_distance_mean": float(final_distance),
        "min_distance_mean": float(min_distance),
        "raw_timeout_steps": int(raw_timeout_steps),
        "offboard_steps": int(offboard_steps),
        "debug_depth_msgs": int(debug_depth_msgs),
        "debug_target_msgs": int(debug_target_msgs),
        "terminal_reason_counts": dict(reasons),
    }


def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_report(path, summary, args):
    per_world = summary.get("per_world", {}) or {}
    world0 = per_world.get("world0", {}) or {}
    world1 = per_world.get("world1", {}) or {}
    woods_easy = per_world.get("woods_easy", {}) or {}
    random_woods = per_world.get("random_woods", {}) or {}

    raw_timeout_steps = 0
    offboard_steps = 0
    for row in per_world.values():
        raw_timeout_steps += int(row.get("raw_timeout_steps", 0))
        offboard_steps += int(row.get("offboard_steps", 0))

    depth_dbg_total = int((summary.get("debug", {}) or {}).get("depth_debug_msgs_total", 0))
    target_dbg_total = int((summary.get("debug", {}) or {}).get("target_debug_msgs_total", 0))

    content = """# Phase 8.1 Report: Perception Feature Enhancement Regression

Generated at: {now}

1. 是否修改 red_ball_detector.py：是（仅新增 debug/统计与 /debug/target_detection_quality）
2. 是否修改 depth_risk_estimator.py：是（仅新增 debug/统计与 /debug/depth_risk_extended；默认不改变 danger 行为）
3. 是否保持 /target/state 向后兼容：是（不改 TargetState.msg，不改字段语义）
4. 是否保持 /obstacle/risk 向后兼容：是（不改 DepthRisk.msg，不改字段语义；默认 danger 使用原 front_q05）
5. 是否新增 /debug/depth_risk_extended：是（std_msgs/String JSON；总消息数={depth_dbg_total}）
6. 是否新增 /debug/target_detection_quality：是（std_msgs/String JSON；总消息数={target_dbg_total}）
7. 是否新增 perception diagnostic recorder：是（scripts/phase8_perception_diagnostic_recorder.py）
8. world0 regression success rate：{w0_sr}
9. world1 regression success rate：{w1_sr}
10. woods_easy regression success rate：{we_sr}
11. random_woods regression success rate：{rw_sr}
12. 是否出现 ROS node crash：未在本脚本中强制检测（运行时需人工确认）
13. RAW_TIMEOUT 次数：{raw_timeout_steps}
14. OFFBOARD drop 次数：{offboard_steps}（按步统计，armed 且 mode!=OFFBOARD）
15. perception_diagnostic_log.csv 是否生成：本脚本不生成（请并行运行 phase8_perception_diagnostic_recorder.py）
16. extended debug topic 是否正常发布：{dbg_ok}
17. 是否影响 frozen policy 行为：默认不影响（debug 默认关闭；开启时仅附加 topic/统计）
18. 当前主要发现：待回归运行后填写（详见 phase8_perception_regression_summary.json）
19. 是否允许进入 Phase 8.2：perception feature ablation：取决于回归是否满足 success-rate/稳定性门槛
20. 是否允许训练新策略：否（Phase 8.1 明确不训练）

Evaluated model: {model}
Evaluated vecnormalize: {vecnormalize}
Config: {config}
Episodes per world: {episodes}
Deterministic: {deterministic}
Output dir: {output_dir}

## Raw summary (JSON)

```json
{summary_json}
```
""".format(
        now=datetime.now().isoformat(timespec="seconds"),
        depth_dbg_total=depth_dbg_total,
        target_dbg_total=target_dbg_total,
        w0_sr=world0.get("success_rate", 0.0),
        w1_sr=world1.get("success_rate", 0.0),
        we_sr=woods_easy.get("success_rate", 0.0),
        rw_sr=random_woods.get("success_rate", 0.0),
        raw_timeout_steps=raw_timeout_steps,
        offboard_steps=offboard_steps,
        dbg_ok="yes" if depth_dbg_total > 0 and target_dbg_total > 0 else "no_or_not_running",
        model=args.model,
        vecnormalize=args.vecnormalize,
        config=args.config,
        episodes=args.episodes_per_world,
        deterministic=bool(args.deterministic),
        output_dir=args.output_dir,
        summary_json=json.dumps(summary, indent=2, sort_keys=True),
    )
    with open(path, "w") as handle:
        handle.write(content)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Phase 8.1 regression eval (no training, perception debug on).")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--vecnormalize", default=DEFAULT_VECNORMALIZE)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--episodes-per-world", type=int, default=10)
    parser.add_argument("--seed-base", type=int, default=81000)
    parser.add_argument("--deterministic", dest="deterministic", action="store_true", default=True)
    parser.add_argument("--stochastic", dest="deterministic", action="store_false")
    parser.add_argument(
        "--worlds",
        default="world0,world1,woods_easy,random_woods",
        help="comma-separated: world0,world1,woods_easy,random_woods",
    )
    parser.add_argument(
        "--allow-inprocess-world-switch",
        action="store_true",
        help="unsafe debug option: do not use for final policy regression; Gazebo is not relaunched",
    )
    return parser


def main():
    args = build_arg_parser().parse_args()
    require_eval_deps()

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    import rospy
    from gazebo_chase_env import GazeboChaseEnv

    config = load_yaml(args.config)
    os.makedirs(args.output_dir, exist_ok=True)

    raw_env = GazeboChaseEnv(**env_kwargs_from_config(config, args.seed_base, WORLD_MAP["world0"]))
    vec_env = DummyVecEnv([lambda: raw_env])
    vec_env = VecNormalize.load(args.vecnormalize, vec_env)
    vec_env.training = False
    vec_env.norm_reward = False
    model = PPO.load(args.model)

    monitor = DebugTopicMonitor()

    all_episode_rows = []
    all_rollout_rows = []
    per_world_summary = {}

    worlds = [w.strip() for w in str(args.worlds).split(",") if w.strip()]
    if len(worlds) > 1 and not bool(args.allow_inprocess_world_switch):
        raise RuntimeError(
            "Multiple worlds require separate Gazebo/PX4 launches. "
            "Run this evaluator once per launched world and aggregate outputs, "
            "or pass --allow-inprocess-world-switch only for unsafe diagnostics."
        )
    try:
        for idx, world in enumerate(worlds):
            if world not in WORLD_MAP:
                raise RuntimeError("Unsupported world '{}'; expected one of {}".format(world, sorted(WORLD_MAP)))
            world_type = WORLD_MAP[world]
            seed_base = int(args.seed_base) + idx * 10000
            episode_rows, rollout_rows = run_eval_for_world(
                raw_env,
                vec_env,
                model,
                monitor,
                world,
                world_type,
                args.episodes_per_world,
                seed_base,
                args.deterministic,
            )
            all_episode_rows.extend(episode_rows)
            all_rollout_rows.extend(rollout_rows)
            per_world_summary[world] = summarize(episode_rows)
            print(
                "world={} episodes={} success_rate={:.3f} timeout_rate={:.3f} raw_timeout_steps={} offboard_steps={}".format(
                    world,
                    per_world_summary[world].get("episodes", 0),
                    per_world_summary[world].get("success_rate", 0.0),
                    per_world_summary[world].get("timeout_rate", 0.0),
                    per_world_summary[world].get("raw_timeout_steps", 0),
                    per_world_summary[world].get("offboard_steps", 0),
                )
            )
    finally:
        monitor.close()
        vec_env.close()

    debug_snapshot = monitor.snapshot()
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model_path": args.model,
        "vecnormalize_path": args.vecnormalize,
        "model_sha256": sha256(args.model) if os.path.exists(args.model) else "",
        "vecnormalize_sha256": sha256(args.vecnormalize) if os.path.exists(args.vecnormalize) else "",
        "episodes_per_world": int(args.episodes_per_world),
        "worlds": worlds,
        "per_world": per_world_summary,
        "debug": {
            "depth_debug_msgs_total": int(debug_snapshot.get("depth_debug_count", 0)),
            "target_debug_msgs_total": int(debug_snapshot.get("target_debug_count", 0)),
        },
    }

    rollout_csv = os.path.join(args.output_dir, "phase8_perception_regression_rollouts.csv")
    episode_csv = os.path.join(args.output_dir, "phase8_perception_regression_episodes.csv")
    summary_json = os.path.join(args.output_dir, "phase8_perception_regression_summary.json")
    report_md = os.path.join(args.output_dir, "phase8_1_report.md")

    write_csv(rollout_csv, rollout_fieldnames(), all_rollout_rows)
    write_csv(episode_csv, episode_fieldnames(), all_episode_rows)
    with open(summary_json, "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    write_report(report_md, summary, args)


if __name__ == "__main__":
    main()
