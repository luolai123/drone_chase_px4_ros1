#!/usr/bin/env python3

import argparse
import csv
import json
import os
import shutil
import sys
import time


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "envs"))
if ENV_DIR not in sys.path:
    sys.path.insert(0, ENV_DIR)

INSTALL_CMD = (
    "/usr/bin/python3 -m pip install --user "
    "stable-baselines3==2.3.2 gymnasium==0.29.1 'torch<2.5' tensorboard pandas"
)
REQUIRED_TOPICS = [
    "/target/state",
    "/obstacle/risk",
    "/mavros/local_position/pose",
    "/mavros/local_position/velocity_local",
    "/mavros/state",
    "/safety_filter/mode",
]


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


def dump_yaml(data, path):
    import yaml
    with open(path, "w") as handle:
        yaml.safe_dump(data, handle, default_flow_style=False, sort_keys=False)


def env_kwargs_from_config(config, seed=None):
    env_cfg = dict(config.get("env", {}))
    woods_cfg = config.get("woods", {})
    kwargs = dict(env_cfg)
    kwargs.setdefault("reset_mode", "episode_soft")
    if seed is not None:
        kwargs["seed"] = seed
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


def wait_for_required_topics(rospy, timeout=8.0):
    deadline = time.monotonic() + timeout
    missing = REQUIRED_TOPICS[:]
    while time.monotonic() < deadline and not rospy.is_shutdown():
        published = {name for name, _type in rospy.get_published_topics()}
        missing = [topic for topic in REQUIRED_TOPICS if topic not in published]
        if not missing:
            return
        time.sleep(0.25)
    raise RuntimeError(
        "Missing required ROS topics: {}. Start Gazebo/PX4, phase3_perception.launch, "
        "and phase6_rl_safety.launch before training.".format(", ".join(missing))
    )


class EpisodeCsvCallback:
    def __init__(self, output_dir, save_freq):
        from stable_baselines3.common.callbacks import BaseCallback

        class _Callback(BaseCallback):
            def __init__(self, outer):
                super().__init__()
                self.outer = outer

            def _on_training_start(self):
                self.outer.on_training_start(self.model)

            def _on_step(self):
                return self.outer.on_step(self)

        self.callback_cls = _Callback
        self.output_dir = output_dir
        self.save_freq = int(save_freq)
        self.csv_path = os.path.join(output_dir, "training_log.csv")
        self.checkpoint_dir = os.path.join(output_dir, "checkpoints")
        self.episode = 0
        self.reset_episode_accumulators()

    def make_callback(self):
        return self.callback_cls(self)

    def reset_episode_accumulators(self):
        self.episode_reward = 0.0
        self.episode_length = 0
        self.target_visible_count = 0
        self.front_q05_sum = 0.0
        self.emergency_count = 0
        self.depth_stop_count = 0
        self.target_lost_count = 0
        self.final_target_distance = 0.0
        self.terminated_reason = ""

    def on_training_start(self, model):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        with open(self.csv_path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames())
            writer.writeheader()
        self.model = model

    def fieldnames(self):
        return [
            "step",
            "episode",
            "episode_reward",
            "episode_length",
            "success",
            "terminated_reason",
            "final_target_distance",
            "target_visible_ratio",
            "mean_front_q05_depth",
            "emergency_count",
            "depth_stop_count",
            "target_lost_count",
        ]

    def on_step(self, callback):
        rewards = callback.locals.get("rewards", [0.0])
        dones = callback.locals.get("dones", [False])
        infos = callback.locals.get("infos", [{}])
        info = infos[0]
        reward = float(rewards[0])

        self.episode_reward += reward
        self.episode_length += 1
        self.target_visible_count += int(bool(info.get("target_visible", False)))
        self.front_q05_sum += float(info.get("front_q05_depth", 0.0))
        mode = str(info.get("safety_mode", ""))
        self.emergency_count += int("EMERGENCY_AVOID" in mode)
        self.depth_stop_count += int("DEPTH_STOP" in mode)
        self.target_lost_count += int("TARGET_LOST" in mode)
        self.final_target_distance = float(info.get("target_distance", 0.0))
        self.terminated_reason = str(info.get("terminal_reason", ""))

        if self.save_freq > 0 and callback.num_timesteps % self.save_freq == 0:
            self.model.save(os.path.join(self.checkpoint_dir, "ppo_step_{}".format(callback.num_timesteps)))
            callback.training_env.save(
                os.path.join(self.checkpoint_dir, "vecnormalize_step_{}.pkl".format(callback.num_timesteps))
            )

        if bool(dones[0]):
            self.write_episode(callback.num_timesteps, info)
            self.episode += 1
            self.reset_episode_accumulators()
        return True

    def write_episode(self, step, info):
        length = max(1, self.episode_length)
        row = {
            "step": step,
            "episode": self.episode,
            "episode_reward": self.episode_reward,
            "episode_length": self.episode_length,
            "success": bool(info.get("success", False)),
            "terminated_reason": self.terminated_reason or "unknown",
            "final_target_distance": self.final_target_distance,
            "target_visible_ratio": float(self.target_visible_count) / float(length),
            "mean_front_q05_depth": self.front_q05_sum / float(length),
            "emergency_count": self.emergency_count,
            "depth_stop_count": self.depth_stop_count,
            "target_lost_count": self.target_lost_count,
        }
        with open(self.csv_path, "a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames())
            writer.writerow(row)


class PeriodicEvalCallback:
    def __init__(self, output_dir, eval_env, eval_freq, episodes, deterministic):
        from stable_baselines3.common.callbacks import BaseCallback

        class _Callback(BaseCallback):
            def __init__(self, outer):
                super().__init__()
                self.outer = outer

            def _on_training_start(self):
                self.outer.on_training_start()

            def _on_step(self):
                return self.outer.on_step(self)

        self.callback_cls = _Callback
        self.output_dir = output_dir
        self.eval_env = eval_env
        self.eval_freq = int(eval_freq)
        self.episodes = int(episodes)
        self.deterministic = bool(deterministic)
        self.rollout_csv = os.path.join(output_dir, "eval_rollouts.csv")
        self.summary_path = os.path.join(output_dir, "eval_summary.json")
        self.summary_history_path = os.path.join(output_dir, "eval_summary_history.jsonl")
        self.last_eval_step = None

    def make_callback(self):
        return self.callback_cls(self)

    def on_training_start(self):
        with open(self.rollout_csv, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.rollout_fieldnames())
            writer.writeheader()
        with open(self.summary_history_path, "w") as handle:
            handle.write("")

    def rollout_fieldnames(self):
        return [
            "eval_step",
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
            "emergency",
            "depth_stop",
            "raw_timeout",
            "height_violation",
            "action_vx",
            "action_vy",
            "action_vz",
            "action_yaw",
            "drone_z",
        ]

    def on_step(self, callback):
        if self.eval_freq <= 0:
            return True
        if callback.num_timesteps > 0 and callback.num_timesteps % self.eval_freq == 0:
            self.run_eval(callback.model, callback.training_env, callback.num_timesteps)
        return True

    def run_eval(self, model, training_env, eval_step):
        from stable_baselines3.common.vec_env import sync_envs_normalization

        sync_envs_normalization(training_env, self.eval_env)
        self.eval_env.training = False
        self.eval_env.norm_reward = False

        episode_summaries = []
        rollout_rows = []
        for episode in range(self.episodes):
            obs = self.eval_env.reset()
            total_reward = 0.0
            visible_count = 0
            emergency_count = 0
            depth_stop_count = 0
            raw_timeout_count = 0
            height_violation_count = 0
            min_target_distance = float("inf")
            final_target_distance = float("inf")
            terminated_reason = ""
            done = False
            step = 0
            while not done:
                action, _state = model.predict(obs, deterministic=self.deterministic)
                obs, rewards, dones, infos = self.eval_env.step(action)
                info = infos[0]
                reward = float(rewards[0])
                done = bool(dones[0])
                total_reward += reward
                visible_count += int(bool(info.get("target_visible", False)))
                final_target_distance = float(info.get("target_distance", final_target_distance))
                min_target_distance = min(min_target_distance, final_target_distance)
                mode = str(info.get("safety_mode", ""))
                emergency = "EMERGENCY_AVOID" in mode
                depth_stop = "DEPTH_STOP" in mode
                raw_timeout = "RAW_TIMEOUT" in mode
                height_violation = bool(info.get("height_violation", False))
                emergency_count += int(emergency)
                depth_stop_count += int(depth_stop)
                raw_timeout_count += int(raw_timeout)
                height_violation_count += int(height_violation)
                terminated_reason = str(info.get("terminal_reason", ""))
                action_row = action[0] if len(action.shape) > 1 else action
                rollout_rows.append(
                    {
                        "eval_step": int(eval_step),
                        "episode": episode,
                        "step": step,
                        "reward": reward,
                        "done": done,
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
                        "emergency": emergency,
                        "depth_stop": depth_stop,
                        "raw_timeout": raw_timeout,
                        "height_violation": height_violation,
                        "action_vx": float(action_row[0]),
                        "action_vy": float(action_row[1]),
                        "action_vz": float(action_row[2]),
                        "action_yaw": float(action_row[3]),
                        "drone_z": float(info.get("drone_z", 0.0)),
                    }
                )
                step += 1

            episode_summaries.append(
                {
                    "episode": episode,
                    "success": terminated_reason == "success",
                    "timeout": terminated_reason == "timeout",
                    "reward": total_reward,
                    "length": step,
                    "terminated_reason": terminated_reason or "unknown",
                    "final_target_distance": final_target_distance,
                    "min_target_distance": min_target_distance,
                    "target_visible_ratio": float(visible_count) / float(max(1, step)),
                    "emergency_count": emergency_count,
                    "depth_stop_count": depth_stop_count,
                    "raw_timeout_count": raw_timeout_count,
                    "height_violation_count": height_violation_count,
                }
            )

        with open(self.rollout_csv, "a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.rollout_fieldnames())
            writer.writerows(rollout_rows)

        summary = summarize_eval_rows(eval_step, episode_summaries)
        with open(self.summary_path, "w") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
        with open(self.summary_history_path, "a") as handle:
            handle.write(json.dumps(summary, sort_keys=True) + "\n")
        self.last_eval_step = int(eval_step)
        print("eval_step={} summary={}".format(eval_step, summary))
        return summary


def mean(values):
    values = list(values)
    if not values:
        return 0.0
    return float(sum(values)) / float(len(values))


def summarize_eval_rows(eval_step, rows):
    return {
        "eval_step": int(eval_step),
        "episodes": len(rows),
        "success_rate": mean(1.0 if row["success"] else 0.0 for row in rows),
        "timeout_rate": mean(1.0 if row["timeout"] else 0.0 for row in rows),
        "reward_mean": mean(row["reward"] for row in rows),
        "length_mean": mean(row["length"] for row in rows),
        "target_visible_ratio_mean": mean(row["target_visible_ratio"] for row in rows),
        "final_distance_mean": mean(row["final_target_distance"] for row in rows),
        "min_distance_mean": mean(row["min_target_distance"] for row in rows),
        "emergency_count_total": int(sum(row["emergency_count"] for row in rows)),
        "depth_stop_count_total": int(sum(row["depth_stop_count"] for row in rows)),
        "raw_timeout_count_total": int(sum(row["raw_timeout_count"] for row in rows)),
        "height_violation_count_total": int(sum(row["height_violation_count"] for row in rows)),
        "reasons": {
            reason: sum(1 for row in rows if row["terminated_reason"] == reason)
            for reason in sorted({row["terminated_reason"] for row in rows})
        },
    }


def summarize_training(output_dir, total_timesteps, config):
    path = os.path.join(output_dir, "training_log.csv")
    rows = []
    if os.path.exists(path):
        with open(path, "r") as handle:
            rows = list(csv.DictReader(handle))
    summary = {
        "total_timesteps": int(total_timesteps),
        "episodes": len(rows),
        "config": config,
    }
    if rows:
        summary.update(
            {
                "success_rate": mean(
                    1.0 if str(row.get("success", "")).lower() == "true" else 0.0
                    for row in rows
                ),
                "episode_reward_mean": mean(float(row.get("episode_reward", 0.0)) for row in rows),
                "episode_length_mean": mean(float(row.get("episode_length", 0.0)) for row in rows),
                "final_distance_mean": mean(float(row.get("final_target_distance", 0.0)) for row in rows),
                "target_visible_ratio_mean": mean(float(row.get("target_visible_ratio", 0.0)) for row in rows),
                "emergency_count_total": int(sum(int(float(row.get("emergency_count", 0))) for row in rows)),
                "depth_stop_count_total": int(sum(int(float(row.get("depth_stop_count", 0))) for row in rows)),
                "target_lost_count_total": int(sum(int(float(row.get("target_lost_count", 0))) for row in rows)),
            }
        )
    with open(os.path.join(output_dir, "train_summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    return summary


def write_training_curve(output_dir):
    path = os.path.join(output_dir, "training_log.csv")
    if not os.path.exists(path):
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    with open(path, "r") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return
    episodes = [int(float(row["episode"])) for row in rows]
    rewards = [float(row["episode_reward"]) for row in rows]
    final_distances = [float(row["final_target_distance"]) for row in rows]
    fig = plt.figure(figsize=(9, 5))
    ax1 = fig.add_subplot(111)
    ax1.plot(episodes, rewards, label="episode_reward", color="#1f77b4")
    ax1.set_xlabel("episode")
    ax1.set_ylabel("episode_reward")
    ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(episodes, final_distances, label="final_distance", color="#d62728", alpha=0.8)
    ax2.set_ylabel("final_target_distance")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "training_curve.png"))
    plt.close(fig)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Train low-dimensional PPO on GazeboChaseEnv.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume-model", default=None)
    parser.add_argument("--resume-vecnormalize", default=None)
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser


def main():
    args = build_arg_parser().parse_args()
    require_training_deps()

    import rospy
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CallbackList
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from gazebo_chase_env import GazeboChaseEnv

    config = load_yaml(args.config)
    output_dir = args.output_dir or config.get("save", {}).get("output_dir", "/home/whk/vf_ws/outputs/phase7/run")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "monitor"), exist_ok=True)
    shutil.copyfile(args.config, os.path.join(output_dir, "config_used.yaml"))

    if args.total_timesteps is not None:
        config.setdefault("ppo", {})["total_timesteps"] = int(args.total_timesteps)
    if args.seed is not None:
        config["seed_override"] = int(args.seed)
    dump_yaml(config, os.path.join(output_dir, "config_effective.yaml"))

    env_kwargs = env_kwargs_from_config(config, seed=args.seed)

    def make_env():
        env = GazeboChaseEnv(**env_kwargs)
        return Monitor(env, filename=os.path.join(output_dir, "monitor", "monitor.csv"))

    vec_env = DummyVecEnv([make_env])
    wait_for_required_topics(rospy)

    if args.resume_vecnormalize:
        vec_env = VecNormalize.load(args.resume_vecnormalize, vec_env)
        vec_env.training = True
        vec_env.norm_reward = True
    else:
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    os.makedirs(os.path.join(output_dir, "eval_monitor"), exist_ok=True)

    def make_eval_env():
        env = GazeboChaseEnv(**env_kwargs)
        return Monitor(env, filename=os.path.join(output_dir, "eval_monitor", "monitor.csv"))

    eval_vec_env = DummyVecEnv([make_eval_env])
    eval_vec_env = VecNormalize(eval_vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    eval_vec_env.training = False
    eval_vec_env.norm_reward = False

    ppo_cfg = dict(config.get("ppo", {}))
    total_timesteps = int(ppo_cfg.pop("total_timesteps", 10000))
    save_freq = int(config.get("save", {}).get("save_freq", 5000))
    eval_cfg = dict(config.get("eval", {}))
    eval_episodes = int(eval_cfg.get("eval_episodes", 5))
    eval_freq = int(eval_cfg.get("eval_freq", save_freq))
    eval_deterministic = bool(eval_cfg.get("deterministic", True))

    if args.resume_model:
        model = PPO.load(args.resume_model, env=vec_env, seed=args.seed)
    else:
        model = PPO("MlpPolicy", vec_env, verbose=1, seed=args.seed, **ppo_cfg)

    episode_logger = EpisodeCsvCallback(output_dir, save_freq)
    eval_logger = PeriodicEvalCallback(
        output_dir,
        eval_vec_env,
        eval_freq,
        eval_episodes,
        eval_deterministic,
    )
    callbacks = CallbackList([episode_logger.make_callback(), eval_logger.make_callback()])
    try:
        model.learn(total_timesteps=total_timesteps, callback=callbacks, reset_num_timesteps=False)
        model.save(os.path.join(output_dir, "ppo_model_final.zip"))
        model.save(os.path.join(output_dir, "model.zip"))
        vec_env.save(os.path.join(output_dir, "vecnormalize.pkl"))
        monitor_src = os.path.join(output_dir, "monitor", "monitor.csv")
        monitor_dst = os.path.join(output_dir, "monitor.csv")
        if os.path.exists(monitor_src):
            shutil.copyfile(monitor_src, monitor_dst)
        train_summary = summarize_training(output_dir, total_timesteps, config)
        write_training_curve(output_dir)
        print("train_summary={}".format(train_summary))
    finally:
        eval_vec_env.close()
        vec_env.close()
    print("saved model and VecNormalize to {}".format(output_dir))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("phase7_train_ppo_lowdim failed: {}".format(exc), file=sys.stderr)
        sys.exit(1)
