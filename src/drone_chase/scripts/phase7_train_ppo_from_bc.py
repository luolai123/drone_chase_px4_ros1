#!/usr/bin/env python3

import argparse
import csv
import json
import math
import os
import shutil
import sys
import time

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

MODEL_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "models"))
if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)

import phase7_train_ppo_lowdim as ppo_base
from bc_policy import load_bc_policy


DEFAULT_BC_MODEL = "/home/whk/vf_ws/outputs/phase7/bc_world0_dagger/bc_v2/bc_policy_best.pt"
DEFAULT_BC_OBS_NORM = "/home/whk/vf_ws/outputs/phase7/bc_world0_dagger/bc_v2/obs_norm_stats.npz"


def load_obs_norm_stats(path, obs_dim=20):
    if not path:
        raise ValueError("bc obs norm path is empty")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    data = np.load(path)
    mean = np.asarray(data["obs_mean"], dtype=np.float64).reshape(-1)
    std = np.asarray(data["obs_std"], dtype=np.float64).reshape(-1)
    if mean.shape[0] != obs_dim or std.shape[0] != obs_dim:
        raise ValueError("obs norm stats shape mismatch: mean={} std={}".format(mean.shape, std.shape))
    std = np.clip(std, 1.0e-6, None)
    return mean, std


def init_vecnormalize_from_obs_norm(vec_env, obs_mean, obs_std, count=1.0e6):
    vec_env.obs_rms.mean[...] = obs_mean
    vec_env.obs_rms.var[...] = obs_std ** 2.0
    vec_env.obs_rms.count = float(count)


def transfer_bc_to_ppo_actor(model, bc_checkpoint, copy_value_net=True, log_std_value=-2.0):
    import torch

    state = bc_checkpoint.get("model_state_dict", {})
    required = ["net.0.weight", "net.0.bias", "net.2.weight", "net.2.bias", "net.4.weight", "net.4.bias"]
    missing = [key for key in required if key not in state]
    if missing:
        raise ValueError("BC checkpoint missing keys: {}".format(", ".join(missing)))

    w1 = state["net.0.weight"]
    b1 = state["net.0.bias"]
    w2 = state["net.2.weight"]
    b2 = state["net.2.bias"]
    w3 = state["net.4.weight"]
    b3 = state["net.4.bias"]

    pi = model.policy.mlp_extractor.policy_net
    vf = model.policy.mlp_extractor.value_net

    with torch.no_grad():
        pi[0].weight.copy_(w1)
        pi[0].bias.copy_(b1)
        pi[2].weight.copy_(w2)
        pi[2].bias.copy_(b2)
        model.policy.action_net.weight.copy_(w3)
        model.policy.action_net.bias.copy_(b3)
        if copy_value_net:
            vf[0].weight.copy_(w1)
            vf[0].bias.copy_(b1)
            vf[2].weight.copy_(w2)
            vf[2].bias.copy_(b2)
        if hasattr(model.policy, "log_std") and model.policy.log_std is not None:
            model.policy.log_std.data.fill_(float(log_std_value))


def summarize_rollouts(rollout_rows, episodes):
    by_episode = {}
    for row in rollout_rows:
        by_episode.setdefault(int(row["episode"]), []).append(row)
    summaries = []
    for episode in range(int(episodes)):
        ep = by_episode.get(episode, [])
        if not ep:
            summaries.append(
                {
                    "episode": episode,
                    "success": False,
                    "timeout": False,
                    "length": 0,
                    "final_distance": math.nan,
                    "min_distance": math.nan,
                    "mean_reward": math.nan,
                    "target_visible_ratio": math.nan,
                }
            )
            continue
        distances = [float(r["target_distance"]) for r in ep if math.isfinite(float(r["target_distance"]))]
        rewards = [float(r["reward"]) for r in ep if math.isfinite(float(r["reward"]))]
        visible_ratio = float(sum(1 for r in ep if bool(r["target_visible"]))) / float(max(1, len(ep)))
        summaries.append(
            {
                "episode": episode,
                "success": any(bool(r["success"]) for r in ep),
                "timeout": any(bool(r["timeout"]) for r in ep),
                "length": int(len(ep)),
                "final_distance": distances[-1] if distances else math.nan,
                "min_distance": min(distances) if distances else math.nan,
                "mean_reward": float(np.mean(rewards)) if rewards else math.nan,
                "target_visible_ratio": visible_ratio,
            }
        )
    final_distances = [s["final_distance"] for s in summaries if isinstance(s["final_distance"], float) and math.isfinite(s["final_distance"])]
    min_distances = [s["min_distance"] for s in summaries if isinstance(s["min_distance"], float) and math.isfinite(s["min_distance"])]
    visible_ratios = [s["target_visible_ratio"] for s in summaries if isinstance(s["target_visible_ratio"], float) and math.isfinite(s["target_visible_ratio"])]
    mean_rewards = [s["mean_reward"] for s in summaries if isinstance(s["mean_reward"], float) and math.isfinite(s["mean_reward"])]
    summary = {
        "episodes": int(episodes),
        "rows": int(len(rollout_rows)),
        "success_rate": float(sum(1 for s in summaries if s["success"])) / float(max(1, episodes)),
        "timeout_rate": float(sum(1 for s in summaries if s["timeout"])) / float(max(1, episodes)),
        "target_visible_ratio": float(np.mean(visible_ratios)) if visible_ratios else math.nan,
        "final_distance_mean": float(np.mean(final_distances)) if final_distances else math.nan,
        "min_distance_mean": float(np.mean(min_distances)) if min_distances else math.nan,
        "mean_reward": float(np.mean(mean_rewards)) if mean_rewards else math.nan,
        "episode_summaries": summaries,
    }
    return summary


def write_csv(rows, path, fieldnames):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def pre_finetune_eval(output_dir, model, vec_env, bc_policy, episodes, deterministic=True):
    from stable_baselines3.common.vec_env import sync_envs_normalization

    eval_rollouts_path = os.path.join(output_dir, "pre_finetune_eval_rollouts.csv")
    eval_summary_path = os.path.join(output_dir, "pre_finetune_eval_summary.json")

    sync_envs_normalization(model.get_env(), vec_env)
    vec_env.training = False
    vec_env.norm_reward = False

    rollout_rows = []
    abs_err_sum = np.zeros((4,), dtype=np.float64)
    err_count = 0

    fieldnames = [
        "episode",
        "step",
        "reward",
        "done",
        "success",
        "timeout",
        "terminated_reason",
        "target_visible",
        "target_distance",
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
    ]

    for episode in range(int(episodes)):
        obs = vec_env.reset()
        done = False
        step = 0
        while not done:
            action, _state = model.predict(obs, deterministic=bool(deterministic))
            raw_obs = vec_env.get_original_obs()
            raw_obs_0 = raw_obs[0] if isinstance(raw_obs, np.ndarray) and raw_obs.ndim > 1 else raw_obs
            expert_action = bc_policy.predict_numpy(np.asarray(raw_obs_0, dtype=np.float32))
            action_vec = action[0] if hasattr(action, "ndim") and action.ndim > 1 else action
            abs_err_sum += np.abs(np.asarray(action_vec, dtype=np.float64) - np.asarray(expert_action, dtype=np.float64))
            err_count += 1

            obs, rewards, dones, infos = vec_env.step(action)
            info = infos[0]
            reward = float(rewards[0])
            done = bool(dones[0])
            terminated_reason = str(info.get("terminal_reason", ""))
            rollout_rows.append(
                {
                    "episode": episode,
                    "step": step,
                    "reward": reward,
                    "done": done,
                    "success": bool(info.get("success", False)),
                    "timeout": bool(info.get("timeout", False)),
                    "terminated_reason": terminated_reason,
                    "target_visible": bool(info.get("target_visible", False)),
                    "target_distance": float(info.get("target_distance", math.nan)),
                    "target_u": float(info.get("target_u", 0.0)),
                    "target_v": float(info.get("target_v", 0.0)),
                    "front_q05_depth": float(info.get("front_q05_depth", 0.0)),
                    "safety_mode": str(info.get("safety_mode", "")),
                    "mavros_mode": str(info.get("mavros_mode", "")),
                    "mavros_armed": bool(info.get("mavros_armed", False)),
                    "action_vx": float(action_vec[0]),
                    "action_vy": float(action_vec[1]),
                    "action_vz": float(action_vec[2]),
                    "action_yaw": float(action_vec[3]),
                    "drone_z": float(info.get("drone_z", 0.0)),
                }
            )
            step += 1

    write_csv(rollout_rows, eval_rollouts_path, fieldnames)
    summary = summarize_rollouts(rollout_rows, episodes)
    mae = (abs_err_sum / float(max(1, err_count))).astype(float).tolist()
    summary["bc_vs_initialized_ppo_action_mae"] = mae
    summary["bc_vs_initialized_ppo_action_mae_mean"] = float(np.mean(mae))
    with open(eval_summary_path, "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=False)
    return summary


def load_optional_json(path):
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r") as handle:
        try:
            return json.load(handle)
        except json.JSONDecodeError:
            return {}


def fmt(value):
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return "yes" if value else "no"
    try:
        v = float(value)
        return "{:.6g}".format(v) if math.isfinite(v) else "nan"
    except (TypeError, ValueError):
        return str(value)


def best_row_from_sweep_summary(sweep_summary):
    if not sweep_summary:
        return {}
    if "best_checkpoint" in sweep_summary:
        return sweep_summary.get("best_checkpoint", {}) or {}
    best_step = sweep_summary.get("best_checkpoint_step")
    for row in sweep_summary.get("rows", []) or []:
        if str(row.get("checkpoint_step")) == str(best_step):
            return row
    return {}


def final_row_from_sweep_summary(sweep_summary):
    if not sweep_summary:
        return {}
    if "final" in sweep_summary:
        return sweep_summary.get("final", {}) or {}
    for row in sweep_summary.get("rows", []) or []:
        if str(row.get("checkpoint_step")) == "final":
            return row
    return {}


def write_phase7_1g_report(output_dir, bc_eval_summary, init_eval_summary, train_eval_summary, sweep_summary):
    init_mae = init_eval_summary.get("bc_vs_initialized_ppo_action_mae", [math.nan] * 4) if init_eval_summary else [math.nan] * 4
    init_allows_training = bool(
        init_eval_summary
        and init_eval_summary.get("success_rate", 0.0) >= 0.2
        and init_eval_summary.get("target_visible_ratio", 0.0) >= 0.9
        and init_eval_summary.get("min_distance_mean", float("inf")) < 1.5
    )
    best_row = best_row_from_sweep_summary(sweep_summary)
    final_row = final_row_from_sweep_summary(sweep_summary)
    best_step = best_row.get("checkpoint_step")
    best_success = best_row.get("success_rate")
    final_success = final_row.get("success_rate")
    allow_72 = False
    allow_world1 = False
    best_policy = "BC v2"
    if best_row and float(best_row.get("success_rate", 0.0)) >= float(bc_eval_summary.get("success_rate", 0.0) if bc_eval_summary else 0.0):
        best_policy = "PPO fine-tuned"
    regression = "N/A"
    if best_row and bc_eval_summary:
        regression = "yes" if float(best_row.get("success_rate", 0.0)) < float(bc_eval_summary.get("success_rate", 0.0)) else "no"
    lines = [
        "# Phase 7.1G PPO Fine-tuning From BC v2 Report",
        "",
        "1. BC v2 baseline success rate：{}".format(fmt(bc_eval_summary.get("success_rate") if bc_eval_summary else None)),
        "2. 是否实现 PPO actor from BC 初始化：{}".format("yes"),
        "3. BC obs normalization 是否迁移到 PPO：{}".format("yes"),
        "4. 同一 obs 下 BC action 与 initialized PPO action MAE：{}".format(init_mae),
        "5. pre-finetune eval 是否完成：{}".format("yes" if init_eval_summary else "no"),
        "6. pre-finetune success rate：{}".format(fmt(init_eval_summary.get("success_rate") if init_eval_summary else None)),
        "7. pre-finetune target_visible_ratio：{}".format(fmt(init_eval_summary.get("target_visible_ratio") if init_eval_summary else None)),
        "8. pre-finetune final_distance_mean：{}".format(fmt(init_eval_summary.get("final_distance_mean") if init_eval_summary else None)),
        "9. pre-finetune min_distance_mean：{}".format(fmt(init_eval_summary.get("min_distance_mean") if init_eval_summary else None)),
        "10. 是否允许开始 PPO fine-tuning：{}".format("yes" if init_allows_training else "no"),
        "11. PPO fine-tuning 是否完成 10k：{}".format("yes" if train_eval_summary else "no"),
        "12. checkpoint sweep 是否完成：{}".format("yes" if sweep_summary else "no"),
        "13. best checkpoint step：{}".format(best_step if best_step is not None else "N/A"),
        "14. best checkpoint success rate：{}".format(fmt(best_success)),
        "15. final checkpoint success rate：{}".format(fmt(final_success)),
        "16. target_visible_ratio：{}".format(fmt(best_row.get("target_visible_ratio_mean"))),
        "17. final_distance_mean：{}".format(fmt(best_row.get("final_distance_mean"))),
        "18. min_distance_mean：{}".format(fmt(best_row.get("min_distance_mean"))),
        "19. RAW_TIMEOUT 次数：{}".format(fmt(best_row.get("raw_timeout_count"))),
        "20. emergency/depth_stop 次数：{}".format(
            "{}/{}".format(
                fmt(best_row.get("emergency_count")),
                fmt(best_row.get("depth_stop_count")),
            )
        ),
        "21. action_vx mean/std/min/max：{}".format(
            "{}/{}/{}/{}".format(
                fmt(best_row.get("action_vx_mean")),
                fmt(best_row.get("action_vx_std")),
                fmt(best_row.get("action_vx_min")),
                fmt(best_row.get("action_vx_max")),
            )
        ),
        "22. yaw_abs_mean：{}".format(fmt(best_row.get("yaw_abs_mean"))),
        "23. 是否出现策略退化：{}".format(regression),
        "24. 当前 best policy 是 BC v2 还是 PPO fine-tuned：{}".format(best_policy),
        "25. 是否允许进入 Phase 7.2：{}".format("yes" if allow_72 else "no"),
        "26. 是否允许进入 world1：{}".format("yes" if allow_world1 else "no"),
        "27. 如果不允许，下一步建议：{}".format(
            "先跑 checkpoint sweep 并确认 fine-tune 没破坏 BC，再考虑更小 lr/更少步数或冻结 actor 前层。"
        ),
        "",
    ]
    report_path = os.path.join(output_dir, "phase7_1g_report.md")
    with open(report_path, "w") as handle:
        handle.write("\n".join(lines))
    return report_path


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Phase 7.1G: PPO fine-tuning from BC v2.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--bc-model", default=DEFAULT_BC_MODEL)
    parser.add_argument("--bc-obs-norm", default=DEFAULT_BC_OBS_NORM)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--allow-train-on-prefail", action="store_true")
    parser.add_argument("--write-report-only", action="store_true")
    return parser


def main():
    args = build_arg_parser().parse_args()
    ppo_base.require_training_deps()

    import rospy
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CallbackList
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from gazebo_chase_env import GazeboChaseEnv

    config = ppo_base.load_yaml(args.config)
    output_dir = args.output_dir or config.get("save", {}).get("output_dir", "/home/whk/vf_ws/outputs/phase7/run")
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "monitor"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "eval_monitor"), exist_ok=True)
    shutil.copyfile(args.config, os.path.join(output_dir, "config_used.yaml"))

    if args.total_timesteps is not None:
        config.setdefault("ppo", {})["total_timesteps"] = int(args.total_timesteps)
    if args.seed is not None:
        config["seed_override"] = int(args.seed)
    ppo_base.dump_yaml(config, os.path.join(output_dir, "config_effective.yaml"))

    env_kwargs = ppo_base.env_kwargs_from_config(config, seed=args.seed)

    def make_env():
        env = GazeboChaseEnv(**env_kwargs)
        return Monitor(env, filename=os.path.join(output_dir, "monitor", "monitor.csv"))

    vec_env = DummyVecEnv([make_env])
    ppo_base.wait_for_required_topics(rospy)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    obs_mean, obs_std = load_obs_norm_stats(args.bc_obs_norm, obs_dim=int(config.get("env", {}).get("obs_dim", 20) or 20))
    init_vecnormalize_from_obs_norm(vec_env, obs_mean, obs_std, count=1.0e6)

    def make_eval_env():
        env = GazeboChaseEnv(**env_kwargs)
        return Monitor(env, filename=os.path.join(output_dir, "eval_monitor", "monitor.csv"))

    eval_vec_env = DummyVecEnv([make_eval_env])
    eval_vec_env = VecNormalize(eval_vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    init_vecnormalize_from_obs_norm(eval_vec_env, obs_mean, obs_std, count=1.0e6)
    eval_vec_env.training = False
    eval_vec_env.norm_reward = False

    ppo_cfg = dict(config.get("ppo", {}))
    total_timesteps = int(ppo_cfg.pop("total_timesteps", 10000))
    save_freq = int(config.get("save", {}).get("save_freq", 2500))
    eval_cfg = dict(config.get("eval", {}))
    eval_episodes = int(eval_cfg.get("eval_episodes", 5))
    eval_freq = int(eval_cfg.get("eval_freq", save_freq))
    eval_deterministic = bool(eval_cfg.get("deterministic", True))

    model = PPO("MlpPolicy", vec_env, verbose=1, seed=args.seed, **ppo_cfg)
    bc_policy, bc_ckpt = load_bc_policy(args.bc_model, device="cpu")
    if args.bc_obs_norm:
        import torch

        data = np.load(args.bc_obs_norm)
        bc_ckpt["obs_mean"] = np.asarray(data["obs_mean"], dtype=np.float32)
        bc_ckpt["obs_std"] = np.asarray(data["obs_std"], dtype=np.float32)
        bc_policy.obs_mean.data.copy_(torch.as_tensor(np.asarray(data["obs_mean"], dtype=np.float32)))
        bc_policy.obs_std.data.copy_(torch.as_tensor(np.asarray(np.maximum(data["obs_std"], 1.0e-6), dtype=np.float32)))

    log_std_init = -2.0
    policy_kwargs = ppo_cfg.get("policy_kwargs", {}) or {}
    if isinstance(policy_kwargs, dict) and "log_std_init" in policy_kwargs:
        try:
            log_std_init = float(policy_kwargs["log_std_init"])
        except (TypeError, ValueError):
            log_std_init = -2.0

    transfer_bc_to_ppo_actor(model, bc_ckpt, copy_value_net=True, log_std_value=log_std_init)
    model.save(os.path.join(output_dir, "initialized_ppo_model.zip"))
    vec_env.save(os.path.join(output_dir, "initialized_vecnormalize.pkl"))

    init_eval_summary = pre_finetune_eval(output_dir, model, eval_vec_env, bc_policy, eval_episodes, deterministic=eval_deterministic)

    pref_ok = (
        init_eval_summary.get("success_rate", 0.0) >= 0.2
        and init_eval_summary.get("target_visible_ratio", 0.0) >= 0.9
        and init_eval_summary.get("min_distance_mean", float("inf")) < 1.5
    )
    if not pref_ok and not args.allow_train_on_prefail:
        report_path = write_phase7_1g_report(
            output_dir,
            load_optional_json(os.path.join(os.path.dirname(args.bc_model), "bc_eval_summary.json")),
            init_eval_summary,
            {},
            {},
        )
        raise RuntimeError("pre-finetune eval failed gating, abort training. See {}".format(report_path))

    if args.skip_training:
        report_path = write_phase7_1g_report(
            output_dir,
            load_optional_json(os.path.join(os.path.dirname(args.bc_model), "bc_eval_summary.json")),
            init_eval_summary,
            {},
            {},
        )
        print("skip_training: wrote {}".format(report_path))
        return

    episode_logger = ppo_base.EpisodeCsvCallback(output_dir, save_freq)
    eval_logger = ppo_base.PeriodicEvalCallback(output_dir, eval_vec_env, eval_freq, eval_episodes, eval_deterministic)
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
        ppo_base.summarize_training(output_dir, total_timesteps, config)
        ppo_base.write_training_curve(output_dir)
    finally:
        eval_vec_env.close()
        vec_env.close()

    train_eval_summary = load_optional_json(os.path.join(output_dir, "eval_summary.json"))
    sweep_summary = load_optional_json(os.path.join(output_dir, "checkpoint_sweep_summary.json"))
    bc_eval_summary = load_optional_json(os.path.join(os.path.dirname(args.bc_model), "bc_eval_summary.json"))
    report_path = write_phase7_1g_report(output_dir, bc_eval_summary, init_eval_summary, train_eval_summary, sweep_summary)
    print("saved PPO-from-BC run to {}, report={}".format(output_dir, report_path))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("phase7_train_ppo_from_bc failed: {}".format(exc), file=sys.stderr)
        sys.exit(1)
