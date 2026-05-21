#!/usr/bin/env python3

import argparse
import csv
import hashlib
import json
import os
import shutil
from datetime import datetime


DEFAULT_REGISTRY_DIR = "/home/whk/vf_ws/outputs/final_policy_registry"
SOURCE_MODEL = (
    "/home/whk/vf_ws/outputs/phase7/"
    "world0_ppo_from_bc_v2_10k_conservative_run4/checkpoints/ppo_step_2500.zip"
)
SOURCE_VECNORMALIZE = (
    "/home/whk/vf_ws/outputs/phase7/"
    "world0_ppo_from_bc_v2_10k_conservative_run4/checkpoints/vecnormalize_step_2500.pkl"
)
PPO_RUN_DIR = "/home/whk/vf_ws/outputs/phase7/world0_ppo_from_bc_v2_10k_conservative_run4"
BC_V2_DIR = "/home/whk/vf_ws/outputs/phase7/bc_world0_dagger/bc_v2"
BC_SOURCE_DATASET = "/home/whk/vf_ws/outputs/phase7/bc_world0_dagger/bc_aggregated_dataset.npz"

DEFAULT_WORLD0_ROBUSTNESS = (
    "/home/whk/vf_ws/outputs/phase7/world0_best_policy_robustness/robustness_eval_summary.json"
)
DEFAULT_WORLD1_ZEROSHOT = (
    "/home/whk/vf_ws/outputs/phase7/world1_zeroshot_from_world0_best/world1_zeroshot_summary.json"
)
DEFAULT_WORLD1_ROBUSTNESS = (
    "/home/whk/vf_ws/outputs/phase7/world1_robustness_from_world0_best/world1_robustness_summary.json"
)
DEFAULT_CHECKPOINT_SWEEP = (
    "/home/whk/vf_ws/outputs/phase7/"
    "world0_ppo_from_bc_v2_10k_conservative_run4/checkpoint_sweep_summary.json"
)
DEFAULT_BC_V2 = "/home/whk/vf_ws/outputs/phase7/bc_world0_dagger/bc_v2/bc_eval_summary.json"
DEFAULT_BC_SUMMARY = "/home/whk/vf_ws/outputs/phase7/bc_world0_dagger/bc_v2/bc_summary.json"
DEFAULT_BC_AGGREGATED = "/home/whk/vf_ws/outputs/phase7/bc_world0_dagger/bc_aggregated_summary.json"
DEFAULT_CONFIG = (
    "/home/whk/vf_ws/outputs/phase7/world0_ppo_from_bc_v2_10k_conservative_run4/config_effective.yaml"
)

TABLE_FIELDS = [
    "stage",
    "world",
    "policy",
    "episodes",
    "success_rate",
    "timeout_rate",
    "collision_rate",
    "out_of_bounds_rate",
    "height_violation_rate",
    "target_visible_ratio",
    "final_distance_mean",
    "min_distance_mean",
    "raw_timeout_count",
    "emergency_count",
    "depth_stop_count",
    "offboard_drop_count",
    "notes",
]


def load_json(path):
    with open(path, "r") as handle:
        return json.load(handle)


def maybe_load_json(path):
    if not path or not os.path.exists(path):
        return {}
    return load_json(path)


def load_config(path):
    if not path or not os.path.exists(path):
        return {}
    try:
        import yaml
    except ImportError:
        return {}
    with open(path, "r") as handle:
        return yaml.safe_load(handle) or {}


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def copy_policy_artifact(src, dst):
    if not os.path.exists(src):
        raise RuntimeError("missing policy artifact: {}".format(src))
    ensure_dir(os.path.dirname(dst))
    shutil.copy2(src, dst)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fmt(value, digits=4):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return "{:.{}f}".format(value, digits)
    return str(value)


def pick(summary, keys, default=None):
    for key in keys:
        if key in summary and summary[key] is not None:
            return summary[key]
    return default


def rate_from_count(summary, count_key, episodes_key="episodes"):
    episodes = summary.get(episodes_key)
    count = summary.get(count_key)
    if episodes in (None, 0) or count is None:
        return None
    return float(count) / float(episodes)


def best_sweep_row(sweep):
    step = sweep.get("best_checkpoint_step")
    for row in sweep.get("rows", []):
        if row.get("checkpoint_step") == step:
            return row
    rows = sweep.get("rows", [])
    return rows[0] if rows else {}


def make_table_rows(world0, world1_zero, world1_robust, sweep, bc_v2):
    best_row = best_sweep_row(sweep)
    return [
        {
            "stage": "Phase 7.1F2 BC v2 online eval",
            "world": "world_0",
            "policy": "bc_v2",
            "episodes": fmt(bc_v2.get("episodes")),
            "success_rate": fmt(bc_v2.get("success_rate")),
            "timeout_rate": fmt(bc_v2.get("timeout_rate")),
            "collision_rate": fmt(pick(bc_v2, ["collision_rate"])),
            "out_of_bounds_rate": fmt(
                pick(bc_v2, ["out_of_bounds_rate"], rate_from_count(bc_v2, "out_of_bounds_count"))
            ),
            "height_violation_rate": fmt(
                pick(bc_v2, ["height_violation_rate"], rate_from_count(bc_v2, "height_violation_count"))
            ),
            "target_visible_ratio": fmt(bc_v2.get("target_visible_ratio")),
            "final_distance_mean": fmt(bc_v2.get("final_distance_mean")),
            "min_distance_mean": fmt(bc_v2.get("min_distance_mean")),
            "raw_timeout_count": fmt(bc_v2.get("raw_timeout_count")),
            "emergency_count": fmt(bc_v2.get("emergency_count")),
            "depth_stop_count": fmt(bc_v2.get("depth_stop_count")),
            "offboard_drop_count": "",
            "notes": "DAgger-lite + BC v2; PPO initialization source.",
        },
        {
            "stage": "Phase 7.1G checkpoint sweep",
            "world": "world_0",
            "policy": "ppo_step_{}".format(sweep.get("best_checkpoint_step", "unknown")),
            "episodes": fmt(best_row.get("episodes")),
            "success_rate": fmt(sweep.get("best_success_rate", best_row.get("success_rate"))),
            "timeout_rate": fmt(best_row.get("timeout_rate")),
            "collision_rate": fmt(best_row.get("collision_rate")),
            "out_of_bounds_rate": fmt(
                pick(best_row, ["out_of_bounds_rate"], rate_from_count(best_row, "out_of_bounds_count"))
            ),
            "height_violation_rate": fmt(
                pick(
                    best_row,
                    ["height_violation_rate"],
                    rate_from_count(best_row, "height_violation_count"),
                )
            ),
            "target_visible_ratio": fmt(
                sweep.get("best_target_visible_ratio_mean", best_row.get("target_visible_ratio_mean"))
            ),
            "final_distance_mean": fmt(
                sweep.get("best_final_distance_mean", best_row.get("final_distance_mean"))
            ),
            "min_distance_mean": fmt(
                sweep.get("best_min_distance_mean", best_row.get("min_distance_mean"))
            ),
            "raw_timeout_count": fmt(best_row.get("raw_timeout_count")),
            "emergency_count": fmt(best_row.get("emergency_count")),
            "depth_stop_count": fmt(best_row.get("depth_stop_count")),
            "offboard_drop_count": "",
            "notes": "Deterministic sweep selected ppo_step_2500; final model had slight regression.",
        },
        {
            "stage": "Phase 7.1H world0 robustness",
            "world": world0.get("world_type", "world_0"),
            "policy": "ppo_step_2500",
            "episodes": fmt(world0.get("episodes")),
            "success_rate": fmt(world0.get("success_rate")),
            "timeout_rate": fmt(world0.get("timeout_rate")),
            "collision_rate": fmt(
                pick(world0, ["collision_rate"], rate_from_count(world0, "collision_count"))
            ),
            "out_of_bounds_rate": fmt(world0.get("out_of_bounds_rate")),
            "height_violation_rate": fmt(world0.get("height_violation_rate")),
            "target_visible_ratio": fmt(world0.get("target_visible_ratio_mean")),
            "final_distance_mean": fmt(world0.get("final_distance_mean")),
            "min_distance_mean": fmt(world0.get("min_distance_mean")),
            "raw_timeout_count": fmt(world0.get("raw_timeout_count")),
            "emergency_count": fmt(world0.get("emergency_count")),
            "depth_stop_count": fmt(world0.get("depth_stop_count")),
            "offboard_drop_count": fmt(world0.get("offboard_drop_count")),
            "notes": "World0 robustness gate passed; reset pollution false.",
        },
        {
            "stage": "Phase 7.2A world1 zero-shot",
            "world": world1_zero.get("world", "world_1"),
            "policy": "ppo_step_2500",
            "episodes": fmt(world1_zero.get("episodes")),
            "success_rate": fmt(world1_zero.get("success_rate")),
            "timeout_rate": fmt(world1_zero.get("timeout_rate")),
            "collision_rate": fmt(world1_zero.get("collision_rate")),
            "out_of_bounds_rate": fmt(world1_zero.get("out_of_bounds_rate")),
            "height_violation_rate": fmt(world1_zero.get("height_violation_rate")),
            "target_visible_ratio": fmt(world1_zero.get("target_visible_ratio_mean")),
            "final_distance_mean": fmt(world1_zero.get("final_distance_mean")),
            "min_distance_mean": fmt(world1_zero.get("min_distance_mean")),
            "raw_timeout_count": fmt(world1_zero.get("raw_timeout_count")),
            "emergency_count": fmt(world1_zero.get("emergency_count")),
            "depth_stop_count": fmt(world1_zero.get("depth_stop_count")),
            "offboard_drop_count": fmt(world1_zero.get("offboard_drop_count")),
            "notes": "Zero-shot transfer to world1 sparse obstacles; 4 obstacles per episode.",
        },
        {
            "stage": "Phase 7.2B world1 robustness/stress",
            "world": world1_robust.get("world", "world_1"),
            "policy": "ppo_step_2500",
            "episodes": fmt(world1_robust.get("total_episodes")),
            "success_rate": fmt(world1_robust.get("total_success_rate")),
            "timeout_rate": fmt(world1_robust.get("timeout_rate")),
            "collision_rate": fmt(world1_robust.get("collision_rate")),
            "out_of_bounds_rate": fmt(world1_robust.get("out_of_bounds_rate")),
            "height_violation_rate": fmt(world1_robust.get("height_violation_rate")),
            "target_visible_ratio": fmt(world1_robust.get("target_visible_ratio_mean")),
            "final_distance_mean": fmt(world1_robust.get("final_distance_mean")),
            "min_distance_mean": fmt(world1_robust.get("min_distance_mean")),
            "raw_timeout_count": fmt(world1_robust.get("raw_timeout_count")),
            "emergency_count": fmt(world1_robust.get("emergency_count")),
            "depth_stop_count": fmt(world1_robust.get("depth_stop_count")),
            "offboard_drop_count": fmt(world1_robust.get("offboard_drop_count")),
            "notes": "Groups A/B/C/D passed; front-obstacle safety intervention passed.",
        },
    ]


def write_csv(path, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TABLE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_table(path, rows):
    with open(path, "w") as handle:
        handle.write("# Final Phase 7 Results Table\n\n")
        handle.write("| " + " | ".join(TABLE_FIELDS) + " |\n")
        handle.write("| " + " | ".join(["---"] * len(TABLE_FIELDS)) + " |\n")
        for row in rows:
            values = [str(row.get(field, "")).replace("|", "\\|") for field in TABLE_FIELDS]
            handle.write("| " + " | ".join(values) + " |\n")


def validation_summary(world0, world1_zero, world1_robust):
    return {
        "world0_robustness": {
            "episodes": world0.get("episodes"),
            "success_rate": world0.get("success_rate"),
            "target_visible_ratio_mean": world0.get("target_visible_ratio_mean"),
            "final_distance_mean": world0.get("final_distance_mean"),
            "min_distance_mean": world0.get("min_distance_mean"),
            "raw_timeout_count": world0.get("raw_timeout_count"),
            "emergency_count": world0.get("emergency_count"),
            "depth_stop_count": world0.get("depth_stop_count"),
            "out_of_bounds_count": world0.get("out_of_bounds_count"),
            "height_violation_count": world0.get("height_violation_count"),
            "offboard_drop_count": world0.get("offboard_drop_count"),
            "reset_pollution_detected": world0.get("reset_pollution_detected"),
        },
        "world1_zero_shot": {
            "episodes": world1_zero.get("episodes"),
            "success_rate": world1_zero.get("success_rate"),
            "obstacles_per_episode": world1_zero.get("obstacles_per_episode"),
            "target_visible_ratio_mean": world1_zero.get("target_visible_ratio_mean"),
            "final_distance_mean": world1_zero.get("final_distance_mean"),
            "min_distance_mean": world1_zero.get("min_distance_mean"),
            "raw_timeout_count": world1_zero.get("raw_timeout_count"),
            "collision_count": world1_zero.get("collision_count"),
            "out_of_bounds_count": world1_zero.get("out_of_bounds_count"),
            "height_violation_count": world1_zero.get("height_violation_count"),
            "depth_stop_count": world1_zero.get("depth_stop_count"),
            "offboard_drop_count": world1_zero.get("offboard_drop_count"),
            "reset_pollution_detected": world1_zero.get("reset_pollution_detected"),
        },
        "world1_robustness_stress": {
            "total_episodes": world1_robust.get("total_episodes"),
            "total_success_rate": world1_robust.get("total_success_rate"),
            "group_success_rates": world1_robust.get("group_success_rates"),
            "target_visible_ratio_mean": world1_robust.get("target_visible_ratio_mean"),
            "final_distance_mean": world1_robust.get("final_distance_mean"),
            "min_distance_mean": world1_robust.get("min_distance_mean"),
            "raw_timeout_count": world1_robust.get("raw_timeout_count"),
            "collision_count": world1_robust.get("collision_count"),
            "out_of_bounds_count": world1_robust.get("out_of_bounds_count"),
            "height_violation_count": world1_robust.get("height_violation_count"),
            "depth_stop_count": world1_robust.get("depth_stop_count"),
            "offboard_drop_count": world1_robust.get("offboard_drop_count"),
            "reset_pollution_detected": world1_robust.get("reset_pollution_detected"),
            "group_d_safety_intervention_passed": world1_robust.get(
                "group_d_safety_intervention_passed"
            ),
        },
    }


def build_manifest(args, world0, world1_zero, world1_robust, sweep, bc_summary, bc_aggregated, config):
    env_cfg = config.get("env", {}) if isinstance(config, dict) else {}
    registered_model = os.path.join(args.output_dir, "best_world0_world1_policy.zip")
    registered_vec = os.path.join(args.output_dir, "best_world0_world1_vecnormalize.pkl")
    return {
        "policy_name": "world0_world1_sparse_best_ppo_from_bc_v2_step2500",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_model_path": os.path.abspath(args.model),
        "source_vecnormalize_path": os.path.abspath(args.vecnormalize),
        "registered_model_path": os.path.abspath(registered_model),
        "registered_vecnormalize_path": os.path.abspath(registered_vec),
        "registered_model_sha256": sha256(registered_model),
        "registered_vecnormalize_sha256": sha256(registered_vec),
        "training_method": (
            "DAgger-lite + BC v2 initialization, then conservative PPO fine-tuning "
            "from BC v2 on world0; checkpoint selected by deterministic sweep."
        ),
        "BC source": {
            "bc_v2_dir": BC_V2_DIR,
            "bc_policy_best": bc_summary.get("bc_policy_best", os.path.join(BC_V2_DIR, "bc_policy_best.pt")),
            "aggregated_dataset": bc_summary.get("dataset", BC_SOURCE_DATASET),
            "aggregated_summary": bc_aggregated,
            "bc_rows": bc_summary.get("rows"),
            "bc_train_rows": bc_summary.get("train_rows"),
            "bc_val_rows": bc_summary.get("val_rows"),
        },
        "PPO fine-tune run dir": PPO_RUN_DIR,
        "checkpoint step": sweep.get("best_checkpoint_step", 2500),
        "observation_dim": 20,
        "action_dim": 4,
        "action_mapping": {
            "action_space": "Box(-1, 1, shape=(4,))",
            "channels": ["vx_body", "vy_body", "vz_body", "yaw_rate"],
            "raw_vx_body": "a0 * max_vx if a0 >= 0 else a0 * abs(min_vx)",
            "raw_vy_body": "a1 * max_vy",
            "raw_vz_body": "a2 * max_vz",
            "raw_yaw_rate": "a3 * max_yaw_rate",
            "max_vx": env_cfg.get("max_vx", 0.5),
            "min_vx": env_cfg.get("min_vx", -0.2),
            "max_vy": env_cfg.get("max_vy", 0.3),
            "max_vz": env_cfg.get("max_vz", 0.25),
            "max_yaw_rate": env_cfg.get("max_yaw_rate", 0.6),
        },
        "VecNormalize status": {
            "loaded_from": os.path.abspath(registered_vec),
            "training": False,
            "norm_reward": False,
            "purpose": "Observation normalization for evaluation; statistics are frozen.",
        },
        "tested worlds": [
            {"world": "world_0", "coverage": "empty world red-ball chase robustness"},
            {
                "world": "world_1",
                "coverage": "sparse obstacles: 4 obstacle zero-shot, 4/6/8 obstacle stress, front obstacle intervention",
            },
        ],
        "validation summary": validation_summary(world0, world1_zero, world1_robust),
        "known limitations": [
            "woods, woods_easy, woods_hard, and random_woods have not been evaluated.",
            "Dense obstacle fields, severe occlusion, and dynamic obstacles have not been validated.",
            "The policy is fixed for world0/world1 sparse obstacle use; additional world1 training is not recommended after 90/90 success.",
            "A separate woods gate is required before claiming woods generalization.",
        ],
    }


def write_json(path, data):
    with open(path, "w") as handle:
        json.dump(data, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def write_reproduce_commands(path, registry_dir):
    model = os.path.join(registry_dir, "best_world0_world1_policy.zip")
    vec = os.path.join(registry_dir, "best_world0_world1_vecnormalize.pkl")
    content = """# Reproduce Commands

## Environment

```bash
source /opt/ros/noetic/setup.bash
source /home/whk/vf_ws/devel/setup.bash
source /home/whk/vf_ws/src/drone_chase/scripts/source_px4_gazebo_env.sh
```

## World0 Eval Runtime

Terminal 1:

```bash
source /opt/ros/noetic/setup.bash
source /home/whk/vf_ws/devel/setup.bash
source /home/whk/vf_ws/src/drone_chase/scripts/source_px4_gazebo_env.sh
roslaunch drone_chase phase2_chase_world.launch world:=world_0 gui:=false num_obstacles:=0
```

Terminal 2:

```bash
source /opt/ros/noetic/setup.bash
source /home/whk/vf_ws/devel/setup.bash
roslaunch drone_chase phase3_perception.launch debug:=false
```

Terminal 3:

```bash
source /opt/ros/noetic/setup.bash
source /home/whk/vf_ws/devel/setup.bash
roslaunch drone_chase phase6_env_runtime.launch
```

Terminal 4:

```bash
source /opt/ros/noetic/setup.bash
source /home/whk/vf_ws/devel/setup.bash
rosrun drone_chase phase7_world0_best_policy_robustness_eval.py \\
  --model {model} \\
  --vecnormalize {vec} \\
  --episodes 30 \\
  --deterministic \\
  --output-dir /home/whk/vf_ws/outputs/phase7/repro_world0_best_policy_robustness
```

## World1 Eval Runtime

Terminal 1:

```bash
source /opt/ros/noetic/setup.bash
source /home/whk/vf_ws/devel/setup.bash
source /home/whk/vf_ws/src/drone_chase/scripts/source_px4_gazebo_env.sh
roslaunch drone_chase phase2_chase_world.launch world:=world_1 gui:=false num_obstacles:=4
```

Terminal 2:

```bash
source /opt/ros/noetic/setup.bash
source /home/whk/vf_ws/devel/setup.bash
roslaunch drone_chase phase3_perception.launch debug:=false
```

Terminal 3:

```bash
source /opt/ros/noetic/setup.bash
source /home/whk/vf_ws/devel/setup.bash
roslaunch drone_chase phase6_env_runtime.launch
```

## Best Policy Eval Commands

World1 zero-shot:

```bash
source /opt/ros/noetic/setup.bash
source /home/whk/vf_ws/devel/setup.bash
rosrun drone_chase phase7_world1_zeroshot_eval.py \\
  --model {model} \\
  --vecnormalize {vec} \\
  --episodes 30 \\
  --world world_1 \\
  --deterministic \\
  --output-dir /home/whk/vf_ws/outputs/phase7/repro_world1_zeroshot_from_registry
```

World1 robustness/stress:

```bash
source /opt/ros/noetic/setup.bash
source /home/whk/vf_ws/devel/setup.bash
rosrun drone_chase phase7_world1_robustness_eval.py \\
  --model {model} \\
  --vecnormalize {vec} \\
  --world world_1 \\
  --deterministic \\
  --output-dir /home/whk/vf_ws/outputs/phase7/repro_world1_robustness_from_registry
```

## Phase 3 Perception

```bash
source /opt/ros/noetic/setup.bash
source /home/whk/vf_ws/devel/setup.bash
roslaunch drone_chase phase3_perception.launch debug:=false
```

## Phase 6 Safety Runtime

```bash
source /opt/ros/noetic/setup.bash
source /home/whk/vf_ws/devel/setup.bash
roslaunch drone_chase phase6_env_runtime.launch
```

## Runtime Checks

```bash
source /opt/ros/noetic/setup.bash
source /home/whk/vf_ws/devel/setup.bash
rostopic echo -n 1 /mavros/state
rostopic echo -n 1 /safety_filter/mode
rostopic echo -n 1 /target/state
rostopic echo -n 1 /obstacle/risk
rostopic echo -n 1 /mavros/setpoint_velocity/cmd_vel
```

## Cleanup

```bash
pkill -f roslaunch || true
pkill -f gzserver || true
pkill -f gzclient || true
pkill -f px4 || true
pkill -f mavros || true
pkill -f safety_filter_node.py || true
pkill -f phase7 || true
```
""".format(model=model, vec=vec)
    with open(path, "w") as handle:
        handle.write(content)


def yes_no(value):
    return "是" if bool(value) else "否"


def write_final_report(path, manifest, world0, world1_zero, world1_robust):
    lines = [
        "# Final Phase 7 World0/World1 Report",
        "",
        "## 1. Project scope",
        "",
        "This registry freezes the current best policy for world0 and world1 sparse-obstacle chase validation only.",
        "It does not include woods, dense obstacle, severe occlusion, or dynamic obstacle claims.",
        "",
        "## 2. Final policy",
        "",
        "- policy_name: {}".format(manifest["policy_name"]),
        "- model: {}".format(manifest["registered_model_path"]),
        "- vecnormalize: {}".format(manifest["registered_vecnormalize_path"]),
        "- checkpoint: ppo_step_{}".format(manifest["checkpoint step"]),
        "",
        "## 3. Training route",
        "",
        "- Pure PPO failed to provide the required stable final behavior.",
        "- Reward v2 fixed the observed reward-hacking behavior.",
        "- BC v1 failed online because of covariate shift.",
        "- DAgger-lite + BC v2 succeeded as the imitation-learning base.",
        "- PPO from BC v2 succeeded; deterministic checkpoint sweep selected ppo_step_2500.",
        "",
        "## 4. World0 validation",
        "",
        "- episodes: {}".format(world0.get("episodes")),
        "- success_rate: {:.4f}".format(float(world0.get("success_rate", 0.0))),
        "- target_visible_ratio_mean: {:.4f}".format(float(world0.get("target_visible_ratio_mean", 0.0))),
        "- final_distance_mean: {:.4f} m".format(float(world0.get("final_distance_mean", 0.0))),
        "- min_distance_mean: {:.4f} m".format(float(world0.get("min_distance_mean", 0.0))),
        "- RAW_TIMEOUT: {}".format(world0.get("raw_timeout_count")),
        "- reset_pollution_detected: {}".format(yes_no(world0.get("reset_pollution_detected"))),
        "",
        "## 5. World1 zero-shot validation",
        "",
        "- episodes: {}".format(world1_zero.get("episodes")),
        "- success_rate: {:.4f}".format(float(world1_zero.get("success_rate", 0.0))),
        "- obstacles_per_episode: {}".format(world1_zero.get("obstacles_per_episode")),
        "- collision_count: {}".format(world1_zero.get("collision_count")),
        "- OFFBOARD drop count: {}".format(world1_zero.get("offboard_drop_count")),
        "",
        "## 6. World1 robustness/stress validation",
        "",
        "- total episodes: {}".format(world1_robust.get("total_episodes")),
        "- total success_rate: {:.4f}".format(float(world1_robust.get("total_success_rate", 0.0))),
        "- Group A/B/C/D success rates: {} / {} / {} / {}".format(
            fmt(world1_robust.get("group_success_rates", {}).get("A")),
            fmt(world1_robust.get("group_success_rates", {}).get("B")),
            fmt(world1_robust.get("group_success_rates", {}).get("C")),
            fmt(world1_robust.get("group_success_rates", {}).get("D")),
        ),
        "- target_visible_ratio_mean: {:.4f}".format(
            float(world1_robust.get("target_visible_ratio_mean", 0.0))
        ),
        "- final_distance_mean: {:.4f} m".format(float(world1_robust.get("final_distance_mean", 0.0))),
        "- min_distance_mean: {:.4f} m".format(float(world1_robust.get("min_distance_mean", 0.0))),
        "- RAW_TIMEOUT: {}".format(world1_robust.get("raw_timeout_count")),
        "- collision/out_of_bounds/height_violation: {} / {} / {}".format(
            world1_robust.get("collision_count"),
            world1_robust.get("out_of_bounds_count"),
            world1_robust.get("height_violation_count"),
        ),
        "- OFFBOARD drop count: {}".format(world1_robust.get("offboard_drop_count")),
        "",
        "## 7. Safety intervention result",
        "",
        "- danger_seen: {}".format(yes_no(world1_robust.get("group_d_danger_seen_all"))),
        "- safety_triggered: {}".format(yes_no(world1_robust.get("group_d_safety_triggered_all"))),
        "- filtered_vx <= 0: {}".format(yes_no(world1_robust.get("group_d_filtered_vx_nonpositive_all"))),
        "- no_collision: {}".format(yes_no(world1_robust.get("group_d_no_collision"))),
        "- no_offboard: {}".format(yes_no(world1_robust.get("group_d_no_offboard_drop"))),
        "- recovered_or_safe: {}".format(yes_no(world1_robust.get("group_d_recovered_or_safe"))),
        "",
        "## 8. Policy artifacts",
        "",
        "- policy_manifest.json",
        "- best_world0_world1_policy.zip",
        "- best_world0_world1_vecnormalize.pkl",
        "- final_results_table.csv",
        "- final_results_table.md",
        "- reproduce_commands.md",
        "",
        "## 9. Reproduction commands",
        "",
        "See reproduce_commands.md in this registry.",
        "",
        "## 10. Limitations",
        "",
        "- Current policy has passed world0 and world1 sparse obstacle validation.",
        "- Current policy has not entered woods.",
        "- Dense obstacles, severe occlusion, and dynamic obstacles are not validated.",
        "- Continued world1 training is not recommended after 90/90 success; it may degrade the fixed behavior.",
        "- Do not claim woods generalization from these results.",
        "",
        "## 11. Next phase recommendation",
        "",
        "Proceed to Phase 7.3A: woods_easy zero-shot validation only if a separate woods gate is defined.",
        "Do not directly claim generalized woods performance before that gate passes.",
    ]
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")


def write_phase7_2c_report(path, output_dir, manifest, world0, world1_zero, world1_robust):
    lines = [
        "# Phase 7.2C Policy Freeze + Artifact Registry Report",
        "",
        "1. 是否创建 final_policy_registry：{}".format(yes_no(os.path.isdir(output_dir))),
        "2. 是否注册 best policy：{}".format(
            yes_no(
                os.path.exists(manifest["registered_model_path"])
                and os.path.exists(manifest["registered_vecnormalize_path"])
            )
        ),
        "3. registered model path：{}".format(manifest["registered_model_path"]),
        "4. registered vecnormalize path：{}".format(manifest["registered_vecnormalize_path"]),
        "5. 是否生成 policy_manifest.json：{}".format(
            yes_no(os.path.exists(os.path.join(output_dir, "policy_manifest.json")))
        ),
        "6. 是否生成 final_results_table.csv：{}".format(
            yes_no(os.path.exists(os.path.join(output_dir, "final_results_table.csv")))
        ),
        "7. 是否生成 final_results_table.md：{}".format(
            yes_no(os.path.exists(os.path.join(output_dir, "final_results_table.md")))
        ),
        "8. 是否生成 reproduce_commands.md：{}".format(
            yes_no(os.path.exists(os.path.join(output_dir, "reproduce_commands.md")))
        ),
        "9. 是否生成 final_phase7_world0_world1_report.md：{}".format(
            yes_no(os.path.exists(os.path.join(output_dir, "final_phase7_world0_world1_report.md")))
        ),
        "10. world0 final success rate：{:.4f} ({}/{})".format(
            float(world0.get("success_rate", 0.0)),
            world0.get("success_count"),
            world0.get("episodes"),
        ),
        "11. world1 zero-shot success rate：{:.4f} ({}/{})".format(
            float(world1_zero.get("success_rate", 0.0)),
            world1_zero.get("success_count"),
            world1_zero.get("episodes"),
        ),
        "12. world1 robustness success rate：{:.4f} ({}/{})".format(
            float(world1_robust.get("total_success_rate", 0.0)),
            world1_robust.get("total_success_count"),
            world1_robust.get("total_episodes"),
        ),
        "13. safety intervention 是否通过：{}".format(
            yes_no(world1_robust.get("group_d_safety_intervention_passed"))
        ),
        "14. 是否允许把当前 policy 作为 world0/world1 best policy：是",
        "15. 是否允许进入 woods_easy zero-shot：是",
        "16. 是否允许声称 woods 已通过：否",
        "17. 当前限制：仅通过 world0 与 world1 sparse obstacle；woods/dense obstacle/severe occlusion/dynamic obstacle 未验证；不建议继续 world1 训练。",
    ]
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Build final Phase 7 policy registry tables and reports.")
    parser.add_argument("--output-dir", default=DEFAULT_REGISTRY_DIR)
    parser.add_argument("--model", default=SOURCE_MODEL)
    parser.add_argument("--vecnormalize", default=SOURCE_VECNORMALIZE)
    parser.add_argument("--world0-robustness", default=DEFAULT_WORLD0_ROBUSTNESS)
    parser.add_argument("--world1-zeroshot", default=DEFAULT_WORLD1_ZEROSHOT)
    parser.add_argument("--world1-robustness", default=DEFAULT_WORLD1_ROBUSTNESS)
    parser.add_argument("--checkpoint-sweep", default=DEFAULT_CHECKPOINT_SWEEP)
    parser.add_argument("--bc-v2", default=DEFAULT_BC_V2)
    parser.add_argument("--bc-summary", default=DEFAULT_BC_SUMMARY)
    parser.add_argument("--bc-aggregated-summary", default=DEFAULT_BC_AGGREGATED)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--tables-only", action="store_true")
    return parser


def main():
    args = build_arg_parser().parse_args()
    ensure_dir(args.output_dir)

    registered_model = os.path.join(args.output_dir, "best_world0_world1_policy.zip")
    registered_vec = os.path.join(args.output_dir, "best_world0_world1_vecnormalize.pkl")
    copy_policy_artifact(args.model, registered_model)
    copy_policy_artifact(args.vecnormalize, registered_vec)

    world0 = load_json(args.world0_robustness)
    world1_zero = load_json(args.world1_zeroshot)
    world1_robust = load_json(args.world1_robustness)
    sweep = load_json(args.checkpoint_sweep)
    bc_v2 = load_json(args.bc_v2)
    bc_summary = maybe_load_json(args.bc_summary)
    bc_aggregated = maybe_load_json(args.bc_aggregated_summary)
    config = load_config(args.config)

    rows = make_table_rows(world0, world1_zero, world1_robust, sweep, bc_v2)
    csv_path = os.path.join(args.output_dir, "final_results_table.csv")
    md_path = os.path.join(args.output_dir, "final_results_table.md")
    write_csv(csv_path, rows)
    write_markdown_table(md_path, rows)

    if not args.tables_only:
        manifest = build_manifest(args, world0, world1_zero, world1_robust, sweep, bc_summary, bc_aggregated, config)
        manifest_path = os.path.join(args.output_dir, "policy_manifest.json")
        reproduce_path = os.path.join(args.output_dir, "reproduce_commands.md")
        final_report_path = os.path.join(args.output_dir, "final_phase7_world0_world1_report.md")
        phase_report_path = os.path.join(args.output_dir, "phase7_2c_report.md")
        write_json(manifest_path, manifest)
        write_reproduce_commands(reproduce_path, args.output_dir)
        write_final_report(final_report_path, manifest, world0, world1_zero, world1_robust)
        write_phase7_2c_report(phase_report_path, args.output_dir, manifest, world0, world1_zero, world1_robust)
        print("wrote {}".format(manifest_path))
        print("wrote {}".format(reproduce_path))
        print("wrote {}".format(final_report_path))
        print("wrote {}".format(phase_report_path))

    print("wrote {}".format(csv_path))
    print("wrote {}".format(md_path))
    print("registered {}".format(registered_model))
    print("registered {}".format(registered_vec))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("phase7_build_final_results_table failed: {}".format(exc))
        raise
