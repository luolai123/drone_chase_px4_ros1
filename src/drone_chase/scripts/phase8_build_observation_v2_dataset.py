#!/usr/bin/env python3

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "envs"))
if ENV_DIR not in sys.path:
    sys.path.insert(0, ENV_DIR)

from observation_builder_v2 import (  # noqa: E402
    FIELD_NAMES,
    OBSERVATION_V2_DIM,
    build_observation_v2,
    self_check,
)


DEFAULT_DIAGNOSTIC_CSV = "/home/whk/vf_ws/outputs/phase8/perception_diagnostics/perception_diagnostic_log.csv"
DEFAULT_ABLATION_ROLLOUTS = (
    "/home/whk/vf_ws/outputs/phase8/policy_feature_ablation/policy_feature_ablation_rollouts.csv"
)
DEFAULT_OUTPUT_DIR = "/home/whk/vf_ws/outputs/phase8/extended_observation_design"
MAX_DEPTH = 10.0
MAX_VZ = 0.25
MAX_YAW_RATE = 0.6


def finite(value, default=math.nan):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(default)
    return value if math.isfinite(value) else float(default)


def mean(values):
    values = [finite(value) for value in values]
    values = [value for value in values if math.isfinite(value)]
    return float(sum(values)) / float(len(values)) if values else 0.0


def std(values):
    values = [finite(value) for value in values]
    values = [value for value in values if math.isfinite(value)]
    if not values:
        return 0.0
    avg = mean(values)
    return float(math.sqrt(sum((value - avg) ** 2 for value in values) / float(len(values))))


def pearson(xs, ys):
    xs = [finite(value) for value in xs]
    ys = [finite(value) for value in ys]
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 3:
        return 0.0
    x_arr = np.asarray([item[0] for item in pairs], dtype=np.float64)
    y_arr = np.asarray([item[1] for item in pairs], dtype=np.float64)
    if float(np.std(x_arr)) <= 1e-12 or float(np.std(y_arr)) <= 1e-12:
        return 0.0
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def load_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def group_by_world(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row.get("world", ""))].append(row)
    return grouped


def group_rollouts(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row.get("world", ""))].append(row)
    return grouped


def aligned_diagnostic(row_index, rollout_count, diagnostic_rows):
    if not diagnostic_rows:
        return {}
    if rollout_count <= 1:
        return diagnostic_rows[0]
    diag_index = int(round(float(row_index) * float(len(diagnostic_rows) - 1) / float(rollout_count - 1)))
    diag_index = max(0, min(len(diagnostic_rows) - 1, diag_index))
    return diagnostic_rows[diag_index]


def episode_labels(episode_rows):
    labels = {}
    for row in episode_rows:
        key = (str(row.get("world", "")), str(row.get("ablation_group", "")), str(row.get("episode", "")))
        done_reason = str(row.get("done_reason", "") or "")
        labels[key] = {
            "success": int(finite(row.get("success", 0), 0)),
            "failure": 0 if int(finite(row.get("success", 0), 0)) else 1,
            "done_reason": done_reason,
        }
    return labels


def infer_episode_csv(ablation_rollouts):
    path = os.path.join(os.path.dirname(os.path.abspath(ablation_rollouts)), "policy_feature_ablation_episodes.csv")
    return path if os.path.exists(path) else ""


def safety_event(safety_mode):
    safety_mode = str(safety_mode or "")
    return int(
        "EMERGENCY_AVOID" in safety_mode
        or "DEPTH_STOP" in safety_mode
        or "TARGET_LOST" in safety_mode
        or "RAW_TIMEOUT" in safety_mode
    )


def approx_target_camera(target_depth, target_u, target_v):
    depth = finite(target_depth, MAX_DEPTH)
    u = finite(target_u, 0.0)
    v = finite(target_v, 0.0)
    return {
        "x": float(u * depth),
        "y": float(v * depth),
        "z": float(depth),
    }


def diagnostic_quality(diag):
    quality = str(diag.get("target_quality", "") or "lost")
    return {
        "visible_ratio_window": finite(diag.get("target_visible_ratio_window", 0.0), 0.0),
        "lost_frames": finite(diag.get("target_lost_frames", 20.0), 20.0),
        "depth_valid_ratio": finite(diag.get("depth_valid_ratio", 0.0), 0.0),
        "radius_px_smooth": finite(diag.get("radius_px", 0.0), 0.0),
        "quality": quality,
        "confidence": finite(diag.get("target_confidence", 0.0), 0.0),
    }


def diagnostic_depth_debug(diag):
    return {
        "front_q05_smoothed": finite(diag.get("front_q05_smoothed", MAX_DEPTH), MAX_DEPTH),
        "sectors": {
            "far_left": {"q05": finite(diag.get("sector_far_left_q05", MAX_DEPTH), MAX_DEPTH)},
            "left": {"q05": finite(diag.get("sector_left_q05", MAX_DEPTH), MAX_DEPTH)},
            "front": {"q05": finite(diag.get("sector_front_q05", MAX_DEPTH), MAX_DEPTH)},
            "right": {"q05": finite(diag.get("sector_right_q05", MAX_DEPTH), MAX_DEPTH)},
            "far_right": {"q05": finite(diag.get("sector_far_right_q05", MAX_DEPTH), MAX_DEPTH)},
        },
    }


def field_stats(observations, missing_counts):
    stats = {}
    row_count = int(observations.shape[0])
    for index, name in enumerate(FIELD_NAMES):
        values = observations[:, index] if row_count else np.asarray([], dtype=np.float32)
        stats[name] = {
            "mean": float(np.mean(values)) if row_count else 0.0,
            "std": float(np.std(values)) if row_count else 0.0,
            "min": float(np.min(values)) if row_count else 0.0,
            "max": float(np.max(values)) if row_count else 0.0,
            "missing_rate": float(missing_counts.get(name, 0)) / float(max(1, row_count)),
        }
    return stats


def correlations(observations, labels):
    success = labels["success"].astype(np.float64)
    failure = labels["failure"].astype(np.float64)
    safety = labels["safety_event"].astype(np.float64)
    result = {}
    for index, name in enumerate(FIELD_NAMES):
        values = observations[:, index].astype(np.float64)
        result[name] = {
            "corr_success": pearson(values, success),
            "corr_failure": pearson(values, failure),
            "corr_safety_event": pearson(values, safety),
        }
    return result


def build_dataset(diagnostic_rows, rollout_rows, episode_rows):
    diag_by_world = group_by_world(diagnostic_rows)
    rollout_by_world = group_rollouts(rollout_rows)
    labels_by_episode = episode_labels(episode_rows)
    previous_action = {}

    observations = []
    output_rows = []
    missing_counts = defaultdict(int)
    success_labels = []
    failure_labels = []
    safety_labels = []

    for world, rows in rollout_by_world.items():
        diag_rows = diag_by_world.get(world, [])
        for world_row_index, row in enumerate(rows):
            diag = aligned_diagnostic(world_row_index, len(rows), diag_rows)
            target_depth = finite(row.get("target_depth", MAX_DEPTH), MAX_DEPTH)
            target_u = finite(row.get("target_u", diag.get("target_u", 0.0)), 0.0)
            target_v = finite(row.get("target_v", diag.get("target_v", 0.0)), 0.0)

            target_state = {
                "visible": bool(int(finite(row.get("target_visible", 0), 0))),
                "position_camera": approx_target_camera(target_depth, target_u, target_v),
                "depth": target_depth,
                "u": target_u,
                "v": target_v,
                "confidence": finite(diag.get("target_confidence", 0.0), 0.0),
            }
            obstacle_risk = {
                "front_q05_depth": finite(row.get("front_q05_depth", MAX_DEPTH), MAX_DEPTH),
                "obstacle_area_ratio": finite(row.get("obstacle_area_ratio", 0.0), 0.0),
                "danger": bool(int(finite(row.get("obstacle_danger", 0), 0))),
            }
            uav_state = {
                "drone_z": finite(diag.get("drone_z", 0.0), 0.0),
                "drone_vx": finite(row.get("filtered_vx_body", 0.0), 0.0),
                "drone_vy": 0.0,
                "drone_vz": finite(row.get("action_vz", 0.0), 0.0) * MAX_VZ,
            }
            key = (str(row.get("world", "")), str(row.get("ablation_group", "")), str(row.get("episode", "")))
            prev = previous_action.get(key, {"prev_vx": 0.0, "prev_vz": 0.0, "prev_yaw_rate": 0.0})

            obs, meta = build_observation_v2(
                target_state=target_state,
                obstacle_risk=obstacle_risk,
                depth_debug=diagnostic_depth_debug(diag) if diag else None,
                detection_quality=diagnostic_quality(diag) if diag else None,
                uav_state=uav_state,
                prev_action=prev,
                return_metadata=True,
            )
            for field, is_missing in meta["missing"].items():
                missing_counts[field] += int(bool(is_missing))
            missing_counts["drone_vy"] += 1

            labels = labels_by_episode.get(
                key,
                {
                    "success": int(str(row.get("done_reason", "")) == "success"),
                    "failure": int(bool(str(row.get("done_reason", "")) and str(row.get("done_reason", "")) != "success")),
                    "done_reason": str(row.get("done_reason", "")),
                },
            )
            safety = safety_event(row.get("safety_mode", ""))

            output = {
                "row_id": len(output_rows),
                "world": row.get("world", ""),
                "ablation_group": row.get("ablation_group", ""),
                "episode": row.get("episode", ""),
                "step": row.get("step", ""),
                "success_label": int(labels["success"]),
                "failure_label": int(labels["failure"]),
                "done_reason": labels.get("done_reason", ""),
                "safety_mode": row.get("safety_mode", ""),
                "safety_event": int(safety),
            }
            for index, name in enumerate(FIELD_NAMES):
                output[name] = float(obs[index])
            output_rows.append(output)
            observations.append(obs)
            success_labels.append(int(labels["success"]))
            failure_labels.append(int(labels["failure"]))
            safety_labels.append(int(safety))

            previous_action[key] = {
                "prev_vx": finite(row.get("raw_vx_body", 0.0), 0.0),
                "prev_vz": finite(row.get("action_vz", 0.0), 0.0) * MAX_VZ,
                "prev_yaw_rate": finite(row.get("action_yaw", 0.0), 0.0) * MAX_YAW_RATE,
            }

    obs_array = np.asarray(observations, dtype=np.float32)
    label_arrays = {
        "success": np.asarray(success_labels, dtype=np.int8),
        "failure": np.asarray(failure_labels, dtype=np.int8),
        "safety_event": np.asarray(safety_labels, dtype=np.int8),
    }
    return obs_array, output_rows, label_arrays, dict(missing_counts)


def write_dataset_csv(path, rows):
    fieldnames = [
        "row_id",
        "world",
        "ablation_group",
        "episode",
        "step",
        "success_label",
        "failure_label",
        "done_reason",
        "safety_mode",
        "safety_event",
    ] + list(FIELD_NAMES)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_outputs(output_dir, observations, rows, labels, missing_counts, args):
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "observation_v2_dataset.csv")
    npz_path = os.path.join(output_dir, "observation_v2_dataset.npz")
    summary_path = os.path.join(output_dir, "observation_v2_dataset_summary.json")

    write_dataset_csv(csv_path, rows)
    np.savez_compressed(
        npz_path,
        observations=observations.astype(np.float32),
        field_names=np.asarray(FIELD_NAMES),
        success=labels["success"],
        failure=labels["failure"],
        safety_event=labels["safety_event"],
    )

    has_bad = bool(not np.all(np.isfinite(observations)))
    stats = field_stats(observations, missing_counts)
    corr = correlations(observations, labels) if observations.shape[0] else {}
    max_missing_field = ""
    max_missing_rate = 0.0
    for name, item in stats.items():
        if float(item["missing_rate"]) > max_missing_rate:
            max_missing_field = name
            max_missing_rate = float(item["missing_rate"])

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "observation_version": "observation_v2",
        "dim": int(OBSERVATION_V2_DIM),
        "rows": int(observations.shape[0]),
        "diagnostic_csv": args.diagnostic_csv,
        "ablation_rollouts": args.ablation_rollouts,
        "episode_csv": args.episode_csv,
        "csv_path": csv_path,
        "npz_path": npz_path,
        "has_nan_or_inf": has_bad,
        "max_missing_field": max_missing_field,
        "max_missing_rate": max_missing_rate,
        "field_stats": stats,
        "correlations": corr,
        "source_notes": [
            "target_x/y/z_camera are reconstructed from target_depth and target_u/v for offline CSV data; runtime builder uses /target/state.position_camera.",
            "drone_vx uses filtered_vx_body as an offline proxy because ablation rollouts did not log MAVROS body velocity.",
            "drone_vy is defaulted to 0.0 in the offline dataset because it was not logged.",
            "drone_vz is approximated from logged normalized action_vz and max_vz=0.25.",
        ],
    }
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    return csv_path, npz_path, summary_path, summary


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Build offline observation_v2 dataset from Phase 8 CSV logs.")
    parser.add_argument("--diagnostic-csv", default=DEFAULT_DIAGNOSTIC_CSV)
    parser.add_argument("--ablation-rollouts", default=DEFAULT_ABLATION_ROLLOUTS)
    parser.add_argument("--episode-csv", default="")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser


def main():
    args = build_arg_parser().parse_args()
    if not args.episode_csv:
        args.episode_csv = infer_episode_csv(args.ablation_rollouts)
    if not args.episode_csv:
        raise RuntimeError("Could not infer episode CSV; pass --episode-csv")

    self_check()
    diagnostic_rows = load_csv(args.diagnostic_csv)
    rollout_rows = load_csv(args.ablation_rollouts)
    episode_rows = load_csv(args.episode_csv)
    observations, rows, labels, missing_counts = build_dataset(diagnostic_rows, rollout_rows, episode_rows)
    csv_path, npz_path, summary_path, summary = write_outputs(
        args.output_dir,
        observations,
        rows,
        labels,
        missing_counts,
        args,
    )
    print("wrote {}".format(csv_path))
    print("wrote {}".format(npz_path))
    print("wrote {}".format(summary_path))
    print("rows={} dim={} has_nan_or_inf={}".format(summary["rows"], summary["dim"], summary["has_nan_or_inf"]))


if __name__ == "__main__":
    main()
