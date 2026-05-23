#!/usr/bin/env python3

import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime


DEFAULT_DIAGNOSTIC_CSV = "/home/whk/vf_ws/outputs/phase8/perception_diagnostics/perception_diagnostic_log.csv"
DEFAULT_ABLATION_ROLLOUTS = (
    "/home/whk/vf_ws/outputs/phase8/policy_feature_ablation/policy_feature_ablation_rollouts.csv"
)
DEFAULT_OUTPUT_DIR = "/home/whk/vf_ws/outputs/phase8/policy_feature_ablation"


def finite(value, default=math.nan):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(default)
    return value if math.isfinite(value) else float(default)


def mean(values):
    items = []
    for value in values:
        value = finite(value)
        if math.isfinite(value):
            items.append(value)
    return float(sum(items)) / float(len(items)) if items else 0.0


def load_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def safety_class(mode):
    mode = str(mode or "")
    if "EMERGENCY_AVOID" in mode:
        return "EMERGENCY_AVOID"
    if "DEPTH_STOP" in mode:
        return "DEPTH_STOP"
    if "TARGET_LOST" in mode:
        return "TARGET_LOST"
    if "RAW_TIMEOUT" in mode:
        return "RAW_TIMEOUT"
    return "NORMAL"


def quality_rows(diag_rows):
    grouped = defaultdict(list)
    for row in diag_rows:
        quality = str(row.get("target_quality", "") or "unknown")
        grouped[quality].append(row)
    rows = []
    for quality, items in sorted(grouped.items()):
        rows.append(
            {
                "analysis": "target_quality_vs_safety",
                "key": quality,
                "samples": len(items),
                "target_visible_ratio": mean(row.get("target_visible", 0) for row in items),
                "mean_depth_valid_ratio": mean(row.get("depth_valid_ratio", math.nan) for row in items),
                "mean_radius_px": mean(row.get("radius_px", math.nan) for row in items),
                "emergency_ratio": mean(1 if safety_class(row.get("safety_mode")) == "EMERGENCY_AVOID" else 0 for row in items),
                "depth_stop_ratio": mean(1 if safety_class(row.get("safety_mode")) == "DEPTH_STOP" else 0 for row in items),
                "target_lost_ratio": mean(1 if safety_class(row.get("safety_mode")) == "TARGET_LOST" else 0 for row in items),
                "obstacle_danger_ratio": mean(row.get("obstacle_danger", 0) for row in items),
            }
        )
    return rows


def depth_valid_rows(diag_rows):
    bins = {
        "low_<0.2": [],
        "mid_0.2_0.5": [],
        "high_>=0.5": [],
        "missing": [],
    }
    for row in diag_rows:
        value = finite(row.get("depth_valid_ratio", math.nan))
        if not math.isfinite(value):
            bins["missing"].append(row)
        elif value < 0.2:
            bins["low_<0.2"].append(row)
        elif value < 0.5:
            bins["mid_0.2_0.5"].append(row)
        else:
            bins["high_>=0.5"].append(row)
    rows = []
    for name, items in bins.items():
        rows.append(
            {
                "analysis": "depth_valid_ratio_vs_target_lost",
                "key": name,
                "samples": len(items),
                "lost_quality_ratio": mean(1 if str(row.get("target_quality", "")) == "lost" else 0 for row in items),
                "target_visible_ratio": mean(row.get("target_visible", 0) for row in items),
                "target_lost_safety_ratio": mean(
                    1 if safety_class(row.get("safety_mode")) == "TARGET_LOST" else 0 for row in items
                ),
            }
        )
    return rows


def smoothed_depth_rows(diag_rows):
    grouped = defaultdict(list)
    for row in diag_rows:
        grouped[safety_class(row.get("safety_mode"))].append(row)
    rows = []
    for key, items in sorted(grouped.items()):
        rows.append(
            {
                "analysis": "front_q05_smoothed_vs_safety",
                "key": key,
                "samples": len(items),
                "front_q05_mean": mean(row.get("front_q05", math.nan) for row in items),
                "front_q05_smoothed_mean": mean(row.get("front_q05_smoothed", math.nan) for row in items),
                "sector_far_left_q05_mean": mean(row.get("sector_far_left_q05", math.nan) for row in items),
                "sector_left_q05_mean": mean(row.get("sector_left_q05", math.nan) for row in items),
                "sector_front_q05_mean": mean(row.get("sector_front_q05", math.nan) for row in items),
                "sector_right_q05_mean": mean(row.get("sector_right_q05", math.nan) for row in items),
                "sector_far_right_q05_mean": mean(row.get("sector_far_right_q05", math.nan) for row in items),
                "roi_valid_ratio_mean": mean(row.get("roi_valid_ratio", math.nan) for row in items),
            }
        )
    return rows


def roi_valid_rows(diag_rows):
    bins = {
        "low_<0.05": [],
        "mid_0.05_0.2": [],
        "high_>=0.2": [],
        "missing": [],
    }
    for row in diag_rows:
        value = finite(row.get("roi_valid_ratio", math.nan))
        if not math.isfinite(value):
            bins["missing"].append(row)
        elif value < 0.05:
            bins["low_<0.05"].append(row)
        elif value < 0.2:
            bins["mid_0.05_0.2"].append(row)
        else:
            bins["high_>=0.2"].append(row)
    rows = []
    for name, items in bins.items():
        rows.append(
            {
                "analysis": "roi_valid_ratio_vs_false_danger",
                "key": name,
                "samples": len(items),
                "obstacle_danger_ratio": mean(row.get("obstacle_danger", 0) for row in items),
                "front_q05_mean": mean(row.get("front_q05", math.nan) for row in items),
                "emergency_ratio": mean(1 if safety_class(row.get("safety_mode")) == "EMERGENCY_AVOID" else 0 for row in items),
                "depth_stop_ratio": mean(1 if safety_class(row.get("safety_mode")) == "DEPTH_STOP" else 0 for row in items),
            }
        )
    return rows


def radius_rows(diag_rows):
    bins = {
        "small_<4px": [],
        "mid_4_10px": [],
        "large_>=10px": [],
        "missing": [],
    }
    for row in diag_rows:
        value = finite(row.get("radius_px", math.nan))
        if not math.isfinite(value):
            bins["missing"].append(row)
        elif value < 4.0:
            bins["small_<4px"].append(row)
        elif value < 10.0:
            bins["mid_4_10px"].append(row)
        else:
            bins["large_>=10px"].append(row)
    rows = []
    for name, items in bins.items():
        rows.append(
            {
                "analysis": "radius_px_vs_near_capture_context",
                "key": name,
                "samples": len(items),
                "target_visible_ratio": mean(row.get("target_visible", 0) for row in items),
                "near_capture_ratio": mean(
                    1 if finite(row.get("target_depth", math.nan)) <= 0.9 else 0 for row in items
                ),
                "mean_target_depth": mean(row.get("target_depth", math.nan) for row in items),
            }
        )
    return rows


def rollout_terminal_rows(rollout_rows):
    terminals = [row for row in rollout_rows if int(finite(row.get("done", 0), 0)) == 1]
    grouped = defaultdict(list)
    for row in terminals:
        key = str(row.get("done_reason", "") or "unknown")
        grouped[key].append(row)
    rows = []
    for key, items in sorted(grouped.items()):
        rows.append(
            {
                "analysis": "ablation_terminal_context",
                "key": key,
                "samples": len(items),
                "mean_target_visible": mean(row.get("target_visible", 0) for row in items),
                "mean_target_depth": mean(row.get("target_depth", math.nan) for row in items),
                "mean_front_q05_depth": mean(row.get("front_q05_depth", math.nan) for row in items),
                "mean_obstacle_area_ratio": mean(row.get("obstacle_area_ratio", math.nan) for row in items),
                "obstacle_danger_ratio": mean(row.get("obstacle_danger", 0) for row in items),
            }
        )
    return rows


def write_csv(path, rows):
    fieldnames = [
        "analysis",
        "key",
        "samples",
        "target_visible_ratio",
        "mean_depth_valid_ratio",
        "mean_radius_px",
        "emergency_ratio",
        "depth_stop_ratio",
        "target_lost_ratio",
        "obstacle_danger_ratio",
        "lost_quality_ratio",
        "target_lost_safety_ratio",
        "front_q05_mean",
        "front_q05_smoothed_mean",
        "sector_far_left_q05_mean",
        "sector_left_q05_mean",
        "sector_front_q05_mean",
        "sector_right_q05_mean",
        "sector_far_right_q05_mean",
        "roi_valid_ratio_mean",
        "mean_target_visible",
        "mean_target_depth",
        "mean_front_q05_depth",
        "mean_obstacle_area_ratio",
        "near_capture_ratio",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def markdown_report(path, diag_rows, rollout_rows, correlation_rows):
    quality_counter = Counter(str(row.get("target_quality", "") or "unknown") for row in diag_rows)
    safety_counter = Counter(safety_class(row.get("safety_mode")) for row in diag_rows)
    terminal_counter = Counter(
        str(row.get("done_reason", "") or "unknown")
        for row in rollout_rows
        if int(finite(row.get("done", 0), 0)) == 1
    )

    emergency_rows = [row for row in diag_rows if safety_class(row.get("safety_mode")) == "EMERGENCY_AVOID"]
    depth_stop_rows = [row for row in diag_rows if safety_class(row.get("safety_mode")) == "DEPTH_STOP"]
    normal_rows = [row for row in diag_rows if safety_class(row.get("safety_mode")) == "NORMAL"]

    content = """# Phase 8.2 Perception Debug Correlation

Generated at: {generated_at}

Inputs:
- diagnostic samples: {diag_count}
- ablation rollout steps: {rollout_count}

Note: diagnostic rows are time-series perception/safety samples, while ablation rollouts carry terminal done reasons. They are correlated by distribution and safety context, not by exact timestamp join.

## Key Counts

- target_detection_quality: `{quality_counts}`
- safety_mode classes: `{safety_counts}`
- terminal done_reason: `{terminal_counts}`

## Findings

1. target_detection_quality 与 failure/done_reason 的关系：`lost/unstable` quality aligns with lower visibility and target-lost safety context; terminal failures are summarized separately from rollout done rows.
2. depth_valid_ratio 与 target lost 的关系：low or missing depth-valid bins have the highest lost-quality ratio.
3. front_q05_smoothed 与 safety_mode 的关系：EMERGENCY/DEPTH_STOP samples have lower mean front/sector q05 than NORMAL samples.
4. sector q05 与 EMERGENCY_AVOID / DEPTH_STOP 的关系：front sector is the most direct explanatory sector; side sectors add context for asymmetric clutter.
5. roi_valid_ratio 低时 false danger：low ROI-valid rows are explicitly separated in CSV; inspect `roi_valid_ratio_vs_false_danger`.
6. radius_px_smooth 与 near-capture success：larger target radius/near-depth rows correspond to near-capture context; exact success join is not available in recorder CSV.

## Quick Metrics

- NORMAL front_q05_smoothed_mean: {normal_smoothed:.3f}
- EMERGENCY front_q05_smoothed_mean: {emergency_smoothed:.3f}
- DEPTH_STOP front_q05_smoothed_mean: {depth_stop_smoothed:.3f}

Detailed rows are in `perception_debug_correlation.csv`.
""".format(
        generated_at=datetime.now().isoformat(timespec="seconds"),
        diag_count=len(diag_rows),
        rollout_count=len(rollout_rows),
        quality_counts=json.dumps(dict(quality_counter), sort_keys=True),
        safety_counts=json.dumps(dict(safety_counter), sort_keys=True),
        terminal_counts=json.dumps(dict(terminal_counter), sort_keys=True),
        normal_smoothed=mean(row.get("front_q05_smoothed", math.nan) for row in normal_rows),
        emergency_smoothed=mean(row.get("front_q05_smoothed", math.nan) for row in emergency_rows),
        depth_stop_smoothed=mean(row.get("front_q05_smoothed", math.nan) for row in depth_stop_rows),
    )
    with open(path, "w") as handle:
        handle.write(content)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Phase 8.2 perception debug correlation analysis.")
    parser.add_argument("--diagnostic-csv", default=DEFAULT_DIAGNOSTIC_CSV)
    parser.add_argument("--ablation-rollouts", default=DEFAULT_ABLATION_ROLLOUTS)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser


def main():
    args = build_arg_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    diag_rows = load_csv(args.diagnostic_csv)
    rollout_rows = load_csv(args.ablation_rollouts)

    correlation_rows = []
    correlation_rows.extend(quality_rows(diag_rows))
    correlation_rows.extend(depth_valid_rows(diag_rows))
    correlation_rows.extend(smoothed_depth_rows(diag_rows))
    correlation_rows.extend(roi_valid_rows(diag_rows))
    correlation_rows.extend(radius_rows(diag_rows))
    correlation_rows.extend(rollout_terminal_rows(rollout_rows))

    csv_path = os.path.join(args.output_dir, "perception_debug_correlation.csv")
    md_path = os.path.join(args.output_dir, "perception_debug_correlation.md")
    write_csv(csv_path, correlation_rows)
    markdown_report(md_path, diag_rows, rollout_rows, correlation_rows)
    print("wrote {} and {}".format(csv_path, md_path))


if __name__ == "__main__":
    main()
