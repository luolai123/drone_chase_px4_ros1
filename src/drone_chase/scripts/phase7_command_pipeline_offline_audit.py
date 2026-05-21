#!/usr/bin/env python3

import argparse
import csv
import math
import os
import sys


DEFAULT_RUN_DIR = "/home/whk/vf_ws/outputs/phase7/world0_20k_rewardfix_v2_run2"
DEFAULT_EVAL_ROLLOUTS = os.path.join(DEFAULT_RUN_DIR, "eval_rollouts.csv")
DEFAULT_CHECKPOINT_SWEEP = os.path.join(DEFAULT_RUN_DIR, "checkpoint_sweep.csv")
DEFAULT_OUTPUT_CSV = os.path.join(DEFAULT_RUN_DIR, "command_pipeline_offline_audit.csv")
DEFAULT_OUTPUT_MD = os.path.join(DEFAULT_RUN_DIR, "command_pipeline_offline_audit.md")


def load_csv(path):
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas is required for offline audit") from exc
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def finite_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return math.nan
    return value if math.isfinite(value) else math.nan


def stats(series):
    try:
        import pandas as pd
    except ImportError:
        pd = None
    if pd is not None:
        values = pd.to_numeric(series, errors="coerce").dropna()
        if len(values) == 0:
            return {"count": 0, "mean": math.nan, "std": math.nan, "min": math.nan, "max": math.nan}
        return {
            "count": int(len(values)),
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    values = [finite_float(v) for v in series]
    values = [v for v in values if math.isfinite(v)]
    if not values:
        return {"count": 0, "mean": math.nan, "std": math.nan, "min": math.nan, "max": math.nan}
    mean = sum(values) / float(len(values))
    var = sum((value - mean) ** 2.0 for value in values) / float(len(values))
    return {"count": len(values), "mean": mean, "std": var ** 0.5, "min": min(values), "max": max(values)}


def ratio(mask):
    if len(mask) == 0:
        return math.nan
    return float(mask.sum()) / float(len(mask))


def map_raw_vx(action_vx, max_vx=0.5, min_vx=-0.2):
    action_vx = float(action_vx)
    return action_vx * max_vx if action_vx >= 0.0 else action_vx * abs(min_vx)


def add_summary(rows, section, label, values=None, value=None, note=""):
    row = {
        "section": section,
        "label": label,
        "count": "",
        "mean": "",
        "std": "",
        "min": "",
        "max": "",
        "value": "",
        "note": note,
    }
    if values is not None:
        result = stats(values)
        row.update({key: result[key] for key in ("count", "mean", "std", "min", "max")})
    if value is not None:
        row["value"] = value
    rows.append(row)


def write_csv(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fieldnames = ["section", "label", "count", "mean", "std", "min", "max", "value", "note"]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_value(value):
    if value is None:
        return "N/A"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(value):
        return "N/A"
    return "{:.4f}".format(value)


def write_markdown(path, lines):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Offline audit for Phase 7 command pipeline logs.")
    parser.add_argument("--eval-rollouts", default=DEFAULT_EVAL_ROLLOUTS)
    parser.add_argument("--checkpoint-sweep", default=DEFAULT_CHECKPOINT_SWEEP)
    parser.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-md", default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--max-vx", type=float, default=0.5)
    parser.add_argument("--min-vx", type=float, default=-0.2)
    return parser


def main():
    args = build_arg_parser().parse_args()
    import pandas as pd

    eval_df = load_csv(args.eval_rollouts)
    sweep_df = load_csv(args.checkpoint_sweep) if os.path.exists(args.checkpoint_sweep) else pd.DataFrame()
    rows = []

    action_col = "policy_action_vx" if "policy_action_vx" in eval_df.columns else "action_vx"
    depth_col = "target_depth" if "target_depth" in eval_df.columns else "target_distance"
    if action_col not in eval_df.columns:
        raise RuntimeError("No action_vx/policy_action_vx column in {}".format(args.eval_rollouts))
    if depth_col not in eval_df.columns:
        raise RuntimeError("No target_depth/target_distance column in {}".format(args.eval_rollouts))

    eval_df["policy_action_vx"] = pd.to_numeric(eval_df[action_col], errors="coerce")
    eval_df["mapped_raw_vx_body"] = eval_df["policy_action_vx"].map(lambda value: map_raw_vx(value, args.max_vx, args.min_vx))
    eval_df["target_depth"] = pd.to_numeric(eval_df[depth_col], errors="coerce")
    eval_df["target_depth_delta"] = eval_df.groupby("episode")["target_depth"].diff()

    if "target_visible" in eval_df.columns:
        eval_df["target_visible_bool"] = eval_df["target_visible"].astype(str).str.lower().isin(("true", "1", "yes"))
    else:
        eval_df["target_visible_bool"] = False

    if "drone_z" in eval_df.columns:
        z = pd.to_numeric(eval_df["drone_z"], errors="coerce")
        eval_df["z_bucket"] = pd.cut(z, [-math.inf, 0.6, 2.5, math.inf], labels=["z < 0.6", "0.6 <= z <= 2.5", "z > 2.5"])
    else:
        eval_df["z_bucket"] = "missing"
    if "front_q05_depth" in eval_df.columns:
        front = pd.to_numeric(eval_df["front_q05_depth"], errors="coerce")
        eval_df["front_bucket"] = pd.cut(front, [-math.inf, 0.5, 0.8, math.inf], labels=["front < 0.5", "0.5 <= front < 0.8", "front >= 0.8"])
    else:
        eval_df["front_bucket"] = "missing"

    add_summary(rows, "eval_rollouts", "policy_action_vx", eval_df["policy_action_vx"])
    add_summary(rows, "eval_rollouts", "mapped_raw_vx_body", eval_df["mapped_raw_vx_body"])
    add_summary(rows, "eval_rollouts", "target_depth_delta", eval_df["target_depth_delta"])
    add_summary(rows, "eval_rollouts", "action_vx_positive_ratio", value=ratio(eval_df["policy_action_vx"] > 0.0))
    add_summary(rows, "eval_rollouts", "filtered_vx_body_available", value=False, note="run2 eval_rollouts has no body-frame filtered command fields")

    if "safety_mode" in eval_df.columns:
        for safety_mode, group in eval_df.groupby("safety_mode", dropna=False):
            add_summary(rows, "by_safety_mode", str(safety_mode), group["mapped_raw_vx_body"])
    for visible, group in eval_df.groupby("target_visible_bool", dropna=False):
        add_summary(rows, "by_target_visible", str(visible), group["mapped_raw_vx_body"])
    for bucket, group in eval_df.groupby("z_bucket", dropna=False):
        add_summary(rows, "by_drone_z", str(bucket), group["mapped_raw_vx_body"])
    for bucket, group in eval_df.groupby("front_bucket", dropna=False):
        add_summary(rows, "by_front_q05", str(bucket), group["mapped_raw_vx_body"])

    decreasing = eval_df[eval_df["target_depth_delta"] < 0.0]
    increasing = eval_df[eval_df["target_depth_delta"] > 0.0]
    add_summary(rows, "depth_decreasing", "policy_action_vx", decreasing["policy_action_vx"])
    add_summary(rows, "depth_decreasing", "mapped_raw_vx_body", decreasing["mapped_raw_vx_body"])
    add_summary(rows, "depth_increasing", "policy_action_vx", increasing["policy_action_vx"])
    add_summary(rows, "depth_increasing", "mapped_raw_vx_body", increasing["mapped_raw_vx_body"])

    if "action_yaw" in eval_df.columns:
        eval_df["policy_action_yaw"] = pd.to_numeric(eval_df["action_yaw"], errors="coerce")
        add_summary(rows, "yaw", "policy_action_yaw", eval_df["policy_action_yaw"])
        if "target_u" in eval_df.columns:
            target_u = pd.to_numeric(eval_df["target_u"], errors="coerce")
            corr = target_u.corr(eval_df["policy_action_yaw"])
            add_summary(rows, "yaw", "corr_target_u_action_yaw", value=corr)

    if not sweep_df.empty:
        add_summary(rows, "checkpoint_sweep", "action_vx_mean", sweep_df.get("action_vx_mean", []))
        add_summary(rows, "checkpoint_sweep", "mapped_raw_vx_body_mean_from_action_mean", sweep_df.get("action_vx_mean", pd.Series(dtype=float)).map(lambda value: map_raw_vx(value, args.max_vx, args.min_vx)))
        if "filtered_vx_mean" in sweep_df.columns:
            add_summary(
                rows,
                "checkpoint_sweep",
                "legacy_filtered_vx_mean_is_published_vx_world",
                sweep_df["filtered_vx_mean"],
                note="checkpoint_sweep subscribes to /mavros/setpoint_velocity/cmd_vel, which safety_filter publishes in map/world frame",
            )
        if "target_depth_lt_1_count" in sweep_df.columns:
            add_summary(rows, "checkpoint_sweep", "target_depth_lt_1_count", sweep_df["target_depth_lt_1_count"])

    write_csv(args.output_csv, rows)

    final_eval = eval_df[eval_df.get("eval_step", pd.Series([None] * len(eval_df))) == eval_df.get("eval_step", pd.Series([None])).max()] if "eval_step" in eval_df.columns else eval_df
    action_stats = stats(final_eval["policy_action_vx"])
    raw_stats = stats(final_eval["mapped_raw_vx_body"])
    depth_drop_raw = stats(decreasing["mapped_raw_vx_body"])
    depth_rise_raw = stats(increasing["mapped_raw_vx_body"])
    action_positive_ratio = ratio(final_eval["policy_action_vx"] > 0.0)

    best_sweep = None
    if not sweep_df.empty:
        best_sweep = sweep_df.sort_values(["success_rate", "min_distance_mean"], ascending=[False, True]).iloc[0].to_dict()

    md = [
        "# Phase 7.1E Offline Command Pipeline Audit",
        "",
        "## Inputs",
        "",
        "- eval_rollouts: `{}`".format(args.eval_rollouts),
        "- checkpoint_sweep: `{}`".format(args.checkpoint_sweep),
        "",
        "## Field Semantics",
        "",
        "- `/safety_filter/debug_cmd_raw` is body-frame (`frame_id=base_link`) from `safety_filter_node.publish_body_debug()`.",
        "- `/safety_filter/debug_cmd_filtered` is body-frame (`frame_id=base_link`) after clamp/rate-limit/safety protection.",
        "- `/mavros/setpoint_velocity/cmd_vel` is world-frame ENU/map (`frame_id=map`) after `body_to_world()`.",
        "- `world0_20k_rewardfix_v2_run2/eval_rollouts.csv` does not contain debug raw/filtered command fields.",
        "- `checkpoint_sweep.csv` column `filtered_vx_*` is legacy naming for `/mavros/setpoint_velocity/cmd_vel.twist.linear.x`, so it is `published_vx_world`, not `debug_filtered_vx_body`.",
        "",
        "## Key Numbers",
        "",
        "- final eval policy_action_vx mean/std/min/max: {}/{}/{}/{}".format(
            format_value(action_stats["mean"]),
            format_value(action_stats["std"]),
            format_value(action_stats["min"]),
            format_value(action_stats["max"]),
        ),
        "- final eval mapped_raw_vx_body mean/std/min/max: {}/{}/{}/{}".format(
            format_value(raw_stats["mean"]),
            format_value(raw_stats["std"]),
            format_value(raw_stats["min"]),
            format_value(raw_stats["max"]),
        ),
        "- final eval action_vx > 0 ratio: {}".format(format_value(action_positive_ratio)),
        "- target_depth decreasing mapped_raw_vx_body mean: {}".format(format_value(depth_drop_raw["mean"])),
        "- target_depth increasing mapped_raw_vx_body mean: {}".format(format_value(depth_rise_raw["mean"])),
    ]
    if best_sweep is not None:
        md.extend(
            [
                "- best sweep checkpoint: {}".format(best_sweep.get("checkpoint_step")),
                "- best sweep success_rate: {}".format(format_value(best_sweep.get("success_rate"))),
                "- best sweep legacy filtered_vx_mean (published_vx_world): {}".format(format_value(best_sweep.get("filtered_vx_mean"))),
                "- best sweep action_vx_mean: {}".format(format_value(best_sweep.get("action_vx_mean"))),
            ]
        )
    md.extend(
        [
            "",
            "## Conclusions",
            "",
            "- The old `filtered_vx` interpretation was likely wrong: available code shows it was logged from the MAVROS world-frame setpoint, not the safety filter body-frame debug topic.",
            "- The run2 policy action magnitude is small. A final-eval mean action_vx of {} maps to about {} m/s body forward command.".format(
                format_value(action_stats["mean"]),
                format_value(raw_stats["mean"]),
            ),
            "- With `dt=0.1` and 250 steps, a mean raw forward speed near {} m/s has limited room to close a 3-5m initial gap, especially when yaw/visibility/reset effects are present.".format(format_value(raw_stats["mean"])),
            "- Offline data alone cannot prove whether safety_filter suppresses body-frame vx because run2 eval_rollouts lacks `/safety_filter/debug_cmd_filtered` fields. The online trace/intervention scripts fill that gap.",
        ]
    )
    write_markdown(args.output_md, md)
    print("wrote {}".format(args.output_csv))
    print("wrote {}".format(args.output_md))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("phase7_command_pipeline_offline_audit failed: {}".format(exc), file=sys.stderr)
        sys.exit(1)
