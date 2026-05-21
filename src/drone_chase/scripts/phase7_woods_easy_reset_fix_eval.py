#!/usr/bin/env python3

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import Counter
from datetime import datetime


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "envs"))
if ENV_DIR not in sys.path:
    sys.path.insert(0, ENV_DIR)


DEFAULT_OUTPUT_DIR = "/home/whk/vf_ws/outputs/phase7/woods_easy_reset_fix"
DEFAULT_CONFIG = (
    "/home/whk/vf_ws/outputs/phase7/world0_ppo_from_bc_v2_10k_conservative_run4/config_effective.yaml"
)
PREVIOUS_SUMMARY = "/home/whk/vf_ws/outputs/phase7/woods_easy_failure_diagnosis/phase7_3b_failure_diagnosis_summary.json"
WOODS_EXPECTED_COUNT = 26


def finite(value, default=0.0):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(default)
    return value if math.isfinite(value) else float(default)


def mean(values):
    values = [finite(value, math.nan) for value in values]
    values = [value for value in values if math.isfinite(value)]
    return sum(values) / len(values) if values else 0.0


def rate(count, total):
    return float(count) / float(total) if total else 0.0


def point_segment_distance_xy(px, py, start, end):
    sx, sy = start[0], start[1]
    ex, ey = end[0], end[1]
    vx = ex - sx
    vy = ey - sy
    denom = vx * vx + vy * vy
    if denom <= 1e-9:
        return math.hypot(px - sx, py - sy)
    t = ((px - sx) * vx + (py - sy) * vy) / denom
    t = max(0.0, min(1.0, t))
    cx = sx + t * vx
    cy = sy + t * vy
    return math.hypot(px - cx, py - cy)


def clearance_ok(
    specs,
    uav_x=0.0,
    uav_y=0.0,
    target_x=4.0,
    target_y=0.0,
    uav_clearance=1.5,
    target_clearance=0.8,
):
    if len(specs) != WOODS_EXPECTED_COUNT:
        return False
    for spec in specs:
        start = spec.get("start")
        end = spec.get("end")
        if not start or not end:
            return False
        radius = finite(spec.get("radius", 0.0))
        if point_segment_distance_xy(uav_x, uav_y, start, end) - radius < uav_clearance - 1e-6:
            return False
        if point_segment_distance_xy(target_x, target_y, start, end) - radius < target_clearance - 1e-6:
            return False
    return True


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, data):
    with open(path, "w") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def load_previous_rate():
    if not os.path.exists(PREVIOUS_SUMMARY):
        return None
    try:
        with open(PREVIOUS_SUMMARY, "r") as handle:
            data = json.load(handle)
        return data.get("reset_summary", {}).get("reset_success_rate")
    except (OSError, ValueError):
        return None


def row_fieldnames():
    return [
        "reset_id",
        "success",
        "failure_reason",
        "uav_z_before",
        "uav_z_after",
        "altitude_recovery_used",
        "altitude_recovery_success",
        "altitude_recovery_duration",
        "safety_mode_before",
        "safety_mode_after",
        "target_visible_initial",
        "target_visible_after_gate",
        "target_respawn_attempts",
        "woods_spawn_success",
        "woods_element_count",
        "clearance_ok",
        "obstacle_risk_available",
        "target_state_available",
        "offboard_ok",
        "out_of_bounds",
        "height_violation",
        "reset_duration",
        "reset_pollution_detected",
    ]


def consecutive_failures(rows):
    best = 0
    current = 0
    for row in rows:
        if row["success"]:
            current = 0
        else:
            current += 1
            best = max(best, current)
    return best


def summarize(rows):
    total = len(rows)
    success_count = sum(1 for row in rows if row["success"])
    target_respawn_failures = sum(1 for row in rows if not row["target_visible_after_gate"])
    offboard_loss_count = sum(1 for row in rows if not row["offboard_ok"])
    out_of_bounds_count = sum(1 for row in rows if row["out_of_bounds"])
    height_violation_count = sum(1 for row in rows if row["height_violation"])
    summary = {
        "total_trials": int(total),
        "success_count": int(success_count),
        "reset_success_rate": rate(success_count, total),
        "previous_reset_success_rate": load_previous_rate(),
        "target_visible_initial_rate": rate(sum(1 for row in rows if row["target_visible_initial"]), total),
        "target_visible_after_gate_rate": rate(sum(1 for row in rows if row["target_visible_after_gate"]), total),
        "altitude_recovery_used_count": sum(1 for row in rows if row["altitude_recovery_used"]),
        "altitude_recovery_success_rate": rate(sum(1 for row in rows if row["altitude_recovery_success"]), total),
        "mean_altitude_recovery_duration": mean(row["altitude_recovery_duration"] for row in rows),
        "active_raw_timeout_before_count": sum(1 for row in rows if "ACTIVE:RAW_TIMEOUT" in row["safety_mode_before"]),
        "safety_mode_after_distribution": dict(Counter(row["safety_mode_after"] for row in rows)),
        "woods_spawn_success_rate": rate(sum(1 for row in rows if row["woods_spawn_success"]), total),
        "woods_element_count_mean": mean(row["woods_element_count"] for row in rows),
        "clearance_ok_rate": rate(sum(1 for row in rows if row["clearance_ok"]), total),
        "target_respawn_attempts_mean": mean(row["target_respawn_attempts"] for row in rows),
        "target_respawn_attempts_max": max([int(row["target_respawn_attempts"]) for row in rows], default=0),
        "target_respawn_failures": int(target_respawn_failures),
        "offboard_loss_count": int(offboard_loss_count),
        "out_of_bounds_count": int(out_of_bounds_count),
        "height_violation_count": int(height_violation_count),
        "reset_pollution_detected": any(row["reset_pollution_detected"] for row in rows),
        "mean_reset_duration": mean(row["reset_duration"] for row in rows),
        "consecutive_reset_failures": int(consecutive_failures(rows)),
    }
    summary["reset_gate_passed"] = bool(
        total >= 50
        and summary["reset_success_rate"] >= 0.95
        and summary["target_visible_after_gate_rate"] >= 0.90
        and summary["altitude_recovery_success_rate"] >= 0.95
        and summary["offboard_loss_count"] == 0
        and not summary["reset_pollution_detected"]
        and summary["out_of_bounds_count"] == 0
        and summary["height_violation_count"] == 0
        and summary["woods_spawn_success_rate"] >= 0.95
        and rate(summary["target_respawn_failures"], total) <= 0.05
    )
    return summary


def build_report(summary):
    previous = summary["previous_reset_success_rate"]
    previous_text = "unknown" if previous is None else "{:.4f}".format(float(previous))
    fields = dict(summary)
    fields["reset_pollution_detected"] = "true" if summary["reset_pollution_detected"] else "false"
    return """# Phase 7.3C Woods Easy Reset Fix Report

1. reset-only trials: {total_trials}
2. reset_success_rate: {reset_success_rate:.4f}
3. previous reset_success_rate from Phase 7.3B: {previous}
4. target_visible_initial_rate: {target_visible_initial_rate:.4f}
5. target_visible_after_gate: {target_visible_after_gate_rate:.4f}
6. altitude_recovery_used count: {altitude_recovery_used_count}
7. altitude_recovery_success_rate: {altitude_recovery_success_rate:.4f}
8. mean altitude_recovery_duration: {mean_altitude_recovery_duration:.4f}s
9. safety_mode ACTIVE:RAW_TIMEOUT before count: {active_raw_timeout_before_count}
10. safety_mode after recovery distribution: {safety_mode_after_distribution}
11. woods_spawn_success_rate: {woods_spawn_success_rate:.4f}
12. woods_element_count mean: {woods_element_count_mean:.4f}
13. clearance_ok rate: {clearance_ok_rate:.4f}
14. target_respawn_attempts mean/max: {target_respawn_attempts_mean:.4f}/{target_respawn_attempts_max}
15. target_respawn_failures: {target_respawn_failures}
16. OFFBOARD loss count: {offboard_loss_count}
17. out_of_bounds count: {out_of_bounds_count}
18. height_violation count: {height_violation_count}
19. reset_pollution_detected: {reset_pollution_detected}
20. mean reset_duration: {mean_reset_duration:.4f}s
21. consecutive reset failures: {consecutive_reset_failures}
22. 是否通过 reset gate: {reset_gate}
23. 是否允许进入 Phase 7.3D woods_easy policy zero-shot re-eval: {allow_73d}
24. 是否允许进入 woods curriculum training: no
25. 是否允许声称 woods 已通过: no

Notes:
- This phase only changes reset stabilization and reset-gate diagnostics.
- No policy training, fine-tuning, reward change, safety_filter_node.py control change, action mapping change, or registry policy update was performed.
""".format(
        previous=previous_text,
        reset_gate="yes" if summary["reset_gate_passed"] else "no",
        allow_73d="yes" if summary["reset_gate_passed"] else "no",
        **fields
    )


def run_trials(args):
    import rospy
    from gazebo_msgs.srv import GetWorldProperties
    from gazebo_chase_env import GazeboChaseEnv

    env = GazeboChaseEnv(
        config_path=args.config,
        reset_mode="soft",
        world_type="woods_easy",
        seed=args.seed_base,
        respawn_target_on_reset=False,
        woods_easy_num_trunks=8,
        woods_easy_num_branches=15,
        woods_easy_num_fallen=3,
        woods_easy_area_x_min=1.5,
        woods_easy_area_x_max=6.5,
        woods_easy_area_y_min=-3.0,
        woods_easy_area_y_max=3.0,
        woods_easy_uav_clearance=1.5,
        woods_easy_target_clearance=0.8,
        woods_skip_gazebo_reset_world=True,
        require_initial_target_visible=True,
        reset_target_z=1.2,
        reset_z_min=0.85,
        reset_z_max=1.65,
        reset_recovery_timeout=8.0,
        reset_recovery_rate=20.0,
        reset_max_vz=0.25,
        target_visible_gate_timeout=5.0,
        target_visible_min_consecutive=5,
        max_target_respawn_attempts=5,
    )
    rospy.wait_for_service("/gazebo/get_world_properties", timeout=10.0)
    world_proxy = rospy.ServiceProxy("/gazebo/get_world_properties", GetWorldProperties)
    rows = []
    try:
        for reset_id in range(int(args.trials)):
            seed = int(args.seed_base) + reset_id
            snap_before = env._snapshot()
            pose_before = snap_before.get("pose")
            uav_z_before = finite(pose_before.pose.position.z, math.nan) if pose_before is not None else math.nan
            safety_mode_before = str(snap_before.get("safety_mode", ""))
            env.config["seed"] = int(seed)
            obs, info = env.reset(seed=seed, options={"reset_mode": "soft"})
            snap_after = env._snapshot()
            pose_after = snap_after.get("pose")
            target_after = snap_after.get("target")
            risk_after = snap_after.get("risk")
            state_after = snap_after.get("mavros_state")
            try:
                model_names = list(world_proxy().model_names)
            except Exception:
                model_names = []
            try:
                woods_specs = rospy.get_param("/drone_chase/random_woods", [])
            except Exception:
                woods_specs = []
            try:
                red_ball_pose = rospy.get_param("/drone_chase/red_ball_pose", {})
            except Exception:
                red_ball_pose = {}
            woods_spawn_success = "random_woods" in model_names and len(woods_specs) == WOODS_EXPECTED_COUNT
            uav_x_after = finite(pose_after.pose.position.x, 0.0) if pose_after is not None else 0.0
            uav_y_after = finite(pose_after.pose.position.y, 0.0) if pose_after is not None else 0.0
            target_x = finite(red_ball_pose.get("x", info.get("reset_target_x", env.config.get("target_x", 4.0))))
            target_y = finite(red_ball_pose.get("y", info.get("reset_target_y", env.config.get("target_y", 0.0))))
            clearance = clearance_ok(
                woods_specs,
                uav_x=uav_x_after,
                uav_y=uav_y_after,
                target_x=target_x,
                target_y=target_y,
                uav_clearance=finite(env.config.get("woods_easy_uav_clearance", 1.5)),
                target_clearance=finite(env.config.get("woods_easy_target_clearance", 0.8)),
            )
            uav_z_after = finite(pose_after.pose.position.z, math.nan) if pose_after is not None else math.nan
            offboard_ok = bool(getattr(state_after, "mode", "") == "OFFBOARD" and getattr(state_after, "armed", False))
            target_state_available = target_after is not None
            obstacle_risk_available = risk_after is not None
            target_visible_after_gate = bool(target_after is not None and target_after.visible and info.get("target_visible_gate_success", False))
            out_of_bounds = bool(info.get("out_of_bounds", False))
            height_violation = bool(not (env.reset_z_min <= uav_z_after <= env.reset_z_max))
            reset_pollution = bool(not woods_spawn_success or not clearance or "reset_world" in str(info.get("reset_output_tail", "")))
            failures = []
            if not bool(info.get("training_ready", False)):
                failures.append("training_not_ready")
            if not target_visible_after_gate:
                failures.append("target_not_visible_after_gate")
            if not bool(info.get("altitude_recovery_success", False)):
                failures.append("altitude_recovery_failed")
            if not woods_spawn_success:
                failures.append("woods_spawn_failed")
            if not clearance:
                failures.append("clearance_failed")
            if not target_state_available:
                failures.append("target_state_missing")
            if not obstacle_risk_available:
                failures.append("obstacle_risk_missing")
            if not offboard_ok:
                failures.append("offboard_lost")
            if out_of_bounds:
                failures.append("out_of_bounds")
            if height_violation:
                failures.append("height_violation")
            if reset_pollution:
                failures.append("reset_pollution")
            success = bool(not failures)
            row = {
                "reset_id": int(reset_id),
                "success": success,
                "failure_reason": "ok" if success else ",".join(failures),
                "uav_z_before": uav_z_before,
                "uav_z_after": uav_z_after,
                "altitude_recovery_used": bool(info.get("altitude_recovery_used", False)),
                "altitude_recovery_success": bool(info.get("altitude_recovery_success", False)),
                "altitude_recovery_duration": finite(info.get("altitude_recovery_duration", 0.0)),
                "safety_mode_before": safety_mode_before,
                "safety_mode_after": str(info.get("safety_mode", snap_after.get("safety_mode", ""))),
                "target_visible_initial": bool(info.get("target_visible_initial_before_gate", False)),
                "target_visible_after_gate": target_visible_after_gate,
                "target_respawn_attempts": int(info.get("target_respawn_attempts", 0)),
                "woods_spawn_success": bool(woods_spawn_success),
                "woods_element_count": int(len(woods_specs)),
                "clearance_ok": bool(clearance),
                "obstacle_risk_available": bool(obstacle_risk_available),
                "target_state_available": bool(target_state_available),
                "offboard_ok": bool(offboard_ok),
                "out_of_bounds": bool(out_of_bounds),
                "height_violation": bool(height_violation),
                "reset_duration": finite(info.get("reset_duration", 0.0)),
                "reset_pollution_detected": bool(reset_pollution),
            }
            rows.append(row)
            print(
                "reset_id={} success={} z_before={:.3f} z_after={:.3f} visible_initial={} visible_gate={} attempts={} reason={}".format(
                    reset_id,
                    row["success"],
                    row["uav_z_before"],
                    row["uav_z_after"],
                    row["target_visible_initial"],
                    row["target_visible_after_gate"],
                    row["target_respawn_attempts"],
                    row["failure_reason"],
                )
            )
    finally:
        env.close()
    return rows


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Phase 7.3C woods_easy reset fix eval.")
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--seed-base", type=int, default=73500)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser


def main():
    args = build_arg_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    rows = run_trials(args)
    summary = summarize(rows)
    summary["created_at"] = datetime.now().isoformat(timespec="seconds")
    summary["output_dir"] = os.path.abspath(args.output_dir)

    csv_path = os.path.join(args.output_dir, "woods_reset_fix_eval.csv")
    summary_path = os.path.join(args.output_dir, "woods_reset_fix_summary.json")
    report_path = os.path.join(args.output_dir, "phase7_3c_reset_fix_report.md")
    write_csv(csv_path, rows, row_fieldnames())
    write_json(summary_path, summary)
    with open(report_path, "w") as handle:
        handle.write(build_report(summary))
    print("wrote {}".format(csv_path))
    print("wrote {}".format(summary_path))
    print("wrote {}".format(report_path))
    print("reset_gate_passed={}".format(summary["reset_gate_passed"]))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("phase7_woods_easy_reset_fix_eval failed: {}".format(exc), file=sys.stderr)
        sys.exit(1)
