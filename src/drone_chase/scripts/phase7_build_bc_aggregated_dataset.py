#!/usr/bin/env python3

import argparse
import json
import math
import os
import sys

import numpy as np


DEFAULT_EXPERT = "/home/whk/vf_ws/outputs/phase7/bc_world0_demos/expert_demos_filtered.npz"
DEFAULT_DAGGER_DIR = "/home/whk/vf_ws/outputs/phase7/bc_world0_dagger"
DEFAULT_OUTPUT = "/home/whk/vf_ws/outputs/phase7/bc_world0_dagger/bc_aggregated_dataset.npz"
DEFAULT_SUMMARY = "/home/whk/vf_ws/outputs/phase7/bc_world0_dagger/bc_aggregated_summary.json"


def load_npz(path, source_name):
    data = np.load(path, allow_pickle=True)
    obs = np.asarray(data["obs"], dtype=np.float32)
    if "actions" in data:
        actions = np.asarray(data["actions"], dtype=np.float32)
    elif "expert_actions" in data:
        actions = np.asarray(data["expert_actions"], dtype=np.float32)
    else:
        raise ValueError("{} has neither actions nor expert_actions".format(path))
    n_rows = int(obs.shape[0])
    done_reasons = np.asarray(data["done_reasons"], dtype=object) if "done_reasons" in data else np.asarray([""] * n_rows, dtype=object)
    safety_modes = np.asarray(data["safety_modes"], dtype=object) if "safety_modes" in data else np.asarray([""] * n_rows, dtype=object)
    success_flags = np.asarray(data["success_flags"], dtype=np.bool_) if "success_flags" in data else np.zeros((n_rows,), dtype=np.bool_)
    return {
        "path": os.path.abspath(path),
        "source": source_name,
        "obs": obs,
        "actions": np.clip(actions, -1.0, 1.0),
        "done_reasons": done_reasons.astype(object),
        "safety_modes": safety_modes.astype(object),
        "success_flags": success_flags.astype(np.bool_),
    }


def bucket_for(obs_row):
    visible = bool(obs_row[0] > 0.5)
    depth = float(obs_row[4])
    if not visible:
        return "lost"
    if depth < 0.8:
        return "capture"
    if depth < 1.5:
        return "near"
    if depth <= 3.0:
        return "mid"
    return "far"


def valid_mask(obs, actions, done_reasons, safety_modes):
    finite = np.all(np.isfinite(obs), axis=1) & np.all(np.isfinite(actions), axis=1)
    bad_done = np.asarray([str(reason) in ("out_of_bounds", "height_violation") for reason in done_reasons], dtype=np.bool_)
    bad_safety = np.asarray(["EMERGENCY" in str(mode) or "DEPTH_STOP" in str(mode) for mode in safety_modes], dtype=np.bool_)
    return finite & (~bad_done) & (~bad_safety)


def choose_balanced_indices(obs, success_flags, seed, lost_max_ratio):
    rng = np.random.default_rng(int(seed))
    buckets = np.asarray([bucket_for(row) for row in obs], dtype=object)
    visible_indices = np.where(buckets != "lost")[0]
    lost_indices = np.where(buckets == "lost")[0]

    selected = list(visible_indices)
    visible_bucket_counts = {bucket: int(np.sum(buckets == bucket)) for bucket in ("far", "mid", "near", "capture")}
    target = max(visible_bucket_counts.values()) if visible_bucket_counts else 0
    target = max(target, 1)
    for bucket in ("far", "mid", "near", "capture"):
        bucket_indices = np.where(buckets == bucket)[0]
        if len(bucket_indices) == 0:
            continue
        deficit = max(0, target - len(bucket_indices))
        if deficit > 0:
            success_bucket = bucket_indices[success_flags[bucket_indices]]
            pool = success_bucket if len(success_bucket) else bucket_indices
            selected.extend(rng.choice(pool, size=deficit, replace=True).tolist())

    visible_selected_count = len(selected)
    max_lost = int(math.floor((float(lost_max_ratio) / max(1.0e-6, 1.0 - float(lost_max_ratio))) * visible_selected_count))
    if len(lost_indices) > 0 and max_lost > 0:
        lost_take = min(len(lost_indices), max_lost)
        selected.extend(rng.choice(lost_indices, size=lost_take, replace=False).tolist())
    rng.shuffle(selected)
    return np.asarray(selected, dtype=np.int64), buckets


def counts(values):
    result = {}
    for value in values:
        key = str(value)
        result[key] = result.get(key, 0) + 1
    return result


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Build balanced BC aggregated dataset from expert and DAgger batches.")
    parser.add_argument("--expert", default=DEFAULT_EXPERT)
    parser.add_argument("--dagger", nargs="*", default=None)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", default=DEFAULT_SUMMARY)
    parser.add_argument("--lost-max-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=471)
    return parser


def main():
    args = build_arg_parser().parse_args()
    dagger_paths = args.dagger
    if dagger_paths is None:
        dagger_paths = [
            os.path.join(DEFAULT_DAGGER_DIR, "dagger_batch1_beta05.npz"),
            os.path.join(DEFAULT_DAGGER_DIR, "dagger_batch2_beta025.npz"),
            os.path.join(DEFAULT_DAGGER_DIR, "dagger_batch3_beta0.npz"),
        ]
    datasets = [load_npz(args.expert, "expert")]
    for index, path in enumerate(dagger_paths, start=1):
        datasets.append(load_npz(path, "dagger{}".format(index)))

    obs_parts = []
    action_parts = []
    done_parts = []
    safety_parts = []
    success_parts = []
    source_parts = []
    input_summaries = []
    removed_nonfinite = 0
    removed_bad_done = 0
    removed_bad_safety = 0
    for dataset in datasets:
        obs = dataset["obs"]
        actions = dataset["actions"]
        done_reasons = dataset["done_reasons"]
        safety_modes = dataset["safety_modes"]
        success_flags = dataset["success_flags"]
        finite = np.all(np.isfinite(obs), axis=1) & np.all(np.isfinite(actions), axis=1)
        bad_done = np.asarray([str(reason) in ("out_of_bounds", "height_violation") for reason in done_reasons], dtype=np.bool_)
        bad_safety = np.asarray(["EMERGENCY" in str(mode) or "DEPTH_STOP" in str(mode) for mode in safety_modes], dtype=np.bool_)
        mask = valid_mask(obs, actions, done_reasons, safety_modes)
        removed_nonfinite += int((~finite).sum())
        removed_bad_done += int((finite & bad_done).sum())
        removed_bad_safety += int((finite & (~bad_done) & bad_safety).sum())
        obs_parts.append(obs[mask])
        action_parts.append(actions[mask])
        done_parts.append(done_reasons[mask])
        safety_parts.append(safety_modes[mask])
        success_parts.append(success_flags[mask])
        source_parts.append(np.asarray([dataset["source"]] * int(mask.sum()), dtype=object))
        input_summaries.append({"path": dataset["path"], "source": dataset["source"], "rows": int(len(obs)), "valid_rows": int(mask.sum())})

    obs_all = np.concatenate(obs_parts, axis=0)
    actions_all = np.concatenate(action_parts, axis=0)
    done_all = np.concatenate(done_parts, axis=0)
    safety_all = np.concatenate(safety_parts, axis=0)
    success_all = np.concatenate(success_parts, axis=0)
    source_all = np.concatenate(source_parts, axis=0)
    selected, raw_buckets = choose_balanced_indices(obs_all, success_all, args.seed, args.lost_max_ratio)

    obs = obs_all[selected]
    actions = actions_all[selected]
    done_reasons = done_all[selected]
    safety_modes = safety_all[selected]
    success_flags = success_all[selected]
    sources = source_all[selected]
    buckets = np.asarray([bucket_for(row) for row in obs], dtype=object)
    sample_weights = np.ones((len(obs),), dtype=np.float32)
    bucket_counts = counts(buckets)
    for bucket, count in bucket_counts.items():
        if count > 0:
            sample_weights[buckets == bucket] = float(len(obs)) / float(len(bucket_counts) * count)

    output = os.path.abspath(args.output)
    summary_path = os.path.abspath(args.summary)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    metadata = {
        "inputs": input_summaries,
        "seed": int(args.seed),
        "lost_max_ratio": float(args.lost_max_ratio),
        "raw_rows": int(len(obs_all)),
        "selected_rows": int(len(obs)),
    }
    np.savez_compressed(
        output,
        obs=obs.astype(np.float32),
        actions=actions.astype(np.float32),
        expert_actions=actions.astype(np.float32),
        episode_ids=np.arange(len(obs), dtype=np.int32),
        done_reasons=done_reasons,
        safety_modes=safety_modes,
        success_flags=success_flags.astype(np.bool_),
        source=sources,
        bucket=buckets,
        sample_weights=sample_weights,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    lost_count = bucket_counts.get("lost", 0)
    near_capture_count = bucket_counts.get("near", 0) + bucket_counts.get("capture", 0)
    summary = {
        "output": output,
        "inputs": input_summaries,
        "raw_valid_rows": int(len(obs_all)),
        "selected_rows": int(len(obs)),
        "bucket_counts_raw": counts(raw_buckets),
        "bucket_counts_final": bucket_counts,
        "source_counts_final": counts(sources),
        "lost_recovery_ratio": float(lost_count) / float(max(1, len(obs))),
        "near_capture_ratio": float(near_capture_count) / float(max(1, len(obs))),
        "removed_nonfinite_rows": int(removed_nonfinite),
        "removed_bad_done_rows": int(removed_bad_done),
        "removed_bad_safety_rows": int(removed_bad_safety),
    }
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print("wrote {}".format(output))
    print("wrote {}".format(summary_path))
    print("selected_rows={} buckets={}".format(len(obs), bucket_counts))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("phase7_build_bc_aggregated_dataset failed: {}".format(exc), file=sys.stderr)
        sys.exit(1)
