#!/usr/bin/env python3

import argparse
import csv
import os
import sys


def read_rows(path):
    with open(path, "r") as handle:
        return list(csv.DictReader(handle))


def numeric(rows, key, default=0.0):
    values = []
    for row in rows:
        try:
            values.append(float(row.get(key, default)))
        except (TypeError, ValueError):
            values.append(float(default))
    return values


def x_values(rows):
    if rows and "episode" in rows[0]:
        return numeric(rows, "episode")
    return list(range(len(rows)))


def plot_series(plt, x, y, title, ylabel, path):
    fig = plt.figure(figsize=(8, 4.5))
    ax = fig.add_subplot(111)
    ax.plot(x, y, linewidth=1.8)
    ax.set_title(title)
    ax.set_xlabel("episode")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Plot Phase 7 training/eval CSV metrics.")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output-dir", default=None)
    return parser


def main():
    args = build_arg_parser().parse_args()
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for plotting") from exc

    rows = read_rows(args.csv)
    if not rows:
        raise RuntimeError("CSV has no rows: {}".format(args.csv))
    output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.csv))
    os.makedirs(output_dir, exist_ok=True)

    x = x_values(rows)
    reward_key = "episode_reward" if "episode_reward" in rows[0] else "reward"
    length_key = "episode_length" if "episode_length" in rows[0] else "length"
    distance_key = "final_target_distance"

    plot_series(plt, x, numeric(rows, reward_key), "Reward", reward_key, os.path.join(output_dir, "reward_curve.png"))
    plot_series(
        plt,
        x,
        numeric(rows, length_key),
        "Episode Length",
        length_key,
        os.path.join(output_dir, "episode_length_curve.png"),
    )
    plot_series(
        plt,
        x,
        [1.0 if str(row.get("success", "")).lower() == "true" else 0.0 for row in rows],
        "Success",
        "success",
        os.path.join(output_dir, "success_curve.png"),
    )
    plot_series(
        plt,
        x,
        numeric(rows, distance_key),
        "Final Target Distance",
        distance_key,
        os.path.join(output_dir, "final_distance_curve.png"),
    )
    print("wrote plots to {}".format(output_dir))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("phase7_plot_training_csv failed: {}".format(exc), file=sys.stderr)
        sys.exit(1)
