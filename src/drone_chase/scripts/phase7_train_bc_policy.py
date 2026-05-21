#!/usr/bin/env python3

import argparse
import csv
import json
import math
import os
import sys

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "models"))
if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)

from bc_policy import BCPolicy


DEFAULT_DATASET = "/home/whk/vf_ws/outputs/phase7/bc_world0_demos/expert_demos_filtered.npz"
DEFAULT_OUTPUT_DIR = "/home/whk/vf_ws/outputs/phase7/bc_world0"


def require_torch():
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for BC training") from exc


def load_dataset(path):
    data = np.load(path, allow_pickle=True)
    obs = np.asarray(data["obs"], dtype=np.float32)
    if "actions" in data:
        actions = np.asarray(data["actions"], dtype=np.float32)
    elif "expert_actions" in data:
        actions = np.asarray(data["expert_actions"], dtype=np.float32)
    else:
        raise ValueError("{} has neither actions nor expert_actions".format(path))
    sample_weights = np.asarray(data["sample_weights"], dtype=np.float32) if "sample_weights" in data else np.ones((len(obs),), dtype=np.float32)
    if obs.ndim != 2 or obs.shape[1] != 20:
        raise ValueError("expected obs shape [N,20], got {}".format(obs.shape))
    if actions.ndim != 2 or actions.shape[1] != 4:
        raise ValueError("expected actions shape [N,4], got {}".format(actions.shape))
    finite = np.all(np.isfinite(obs), axis=1) & np.all(np.isfinite(actions), axis=1)
    obs = obs[finite]
    actions = np.clip(actions[finite], -1.0, 1.0)
    sample_weights = np.clip(sample_weights[finite], 1.0e-6, 1.0e6)
    if len(obs) < 16:
        raise ValueError("dataset too small after finite filter: {} rows".format(len(obs)))
    metadata = {}
    if "metadata_json" in data:
        try:
            metadata = json.loads(str(data["metadata_json"]))
        except Exception:
            metadata = {}
    return obs, actions, sample_weights, metadata, int((~finite).sum())


def split_indices(n_rows, val_ratio, seed):
    rng = np.random.default_rng(int(seed))
    indices = rng.permutation(n_rows)
    val_size = max(1, int(round(float(val_ratio) * n_rows)))
    val_size = min(val_size, n_rows - 1)
    return indices[val_size:], indices[:val_size]


def array_stats(arr):
    if len(arr) == 0:
        return {
            "mean": [math.nan] * 4,
            "std": [math.nan] * 4,
            "min": [math.nan] * 4,
            "max": [math.nan] * 4,
        }
    return {
        "mean": np.mean(arr, axis=0).astype(float).tolist(),
        "std": np.std(arr, axis=0).astype(float).tolist(),
        "min": np.min(arr, axis=0).astype(float).tolist(),
        "max": np.max(arr, axis=0).astype(float).tolist(),
    }


def sign_agreement(pred, target, dim):
    pred_sign = pred[:, dim] >= 0.0
    target_sign = target[:, dim] >= 0.0
    return float(np.mean(pred_sign == target_sign))


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "y", "on")


def write_curve(log_rows, output_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    epochs = [row["epoch"] for row in log_rows]
    train_loss = [row["train_loss"] for row in log_rows]
    val_loss = [row["val_loss"] for row in log_rows]
    plt.figure(figsize=(8, 4.5))
    plt.plot(epochs, train_loss, label="train")
    plt.plot(epochs, val_loss, label="val")
    plt.xlabel("epoch")
    plt.ylabel("MSE")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=140)
    plt.close()
    return True


def evaluate_policy(policy, obs, actions, device):
    import torch

    policy.eval()
    with torch.no_grad():
        pred = policy(torch.as_tensor(obs, dtype=torch.float32, device=device)).cpu().numpy()
    mae = np.mean(np.abs(pred - actions), axis=0)
    mse = float(np.mean((pred - actions) ** 2))
    return {
        "mse": mse,
        "mae": mae.astype(float).tolist(),
        "vx_mae": float(mae[0]),
        "vy_mae": float(mae[1]),
        "vz_mae": float(mae[2]),
        "yaw_mae": float(mae[3]),
        "action_vx_sign_agreement": sign_agreement(pred, actions, 0),
        "action_yaw_sign_agreement": sign_agreement(pred, actions, 3),
        "expert_action_stats": array_stats(actions),
        "bc_action_stats": array_stats(pred),
    }


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Train a Phase 7.1F behavior cloning policy.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--activation", choices=["tanh", "relu"], default="tanh")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--normalize-obs", default="true")
    parser.add_argument("--dataset-balance", default="false")
    parser.add_argument("--loss-weight-vx", type=float, default=1.0)
    parser.add_argument("--loss-weight-vy", type=float, default=1.0)
    parser.add_argument("--loss-weight-vz", type=float, default=1.0)
    parser.add_argument("--loss-weight-yaw", type=float, default=1.0)
    return parser


def main():
    args = build_arg_parser().parse_args()
    require_torch()
    import torch
    from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

    torch.manual_seed(int(args.seed))
    obs, actions, sample_weights, metadata, removed_rows = load_dataset(args.dataset)
    train_idx, val_idx = split_indices(len(obs), args.val_ratio, args.seed)
    train_obs = obs[train_idx]
    train_actions = actions[train_idx]
    train_weights = sample_weights[train_idx]
    val_obs = obs[val_idx]
    val_actions = actions[val_idx]
    normalize_obs = str_to_bool(args.normalize_obs)
    if normalize_obs:
        obs_mean = train_obs.mean(axis=0)
        obs_std = np.maximum(train_obs.std(axis=0), 1.0e-6)
    else:
        obs_mean = np.zeros((20,), dtype=np.float32)
        obs_std = np.ones((20,), dtype=np.float32)

    device = torch.device(args.device)
    loss_weights = torch.as_tensor(
        [args.loss_weight_vx, args.loss_weight_vy, args.loss_weight_vz, args.loss_weight_yaw],
        dtype=torch.float32,
        device=device,
    )
    policy = BCPolicy(
        obs_dim=20,
        action_dim=4,
        hidden_dim=int(args.hidden_dim),
        activation=str(args.activation),
        obs_mean=obs_mean,
        obs_std=obs_std,
    ).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=float(args.lr))
    dataset = TensorDataset(
        torch.as_tensor(train_obs, dtype=torch.float32),
        torch.as_tensor(train_actions, dtype=torch.float32),
    )
    if str_to_bool(args.dataset_balance):
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(train_weights, dtype=torch.double),
            num_samples=len(train_weights),
            replacement=True,
        )
        loader = DataLoader(dataset, batch_size=int(args.batch_size), sampler=sampler, drop_last=False)
    else:
        loader = DataLoader(dataset, batch_size=int(args.batch_size), shuffle=True, drop_last=False)

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    best_path = os.path.join(output_dir, "bc_policy_best.pt")
    final_path = os.path.join(output_dir, "bc_policy_final.pt")
    csv_path = os.path.join(output_dir, "bc_train_log.csv")
    curve_path = os.path.join(output_dir, "bc_train_curve.png")
    summary_path = os.path.join(output_dir, "bc_summary.json")
    norm_path = os.path.join(output_dir, "obs_norm_stats.npz")
    np.savez_compressed(norm_path, obs_mean=obs_mean.astype(np.float32), obs_std=obs_std.astype(np.float32))

    log_rows = []
    best_val_loss = float("inf")
    best_epoch = -1
    for epoch in range(1, int(args.epochs) + 1):
        policy.train()
        train_loss_sum = 0.0
        train_count = 0
        for batch_obs, batch_actions in loader:
            batch_obs = batch_obs.to(device)
            batch_actions = batch_actions.to(device)
            pred = policy(batch_obs)
            loss = torch.mean(((pred - batch_actions) ** 2) * loss_weights)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_size = int(batch_obs.shape[0])
            train_loss_sum += float(loss.item()) * batch_size
            train_count += batch_size
        train_loss = train_loss_sum / float(max(1, train_count))

        policy.eval()
        with torch.no_grad():
            val_pred = policy(torch.as_tensor(val_obs, dtype=torch.float32, device=device))
            val_loss = float(
                torch.mean(
                    ((val_pred - torch.as_tensor(val_actions, dtype=torch.float32, device=device)) ** 2)
                    * loss_weights
                ).item()
            )
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        log_rows.append(row)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            policy.save(
                best_path,
                {
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "dataset": os.path.abspath(args.dataset),
                    "seed": int(args.seed),
                    "obs_norm_stats": norm_path,
                },
            )
        if epoch == 1 or epoch % 10 == 0 or epoch == int(args.epochs):
            print("epoch={} train_loss={:.6f} val_loss={:.6f}".format(epoch, train_loss, val_loss))

    policy.save(
        final_path,
        {
            "epoch": int(args.epochs),
            "val_loss": float(log_rows[-1]["val_loss"]),
            "dataset": os.path.abspath(args.dataset),
            "seed": int(args.seed),
            "obs_norm_stats": norm_path,
        },
    )

    best_policy, _checkpoint = __import__("bc_policy").load_bc_policy(best_path, device=device)
    val_metrics = evaluate_policy(best_policy, val_obs, val_actions, device)
    train_metrics = evaluate_policy(best_policy, train_obs, train_actions, device)
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "val_loss"])
        writer.writeheader()
        writer.writerows(log_rows)
    curve_written = write_curve(log_rows, curve_path)
    summary = {
        "dataset": os.path.abspath(args.dataset),
        "output_dir": output_dir,
        "rows": int(len(obs)),
        "train_rows": int(len(train_obs)),
        "val_rows": int(len(val_obs)),
        "removed_nonfinite_rows": int(removed_rows),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "normalize_obs": bool(normalize_obs),
        "dataset_balance": bool(str_to_bool(args.dataset_balance)),
        "loss_weights": {
            "vx": float(args.loss_weight_vx),
            "vy": float(args.loss_weight_vy),
            "vz": float(args.loss_weight_vz),
            "yaw": float(args.loss_weight_yaw),
        },
        "best_epoch": int(best_epoch),
        "train_loss_final": float(log_rows[-1]["train_loss"]),
        "val_loss_final": float(log_rows[-1]["val_loss"]),
        "val_loss_best": float(best_val_loss),
        "train_metrics_best": train_metrics,
        "val_metrics_best": val_metrics,
        "bc_policy_best": best_path,
        "bc_policy_final": final_path,
        "bc_train_log": csv_path,
        "bc_train_curve": curve_path if curve_written else "",
        "obs_norm_stats": norm_path,
        "demo_metadata": metadata,
    }
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print("wrote {}".format(best_path))
    print("wrote {}".format(final_path))
    print("wrote {}".format(summary_path))
    print(
        "best_val_loss={:.6f} vx_mae={:.6f} yaw_mae={:.6f} vx_sign={:.3f} yaw_sign={:.3f}".format(
            summary["val_loss_best"],
            val_metrics["vx_mae"],
            val_metrics["yaw_mae"],
            val_metrics["action_vx_sign_agreement"],
            val_metrics["action_yaw_sign_agreement"],
        )
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print("phase7_train_bc_policy failed: {}".format(exc), file=sys.stderr)
        sys.exit(1)
