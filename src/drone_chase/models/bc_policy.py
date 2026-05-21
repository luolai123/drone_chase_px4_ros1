#!/usr/bin/env python3

import os

import torch
import torch.nn as nn


class BCPolicy(nn.Module):
    def __init__(self, obs_dim=20, action_dim=4, hidden_dim=128, activation="tanh", obs_mean=None, obs_std=None):
        super().__init__()
        if activation == "relu":
            act = nn.ReLU
        elif activation == "tanh":
            act = nn.Tanh
        else:
            raise ValueError("unsupported activation={}".format(activation))

        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.activation = str(activation)
        self.net = nn.Sequential(
            nn.Linear(self.obs_dim, self.hidden_dim),
            act(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            act(),
            nn.Linear(self.hidden_dim, self.action_dim),
            nn.Tanh(),
        )
        mean = torch.zeros(self.obs_dim, dtype=torch.float32) if obs_mean is None else torch.as_tensor(obs_mean, dtype=torch.float32)
        std = torch.ones(self.obs_dim, dtype=torch.float32) if obs_std is None else torch.as_tensor(obs_std, dtype=torch.float32)
        std = torch.clamp(std, min=1.0e-6)
        self.register_buffer("obs_mean", mean.reshape(self.obs_dim))
        self.register_buffer("obs_std", std.reshape(self.obs_dim))

    def normalize_obs(self, obs):
        return (obs - self.obs_mean) / self.obs_std

    def forward(self, obs):
        return self.net(self.normalize_obs(obs.float()))

    def predict_numpy(self, obs_np, device=None):
        if device is None:
            device = next(self.parameters()).device
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
        single = obs.ndim == 1
        if single:
            obs = obs.unsqueeze(0)
        with torch.no_grad():
            action = self(obs).cpu().numpy()
        return action[0] if single else action

    def checkpoint(self, extra=None):
        data = {
            "model_state_dict": self.state_dict(),
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "hidden_dim": self.hidden_dim,
            "activation": self.activation,
            "obs_mean": self.obs_mean.detach().cpu(),
            "obs_std": self.obs_std.detach().cpu(),
        }
        if extra:
            data.update(extra)
        return data

    def save(self, path, extra=None):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(self.checkpoint(extra), path)


def load_bc_policy(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device)
    policy = BCPolicy(
        obs_dim=int(checkpoint.get("obs_dim", 20)),
        action_dim=int(checkpoint.get("action_dim", 4)),
        hidden_dim=int(checkpoint.get("hidden_dim", 128)),
        activation=str(checkpoint.get("activation", "tanh")),
        obs_mean=checkpoint.get("obs_mean", None),
        obs_std=checkpoint.get("obs_std", None),
    )
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.to(device)
    policy.eval()
    return policy, checkpoint
