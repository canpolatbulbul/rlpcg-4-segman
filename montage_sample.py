#!/usr/bin/env python3
"""
montage_sample.py — visualize final grids from the SUBMIT-less PCG env (no masking).

Usage
-----
python montage_sample.py --model runs/ppo_grid_nomask/ppo_grid_nomask_final.zip \
  --n 16 --size 13 --max_steps 48 --out montage.png --deterministic

Optional:
  --save_npy samples.npy     # also save raw integer grids (n, H, W)
"""
import argparse
import math
import os
from typing import Any, Tuple, Sequence

import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO

from grid_pcg_env import GridPCGEnv

# ---- Tile IDs (must match env) ----
EMPTY, WALL, ROBOT, OBJECT, GOAL, MOVABLE = 0, 1, 2, 3, 4, 5


# ---------- helpers ----------
def make_env(size: int, max_steps: int, seed: int):
    def _thunk():
        return GridPCGEnv(size=size, max_steps=max_steps, seed=seed)
    return _thunk


def decode_metrics(info: Any) -> Tuple[int, int, float, int]:
    """
    Extract (L1, L2, wall_ratio, valid) from VecEnv info (handles list/dict).
    Metrics are only populated at episode end by the env.
    """
    d = {}
    if isinstance(info, (list, tuple)) and len(info) and isinstance(info[0], dict):
        d = info[0]
    elif isinstance(info, dict):
        d = info
    L1 = int(d.get("L1", 0))
    L2 = int(d.get("L2", 0))
    w  = float(d.get("wall_ratio", 0.0))
    valid = int(d.get("valid", 0))
    return L1, L2, w, valid


def grid_to_rgb(grid: np.ndarray) -> np.ndarray:
    """Map grid (H, W) tile ids → RGB image in [0,1]."""
    H, W = grid.shape
    img = np.ones((H, W, 3), dtype=np.float32)
    colors = {
        EMPTY:  (0.94, 0.94, 0.94),  # light gray
        WALL:   (0.62, 0.43, 0.34),  # brown
        ROBOT:  (0.20, 0.45, 0.95),  # blue
        OBJECT: (0.97, 0.75, 0.25),  # orange/yellow
        GOAL:   (0.95, 0.35, 0.75),  # magenta
        MOVABLE: (0.40, 0.70, 0.40),  # green (movable obstacles)
    }
    for tid, col in colors.items():
        img[grid == tid] = col
    return img


def unwrap_base_env(vec_env) -> Any:
    """Get the underlying (non-Vec) env for direct attribute access."""
    base = vec_env.envs[0]
    while hasattr(base, "env"):
        base = base.env
    return base


def vec_reset_compat(env: DummyVecEnv):
    """Normalize reset to return just obs (VecEnv usually does)."""
    out = env.reset()
    if isinstance(out, tuple) and len(out) == 2:
        obs, _ = out
        return obs
    return out


def vec_step_compat(env: DummyVecEnv, action: Sequence[int]):
    """
    Call env.step(action) and normalize return to 4 fields:
        (obs, reward, done, info)
    SB3 VecEnvs generally use the 4-tuple Gym style; this keeps things robust.
    """
    out = env.step(action)
    if isinstance(out, tuple):
        if len(out) == 4:
            return out  # (obs, rewards, dones, infos)
        elif len(out) == 5:
            obs, reward, terminated, truncated, info = out
            # merge terminations for vec API
            if isinstance(terminated, (list, np.ndarray)) and isinstance(truncated, (list, np.ndarray)):
                done = np.logical_or(terminated, truncated).tolist()
            else:
                done = bool(terminated or truncated)
            return obs, reward, done, info
    raise RuntimeError("Unexpected VecEnv.step() return format")


def extract_final_grid(info):
    if isinstance(info, (list, tuple)) and len(info) and isinstance(info[0], dict):
        d = info[0]
    elif isinstance(info, dict):
        d = info
    else:
        return None
    return d.get("final_grid", None)


def rollout_one(env: DummyVecEnv, model: PPO, max_steps: int, deterministic: bool):
    """Run one episode to termination (which will happen at max_steps)."""
    obs = vec_reset_compat(env)
    done = [False]
    info = [{}]
    steps = 0

    while steps < max_steps:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, done, info = vec_step_compat(env, action)
        steps += 1
        if isinstance(done, (list, np.ndarray)):
            if done[0]:
                break
        else:
            if done:
                break

    base = unwrap_base_env(env)
    final_grid = extract_final_grid(info)

    if final_grid is None:
        # Fallback in case the info didn’t carry it (or you didn’t patch the env):
        base = unwrap_base_env(env)
        final_grid = getattr(base, "last_final_grid", base.grid).copy()
    return final_grid, decode_metrics(info)


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True, help="Path to PPO .zip")
    ap.add_argument("--n", type=int, default=16, help="Number of samples")
    ap.add_argument("--size", type=int, default=13)
    ap.add_argument("--max_steps", type=int, default=48)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out", type=str, default="montage.png")
    ap.add_argument("--deterministic", action="store_true", help="Greedy actions")
    ap.add_argument("--stochastic", action="store_true", help="Sample actions (overrides --deterministic)")
    ap.add_argument("--save_npy", type=str, default="", help="Optional path to save raw grids as .npy")
    args = ap.parse_args()

    deterministic = args.deterministic and not args.stochastic

    # Single-env vec wrapper for evaluation
    env = DummyVecEnv([make_env(args.size, args.max_steps, args.seed)])

    # Load PPO model with env attached (avoids n_envs mismatch issues)
    model = PPO.load(args.model, env=env, device="auto")

    # Roll out n episodes
    grids, metas = [], []
    for _ in range(args.n):
        g, meta = rollout_one(env, model, args.max_steps, deterministic=deterministic)
        grids.append(g)
        metas.append(meta)

    # Optional: save raw integer grids
    if args.save_npy:
        np.save(args.save_npy, np.stack(grids, axis=0))
        print(f"Saved raw grids to {args.save_npy}")

    # Build montage
    cols = int(math.ceil(math.sqrt(args.n)))
    rows = int(math.ceil(args.n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = np.array(axes).reshape(rows, cols)

    for idx in range(rows * cols):
        ax = axes[idx // cols, idx % cols]
        ax.axis("off")
        if idx >= len(grids):
            continue
        img = grid_to_rgb(grids[idx])
        L1, L2, w, valid = metas[idx]
        title = f"L1={L1} L2={L2} w={w:.2f}"
        if valid:
            title = "✓ " + title
        ax.set_title(title, fontsize=10)
        ax.imshow(img, interpolation="nearest")

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=150)
    print(f"Saved montage to {args.out}")


if __name__ == "__main__":
    main()
