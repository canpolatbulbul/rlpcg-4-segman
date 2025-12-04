#!/usr/bin/env python3
"""
montage_phase2_critical_only.py — Generate montage with ONLY movable-critical grids.

Keeps generating until we have N movable-critical grids.

Usage:
  python montage_phase2_critical_only.py \
    --model runs/phase2_challenge/phase2_final.zip \
    --phase1_model runs/phase1_structure/phase1_final.zip \
    --n 16 --out montage_critical_only.png
"""

import argparse
import math
import os
from typing import Any, Tuple

import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
from phase2_env import Phase2Env, EMPTY, WALL, ROBOT, OBJECT, GOAL, MOVABLE


def make_env(phase1_model_path: str, size: int, max_steps: int, phase1_deterministic: bool, seed: int):
    def _thunk():
        return Phase2Env(
            phase1_model_path=phase1_model_path,
            size=size,
            max_steps=max_steps,
            phase1_deterministic=phase1_deterministic,
            seed=seed
        )
    return _thunk


def decode_metrics(info: Any) -> dict:
    """Extract metrics from info."""
    d = {}
    if isinstance(info, (list, tuple)) and len(info) and isinstance(info[0], dict):
        d = info[0]
    elif isinstance(info, dict):
        d = info
    
    return {
        "n_movable": int(d.get("n_movable", 0)),
        "relaxed_solvable": bool(d.get("relaxed_solvable", False)),
        "strict_solvable": bool(d.get("strict_solvable", False)),
        "movable_critical": bool(d.get("movable_critical", False)),
        "valid": int(d.get("valid", 0)),
        "L1_relaxed": int(d.get("L1_relaxed", 0)),
        "L2_relaxed": int(d.get("L2_relaxed", 0)),
    }


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
    """Get the underlying (non-Vec) env."""
    base = vec_env.envs[0]
    while hasattr(base, "env"):
        base = base.env
    return base


def vec_reset_compat(env: DummyVecEnv):
    """Normalize reset to return just obs."""
    out = env.reset()
    if isinstance(out, tuple) and len(out) == 2:
        obs, _ = out
        return obs
    return out


def vec_step_compat(env: DummyVecEnv, action):
    """Call env.step(action) and normalize return."""
    out = env.step(action)
    if isinstance(out, tuple):
        if len(out) == 4:
            return out
        elif len(out) == 5:
            obs, reward, terminated, truncated, info = out
            if isinstance(terminated, (list, np.ndarray)) and isinstance(truncated, (list, np.ndarray)):
                done = np.logical_or(terminated, truncated).tolist()
            else:
                done = bool(terminated or truncated)
            return obs, reward, done, info
    raise RuntimeError("Unexpected VecEnv.step() return format")


def extract_final_grid(info):
    """Extract final grid from info dict."""
    if isinstance(info, (list, tuple)) and len(info) and isinstance(info[0], dict):
        d = info[0]
    elif isinstance(info, dict):
        d = info
    else:
        return None
    return d.get("final_grid", None)


def rollout_one(env: DummyVecEnv, model: PPO, max_steps: int, deterministic: bool):
    """Run one episode to termination."""
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

    final_grid = extract_final_grid(info)
    if final_grid is None:
        base = unwrap_base_env(env)
        final_grid = base.grid.copy()
    
    return final_grid, decode_metrics(info)


def main():
    ap = argparse.ArgumentParser(description="Generate montage with ONLY movable-critical grids")
    ap.add_argument("--model", type=str, required=True, help="Path to Phase 2 model .zip")
    ap.add_argument("--phase1_model", type=str, required=True, help="Path to Phase 1 model .zip")
    ap.add_argument("--n", type=int, default=16, help="Number of CRITICAL samples to collect")
    ap.add_argument("--size", type=int, default=13)
    ap.add_argument("--max_steps", type=int, default=80)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out", type=str, default="montage_critical_only.png")
    ap.add_argument("--deterministic", action="store_true", help="Greedy actions")
    ap.add_argument("--stochastic", action="store_true", help="Sample actions (overrides --deterministic)")
    ap.add_argument("--phase1_deterministic", action="store_true",
                    help="Use Phase 1 model deterministically for base puzzles")
    ap.add_argument("--save_npy", type=str, default="", help="Optional path to save raw grids as .npy")
    ap.add_argument("--max_attempts", type=int, default=1000, 
                    help="Maximum episodes to try before giving up (default: 1000)")
    
    args = ap.parse_args()
    deterministic = args.deterministic and not args.stochastic

    # Single-env vec wrapper
    env = DummyVecEnv([make_env(
        args.phase1_model,
        args.size,
        args.max_steps,
        args.phase1_deterministic,
        args.seed
    )])

    # Load model
    model = PPO.load(args.model, env=env, device="auto")

    # Roll out episodes until we have N movable-critical grids
    critical_grids = []
    critical_metas = []
    attempts = 0
    
    print(f"Generating {args.n} movable-critical grids...")
    print(f"(Will try up to {args.max_attempts} episodes)")
    
    while len(critical_grids) < args.n and attempts < args.max_attempts:
        g, meta = rollout_one(env, model, args.max_steps, deterministic=deterministic)
        attempts += 1
        
        # Only keep if movable-critical
        if meta.get("movable_critical", False):
            critical_grids.append(g)
            critical_metas.append(meta)
            print(f"Found {len(critical_grids)}/{args.n} critical grids (tried {attempts} episodes)")
    
    if len(critical_grids) < args.n:
        print(f"\nWarning: Only found {len(critical_grids)} critical grids after {attempts} attempts.")
        print(f"Critical rate: {100.0 * len(critical_grids) / attempts:.1f}%")
    else:
        print(f"\nSuccess! Found {args.n} critical grids in {attempts} attempts.")
        print(f"Critical rate: {100.0 * len(critical_grids) / attempts:.1f}%")

    if len(critical_grids) == 0:
        print("No critical grids found. Cannot generate montage.")
        return

    # Optional: save raw grids
    if args.save_npy:
        np.save(args.save_npy, np.stack(critical_grids, axis=0))
        print(f"Saved raw grids to {args.save_npy}")

    # Build montage (only with critical grids we found)
    n_grids = len(critical_grids)
    cols = int(math.ceil(math.sqrt(n_grids)))
    rows = int(math.ceil(n_grids / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = np.array(axes).reshape(rows, cols)

    for idx in range(rows * cols):
        ax = axes[idx // cols, idx % cols]
        ax.axis("off")
        if idx >= len(critical_grids):
            continue
        
        img = grid_to_rgb(critical_grids[idx])
        meta = critical_metas[idx]
        
        # Build title
        n_mov = meta["n_movable"]
        L1 = meta["L1_relaxed"]
        L2 = meta["L2_relaxed"]
        title = f"✓ CRIT  m={n_mov}  L1={L1} L2={L2}"
        
        ax.set_title(title, fontsize=10, color='green', weight='bold')
        ax.imshow(img, interpolation="nearest")

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=150)
    print(f"\nSaved montage to {args.out}")
    
    # Summary statistics
    movable_counts = [meta["n_movable"] for meta in critical_metas]
    print(f"\n=== Critical Grid Statistics ===")
    print(f"Movable count: avg={np.mean(movable_counts):.1f}, min={np.min(movable_counts)}, max={np.max(movable_counts)}")


if __name__ == "__main__":
    main()

