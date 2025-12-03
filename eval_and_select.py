# eval_and_select.py
# Evaluate a trained PPO on GridPCGEnv, save results, and keep "good" layouts.
# Usage example is at the bottom of this file.

import os
import csv
import shutil
import argparse
from pathlib import Path
from typing import Dict, Any

import numpy as np
import matplotlib.pyplot as plt

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

from grid_pcg_env import GridPCGEnv, EMPTY, WALL, ROBOT, OBJECT, GOAL, MOVABLE

# ---------- plotting ----------

COLORS = {
    EMPTY: (0.94, 0.94, 0.94),   # light gray
    WALL:  (0.53, 0.35, 0.24),   # brown
    ROBOT: (0.22, 0.49, 0.99),   # blue
    OBJECT:(0.98, 0.36, 0.85),   # magenta/pink
    GOAL:  (0.98, 0.70, 0.19),   # gold
    MOVABLE: (0.40, 0.70, 0.40),  # green (movable obstacles)
}

def save_grid_png(grid: np.ndarray, out_path: Path, title: str = ""):
    h, w = grid.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    for t, col in COLORS.items():
        rgb[grid == t] = col
    plt.figure(figsize=(3, 3))
    plt.imshow(rgb, interpolation='nearest')
    plt.axis('off')
    if title:
        plt.title(title, fontsize=9)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(pad=0.1)
    plt.savefig(out_path, dpi=180)
    plt.close()


# ---------- eval loop ----------

def make_env(size: int, max_steps: int, use_curriculum: bool, seed: int):
    def _thunk():
        env = GridPCGEnv(size=size, max_steps=max_steps, seed=seed)
        env.use_curriculum = use_curriculum
        return Monitor(env)
    return _thunk

def run_episodes(
        model_path: str,
        episodes: int,
        out_dir: str,
        size: int,
        max_steps: int,
        use_curriculum: bool,
        deterministic: bool,
        seed: int,
        thresholds: Dict[str, float],
        n_envs: int = 1,
):
    out = Path(out_dir)
    eps_dir = out / "episodes"
    keep_dir = out / "keepers"
    out.mkdir(parents=True, exist_ok=True)
    eps_dir.mkdir(parents=True, exist_ok=True)
    keep_dir.mkdir(parents=True, exist_ok=True)

    # Parallel Envs
    # We use SubprocVecEnv for parallelism if n_envs > 1
    env_fns = [make_env(size, max_steps, use_curriculum, seed + i) for i in range(n_envs)]
    if n_envs > 1:
        vec_env = SubprocVecEnv(env_fns)
    else:
        vec_env = DummyVecEnv(env_fns)

    model = PPO.load(model_path, env=vec_env, device="auto")

    # CSV
    csv_path = out / "results.csv"
    fieldnames = [
        "episode", "reward", "valid",
        "L1", "L2", "Lsum",
        "wall_ratio", "adj_per_wall", "iso_frac",
        "n_movable", "n_movable_on_path",
        "relaxed_solvable", "strict_solvable", "movable_critical",
        "L1_relaxed", "L2_relaxed", "L1_strict", "L2_strict",
        "png_path", "npy_path", "kept"
    ]
    f = open(csv_path, "w", newline="")
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    print(f"Running {episodes} episodes with {n_envs} parallel environments...")
    
    obs = vec_env.reset()
    ep_counts = 0
    
    # We need to track cumulative reward per env if we want to log it, 
    # but PPO `dones` usually reset it. 
    # For simplicity in eval, we might just rely on the info dict if the env provided it,
    # or just ignore 'reward' column since we care about structural metrics.
    # However, let's try to track it.
    current_ep_rewards = np.zeros(n_envs, dtype=np.float32)

    while ep_counts < episodes:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, rewards, dones, infos = vec_env.step(action)
        
        current_ep_rewards += rewards

        for i in range(n_envs):
            if dones[i]:
                # Episode finished
                if ep_counts >= episodes:
                    break
                
                ep_counts += 1
                final_info = infos[i]
                ep_rew = current_ep_rewards[i]
                current_ep_rewards[i] = 0.0 # reset tracker

                # Fallbacks
                final_info.setdefault("valid", 0)
                final_info.setdefault("L1", 0)
                final_info.setdefault("L2", 0)
                final_info.setdefault("wall_ratio", 0.0)
                final_info.setdefault("adj_per_wall", 0.0)
                final_info.setdefault("iso_frac", 0.0)
                final_info.setdefault("n_movable", 0)
                final_info.setdefault("n_movable_on_path", 0)
                final_info.setdefault("relaxed_solvable", False)
                final_info.setdefault("strict_solvable", False)
                final_info.setdefault("movable_critical", False)
                final_info.setdefault("L1_relaxed", 0)
                final_info.setdefault("L2_relaxed", 0)
                final_info.setdefault("L1_strict", None)
                final_info.setdefault("L2_strict", None)
                
                # Get grid from info (crucial for SubprocVecEnv)
                last_grid = final_info.get("final_grid", None)
                if last_grid is None:
                    # Should not happen if env is updated, but fallback for safety
                    # This fallback ONLY works if n_envs=1 and DummyVecEnv
                    if n_envs == 1 and isinstance(vec_env, DummyVecEnv):
                         last_grid = vec_env.envs[0].env.env.grid.copy()
                    else:
                        # Cannot recover grid from subprocess without info
                        print(f"Warning: 'final_grid' not found in info for ep {ep_counts}. Skipping save.")
                        continue

                ep_id = f"{ep_counts:05d}"

                # Apply selection thresholds
                # Optionally require movable-critical condition
                keep = (
                        (final_info["valid"] == 1) and
                        (thresholds["w_min"] <= final_info["wall_ratio"] <= thresholds["w_max"]) and
                        (thresholds["movable_min"] <= final_info["n_movable"] <= thresholds["movable_max"]) and
                        (final_info["n_movable_on_path"] >= thresholds["min_movable_on_path"]) and
                        (final_info["adj_per_wall"] >= thresholds["adj_min"]) and
                        (final_info["iso_frac"] <= thresholds["iso_max"]) and
                        ((final_info["L1"] + final_info["L2"]) >= thresholds["min_Lsum"]) and
                        (final_info["L1"] >= thresholds["min_L1"]) and
                        (final_info["L2"] >= thresholds["min_L2"])
                )
                # Optionally add: and final_info.get("movable_critical", False)

                png_path = ""
                npy_path = ""

                if keep:
                    # Save directly to keep_dir
                    png_path = keep_dir / f"ep_{ep_id}.png"
                    npy_path = keep_dir / f"ep_{ep_id}.npy"
                    title = f"L1={final_info['L1']} L2={final_info['L2']} w={final_info['wall_ratio']:.02f} m={final_info['n_movable']}"
                    save_grid_png(last_grid, png_path, title=title)
                    np.save(npy_path, last_grid)

                # Write CSV row
                writer.writerow({
                    "episode": ep_counts,
                    "reward": ep_rew,
                    "valid": final_info["valid"],
                    "L1": final_info["L1"],
                    "L2": final_info["L2"],
                    "Lsum": final_info["L1"] + final_info["L2"],
                    "wall_ratio": final_info["wall_ratio"],
                    "adj_per_wall": final_info["adj_per_wall"],
                    "iso_frac": final_info["iso_frac"],
                    "n_movable": final_info["n_movable"],
                    "n_movable_on_path": final_info["n_movable_on_path"],
                    "relaxed_solvable": int(final_info.get("relaxed_solvable", False)),
                    "strict_solvable": int(final_info.get("strict_solvable", False)),
                    "movable_critical": int(final_info.get("movable_critical", False)),
                    "L1_relaxed": final_info.get("L1_relaxed", 0),
                    "L2_relaxed": final_info.get("L2_relaxed", 0),
                    "L1_strict": final_info.get("L1_strict", ""),
                    "L2_strict": final_info.get("L2_strict", ""),
                    "png_path": str(png_path),
                    "npy_path": str(npy_path),
                    "kept": int(keep),
                })
                
                if ep_counts % 100 == 0:
                    print(f"Processed {ep_counts}/{episodes} episodes...")

    f.close()
    vec_env.close()
    
    # Read CSV and print summary statistics
    if HAS_PANDAS:
        try:
        df = pd.read_csv(csv_path)
        valid_count = df["valid"].sum()
        relaxed_count = df["relaxed_solvable"].sum() if "relaxed_solvable" in df.columns else 0
        strict_count = df["strict_solvable"].sum() if "strict_solvable" in df.columns else 0
        critical_count = df["movable_critical"].sum() if "movable_critical" in df.columns else 0
        kept_count = df["kept"].sum()
        
        print(f"\n=== Summary Statistics ===")
        print(f"Total episodes: {episodes}")
        print(f"Valid levels: {valid_count} ({100.0*valid_count/episodes:.1f}%)")
        if "relaxed_solvable" in df.columns:
            print(f"Relaxed solvable: {relaxed_count} ({100.0*relaxed_count/episodes:.1f}%)")
        if "strict_solvable" in df.columns:
            print(f"Strict solvable: {strict_count} ({100.0*strict_count/episodes:.1f}%)")
        if "movable_critical" in df.columns:
            critical_pct = 100.0 * critical_count / max(1, valid_count)
            print(f"Movable-critical: {critical_count}/{valid_count} ({critical_pct:.1f}% of valid levels)")
        print(f"Keepers (passed thresholds): {kept_count} ({100.0*kept_count/episodes:.1f}%)")
        except Exception as e:
            print(f"Could not generate summary statistics: {e}")
    else:
        print("(Install pandas for summary statistics)")
    
    print(f"\nDone. Saved CSV to: {csv_path}")
    print(f"Keepers saved in: {keep_dir}")


# ---------- CLI ----------

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate PPO on GridPCGEnv and select keepers.")
    p.add_argument("--model", required=True, help="Path to PPO .zip")
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--out_dir", default="eval_out")

    p.add_argument("--size", type=int, default=13)
    p.add_argument("--max_steps", type=int, default=192)
    p.add_argument("--use_curriculum", action="store_true")
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_envs", type=int, default=16, help="Number of parallel environments")

    # Selection thresholds
    p.add_argument("--w_min", type=float, default=0.20)
    p.add_argument("--w_max", type=float, default=0.30)
    p.add_argument("--movable_min", type=int, default=3)
    p.add_argument("--movable_max", type=int, default=7)
    p.add_argument("--min_movable_on_path", type=int, default=2)
    p.add_argument("--adj_min", type=float, default=0.15)
    p.add_argument("--iso_max", type=float, default=0.30)
    p.add_argument("--min_Lsum", type=int, default=14)
    p.add_argument("--min_L1", type=int, default=4)
    p.add_argument("--min_L2", type=int, default=4)

    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    thresholds = dict(
        w_min=args.w_min, w_max=args.w_max,
        movable_min=args.movable_min, movable_max=args.movable_max,
        min_movable_on_path=args.min_movable_on_path,
        adj_min=args.adj_min, iso_max=args.iso_max,
        min_Lsum=args.min_Lsum,
        min_L1=args.min_L1, min_L2=args.min_L2,
    )
    run_episodes(
        model_path=args.model,
        episodes=args.episodes,
        out_dir=args.out_dir,
        size=args.size,
        max_steps=args.max_steps,
        use_curriculum=args.use_curriculum,
        deterministic=args.deterministic,
        seed=args.seed,
        thresholds=thresholds,
        n_envs=args.n_envs,
    )
