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

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
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
):
    out = Path(out_dir)
    eps_dir = out / "episodes"
    keep_dir = out / "keepers"
    out.mkdir(parents=True, exist_ok=True)
    eps_dir.mkdir(parents=True, exist_ok=True)
    keep_dir.mkdir(parents=True, exist_ok=True)

    # Env (single env for simple per-episode accounting)
    vec_env = DummyVecEnv([make_env(size, max_steps, use_curriculum, seed)])
    model = PPO.load(model_path, env=vec_env, device="auto")

    # CSV
    csv_path = out / "results.csv"
    fieldnames = [
        "episode", "reward", "valid",
        "L1", "L2", "Lsum",
        "wall_ratio", "adj_per_wall", "iso_frac",
        "n_movable",
        "png_path", "npy_path", "kept"
    ]
    f = open(csv_path, "w", newline="")
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    keep_count = 0
    obs = vec_env.reset()
    for ep in range(episodes):
        done = False
        ep_rew = 0.0
        final_info: Dict[str, Any] = {}
        last_grid = None

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, dones, infos = vec_env.step(action)
            ep_rew += float(reward[0])
            done = bool(dones[0])

            # If env exposes the live grid in info at end, we’ll read the final one below.
            if done:
                # Some envs stuff metrics in infos[0] at termination.
                final_info = infos[0] if isinstance(infos, (list, tuple)) else infos
                # Fallbacks for missing keys
                final_info.setdefault("valid", 0)
                final_info.setdefault("L1", 0)
                final_info.setdefault("L2", 0)
                final_info.setdefault("wall_ratio", 0.0)
                final_info.setdefault("adj_per_wall", 0.0)
                final_info.setdefault("iso_frac", 0.0)
                final_info.setdefault("n_movable", 0)
                # final_grid was added in your env on termination:
                last_grid = final_info.get("final_grid", None)

        ep_id = f"{ep:05d}"
        if last_grid is None:
            # Try to pull it directly from underlying env (rarely needed)
            last_grid = vec_env.envs[0].env.env.grid.copy()

        png_path = eps_dir / f"ep_{ep_id}.png"
        npy_path = eps_dir / f"ep_{ep_id}.npy"
        title = f"L1={final_info['L1']} L2={final_info['L2']} w={final_info['wall_ratio']:.02f} m={final_info['n_movable']}"
        save_grid_png(last_grid, png_path, title=title)
        np.save(npy_path, last_grid)

        # Apply selection thresholds
        keep = (
                (final_info["valid"] == 1) and
                (thresholds["w_min"] <= final_info["wall_ratio"] <= thresholds["w_max"]) and
                (thresholds["movable_min"] <= final_info["n_movable"] <= thresholds["movable_max"]) and
                (final_info["adj_per_wall"] >= thresholds["adj_min"]) and
                (final_info["iso_frac"] <= thresholds["iso_max"]) and
                ((final_info["L1"] + final_info["L2"]) >= thresholds["min_Lsum"])
        )
        if keep:
            keep_count += 1
            shutil.copy2(png_path, keep_dir / png_path.name)
            shutil.copy2(npy_path, keep_dir / npy_path.name)

        # Write CSV row
        writer.writerow({
            "episode": ep,
            "reward": ep_rew,
            "valid": final_info["valid"],
            "L1": final_info["L1"],
            "L2": final_info["L2"],
            "Lsum": final_info["L1"] + final_info["L2"],
            "wall_ratio": final_info["wall_ratio"],
            "adj_per_wall": final_info["adj_per_wall"],
            "iso_frac": final_info["iso_frac"],
            "n_movable": final_info["n_movable"],
            "png_path": str(png_path),
            "npy_path": str(npy_path),
            "kept": int(keep),
        })

        # reset for next ep
        obs = vec_env.reset()

    f.close()
    print(f"\nDone. Saved CSV to: {csv_path}")
    print(f"Keepers: {keep_count} / {episodes}  (kept in: {keep_dir})")


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

    # Selection thresholds
    p.add_argument("--w_min", type=float, default=0.20)
    p.add_argument("--w_max", type=float, default=0.30)
    p.add_argument("--movable_min", type=int, default=3)
    p.add_argument("--movable_max", type=int, default=7)
    p.add_argument("--adj_min", type=float, default=0.15)
    p.add_argument("--iso_max", type=float, default=0.30)
    p.add_argument("--min_Lsum", type=int, default=14)

    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    thresholds = dict(
        w_min=args.w_min, w_max=args.w_max,
        movable_min=args.movable_min, movable_max=args.movable_max,
        adj_min=args.adj_min, iso_max=args.iso_max,
        min_Lsum=args.min_Lsum,
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
    )
