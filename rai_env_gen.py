#!/usr/bin/env python3
"""
rai_env_gen.py — Generate PCG environments and save as rai Config files

Usage
-----
python3 rai_env_gen.py --n 16 --size 13 --max_steps 1024 --stochastic
"""
import argparse
import math
import os
import random
from typing import Any, Tuple, Sequence

import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO

from grid_pcg_env import GridPCGEnv

import robotic as ry

# ---- Tile IDs (must match env) ----
EMPTY, WALL, ROBOT, OBJECT, GOAL = 0, 1, 2, 3, 4


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
    ap.add_argument("--n", type=int, default=16, help="Number of samples")
    ap.add_argument("--size", type=int, default=13)
    ap.add_argument("--max_steps", type=int, default=48)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--deterministic", action="store_true", help="Greedy actions")
    ap.add_argument("--stochastic", action="store_true", help="Sample actions (overrides --deterministic)")
    args = ap.parse_args()

    deterministic = args.deterministic and not args.stochastic

    # Single-env vec wrapper for evaluation
    env = DummyVecEnv([make_env(args.size, args.max_steps, args.seed)])

    # Load PPO model with env attached (avoids n_envs mismatch issues)
    model = "runs/ppo_grid_corridors_v3/ppo_grid_nomask_final"
    model = PPO.load(model, env=env, device="auto")

    fn = random.randint(0, 100)
    os.makedirs(f"ry_config/case_run_id_{fn}", exist_ok=True)

    # Roll out n episodes
    for i in range(args.n):
        print(f"Rolling out sample {i+1}/{args.n}...")
        g, meta = rollout_one(env, model, args.max_steps, deterministic=deterministic)
        valid = meta[3]

        env_size = 4.0
        wall_thickness = 0.1
        ob_s = (env_size-wall_thickness)/float(args.size)
        sx = -(env_size/2) + ob_s / 2 + .05
        sy = (env_size/2) - ob_s / 2 - .05

        if valid:
            C = ry.Config()
            C_aux = ry.Config()
            C.addFile('ry_config/base.g')
            C_aux.addFile('ry_config/base-aux.g')
            for r in range(g.shape[0]):
                for c in range(g.shape[1]):

                    if g[r, c] == WALL:
                        f = C.addFrame(f"block_{r}_{c}", "world", f"shape:ssBox, size:[{ob_s}, {ob_s}, 0.2, 0.01], color:[0.6953, 0.515625, 0.453125], contact:1")
                        f.setRelativePosition([sx+ob_s*c, sy-ob_s*r, 0.1])

                        f_aux = C_aux.addFrame(f"block_{r}_{c}", "world", f"shape:ssBox, size:[{ob_s}, {ob_s}, 0.2, 0.01], color:[1 0 0], contact:1")
                        f_aux.setRelativePosition([sx+ob_s*c, sy-ob_s*r, 0.1])

                    elif g[r, c] == ROBOT:
                        f = C.frame("ego").setRelativePosition([sx+ob_s*c, sy-ob_s*r, 0.0])
                        f.setShape(ry.ST.ssCylinder, size=[.2, ob_s*.45, .02])

                        f_aux = C_aux.frame("ego").setRelativePosition([sx+ob_s*c, sy-ob_s*r, 0.0])

                    elif g[r, c] == OBJECT:

                        f = C.addFrame("obj1Joint", "world")
                        f.setRelativePosition([sx+ob_s*c, sy-ob_s*r, 0.1])
                        C.addFrame(f"obj1", "obj1Joint", f"shape:ssBox, size:[{ob_s*.6}, {ob_s*.6}, 0.2, 0.01], color:[0 0 1], contact:1, joint:rigid, logical:{'{movable_go}'}")

                        f_aux = C_aux.addFrame("obj1Joint", "world")
                        f_aux.setRelativePosition([sx+ob_s*c, sy-ob_s*r, 0.1])
                        C_aux.addFrame(f"obj1", "obj1Joint", f"shape:ssBox, size:[{ob_s*.6}, {ob_s*.6}, 0.2, 0.01], color:[0 0 1], contact:1, logical:{'{movable_go}'}")
                        C_aux.addFrame("obj1_cam", "obj1", f"Q:'t(0 0 7) d(180 1 0 0)' shape:camera, width:300, height:300")

                    elif g[r, c] == GOAL:
                        f = C.addFrame(f"goal1", "world", f"shape:ssBox, size:[{ob_s*.6}, {ob_s*.6}, 0.2, 0.01], color:[0 0 1 .3], contact:0, logical{'{goal}'}")
                        f.setRelativePosition([sx+ob_s*c, sy-ob_s*r, 0.1])

                        f_aux = C_aux.addFrame(f"goal1", "world", f"shape:ssBox, size:[{ob_s*.6}, {ob_s*.6}, 0.2, 0.01], color:[0 0 1], contact:0, logical{'{goal}'}")
                        f_aux.setRelativePosition([sx+ob_s*c, sy-ob_s*r, 0.1])

            #C.view(True)
            #C_aux.view(True)

            #Save the config str as a .g file in the ry_config folder, in a unique subfolder
            new_C = C.write()
            new_C_aux = C_aux.write()
            os.makedirs(f"ry_config/case_run_id_{fn}/pcg-{i}", exist_ok=True)
            with open(f"ry_config/case_run_id_{fn}/pcg-{i}/pcg-{i}.g", "w") as f:
                f.write(new_C)
            with open(f"ry_config/case_run_id_{fn}/pcg-{i}/pcg-{i}-aux.g", "w") as f:
                f.write(new_C_aux)

            

if __name__ == "__main__":
    main()
