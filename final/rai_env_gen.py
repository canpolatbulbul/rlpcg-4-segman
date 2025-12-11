#!/usr/bin/env python3
"""
rai_env_gen.py — Generate PCG environments using 2-phase training and save as rai Config files

Usage
-----
# Generate grids using Phase 1 + Phase 2 models
python3 rai_env_gen.py \
  --phase1_model <path_to_phase1_model> \
  --phase2_model <path_to_phase2_model> \
  --n 16 \
  --size 13 \
  --stochastic
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

from phase1_env import Phase1Env
from phase2_env import Phase2Env
from grid_pcg_env import EMPTY, WALL, ROBOT, OBJECT, GOAL, MOVABLE

import robotic as ry

# ---------- helpers ----------
def make_phase1_env(size: int, max_steps: int, seed: int, wall_target: float, n_objects: int = 1):
    """
    Create Phase 1 environment factory.
    
    Note: wall_target should match the value used during Phase 1 training.
    The environment uses wall_target for:
    - Early termination logic (allows termination when wall ratio is within ±0.05 of target)
    - Reward calculation (penalty for deviation from target)
    - Early termination bonus scaling
    
    Using a different wall_target during generation can affect when episodes terminate,
    even though the model's policy is already trained.
    """
    def _thunk():
        return Phase1Env(size=size, max_steps=max_steps, seed=seed, wall_target=wall_target, n_objects=n_objects)
    return _thunk


def make_phase2_env(phase1_model_path: str, size: int, max_steps: int, seed: int, phase1_deterministic: bool, n_objects: int = 1):
    """Create Phase 2 environment factory."""
    def _thunk():
        return Phase2Env(
            phase1_model_path=phase1_model_path,
            size=size,
            max_steps=max_steps,
            seed=seed,
            phase1_deterministic=phase1_deterministic,
            n_objects=n_objects
        )
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


def decode_phase2_metrics(info: Any) -> dict:
    """
    Extract Phase 2 metrics including movable_critical from VecEnv info.
    """
    d = {}
    if isinstance(info, (list, tuple)) and len(info) and isinstance(info[0], dict):
        d = info[0]
    elif isinstance(info, dict):
        d = info
    return {
        "valid": int(d.get("valid", 0)),
        "movable_critical": bool(d.get("movable_critical", False)),
        "n_movable": int(d.get("n_movable", 0)),
        "relaxed_solvable": bool(d.get("relaxed_solvable", False)),
        "strict_solvable": bool(d.get("strict_solvable", False)),
    }


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
    """Extract final grid from info dict."""
    if isinstance(info, (list, tuple)) and len(info) and isinstance(info[0], dict):
        d = info[0]
    elif isinstance(info, dict):
        d = info
    else:
        return None
    return d.get("final_grid", None)


def rollout_phase1(env: DummyVecEnv, model: PPO, max_steps: int, deterministic: bool):
    """Run Phase 1 episode to generate base puzzle."""
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
        # Fallback: get grid from env
        final_grid = base.grid.copy()
    
    return final_grid, decode_metrics(info)


def rollout_phase2(env: DummyVecEnv, model: PPO, max_steps: int, deterministic: bool):
    """Run Phase 2 episode to add movables to base puzzle."""
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
    phase2_metrics = decode_phase2_metrics(info)
    
    if final_grid is None:
        # Fallback: get grid from env
        final_grid = base.grid.copy()
    
    # Return grid and metrics tuple (for compatibility) + full metrics dict
    L1, L2, w, valid = decode_metrics(info)
    return final_grid, (L1, L2, w, valid, phase2_metrics)


def grid_to_rai_config(grid: np.ndarray, case_id: int, sample_id: int, size: int, n_objects: int = 1):
    """
    Convert numpy grid to rai Config files (.g format).
    
    Args:
        grid: Final grid with tiles (EMPTY, WALL, ROBOT, OBJECT, GOAL, MOVABLE)
        case_id: Case ID for folder naming
        sample_id: Sample ID for file naming
        size: Grid size (assumed square)
        n_objects: Number of objects (and goals) in the grid
    
    Returns:
        (C, C_aux): Tuple of rai Config objects
    """
    env_size = 4.0
    wall_thickness = 0.1
    ob_s = (env_size - wall_thickness) / float(size)
    sx = -(env_size / 2) + ob_s / 2 + 0.05
    sy = (env_size / 2) - ob_s / 2 - 0.05

    C = ry.Config()
    C_aux = ry.Config()
    C.addFile('ry_config/base.g')
    C_aux.addFile('ry_config/base-aux.g')

    # Collect object and goal positions first
    object_positions = []
    goal_positions = []
    robot_pos = None
    
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            tile = int(grid[r, c])
            if tile == ROBOT:
                robot_pos = (r, c)
            elif tile == OBJECT:
                object_positions.append((r, c))
            elif tile == GOAL:
                goal_positions.append((r, c))
    
    # Pair objects with goals (greedy: closest pairing)
    def manhattan_dist(pos1, pos2):
        r1, c1 = pos1
        r2, c2 = pos2
        return abs(r1 - r2) + abs(c1 - c2)
    
    used_goals = set()
    object_goal_pairs = []
    for i, (or_, oc) in enumerate(object_positions):
        best_goal_idx = None
        best_dist = float('inf')
        for j, (gr, gc) in enumerate(goal_positions):
            if j in used_goals:
                continue
            dist = manhattan_dist((or_, oc), (gr, gc))
            if dist < best_dist:
                best_dist = dist
                best_goal_idx = j
        if best_goal_idx is not None:
            object_goal_pairs.append((i, best_goal_idx))
            used_goals.add(best_goal_idx)
    
    # Now process all tiles
    movable_count = 0  # Track movable objects for naming
    object_indices = {}  # Map (r, c) -> object index
    goal_indices = {}   # Map (r, c) -> goal index
    
    # Build index maps
    for i, (r, c) in enumerate(object_positions):
        object_indices[(r, c)] = i + 1  # 1-indexed
    for i, (r, c) in enumerate(goal_positions):
        goal_indices[(r, c)] = i + 1  # 1-indexed

    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            tile = int(grid[r, c])
            x_pos = sx + ob_s * c
            y_pos = sy - ob_s * r

            if tile == WALL:
                # Add wall block
                f = C.addFrame(f"block_{r}_{c}", "world", 
                             f"shape:ssBox, size:[{ob_s}, {ob_s}, 0.2, 0.01], color:[0.6953, 0.515625, 0.453125], contact:1")
                f.setRelativePosition([x_pos, y_pos, 0.1])

                f_aux = C_aux.addFrame(f"block_{r}_{c}", "world", 
                                     f"shape:ssBox, size:[{ob_s}, {ob_s}, 0.2, 0.01], color:[1 0 0], contact:1")
                f_aux.setRelativePosition([x_pos, y_pos, 0.1])

            elif tile == ROBOT:
                # Set robot position
                f = C.frame("ego").setRelativePosition([x_pos, y_pos, 0.0])
                f.setShape(ry.ST.ssCylinder, size=[.2, ob_s * .45, .02])
                
                # Update cam0 camera for MOSeGMan
                # Main config: attached to ego (for robot's perspective)
                try:
                    C.delFrame("cam0")
                except:
                    pass
                C.addFrame("cam0", "ego", f"Q:'t(0 0 0.5) d(180 1 0 0)' shape:camera, width:300, height:300")

                f_aux = C_aux.frame("ego").setRelativePosition([x_pos, y_pos, 0.0])
                
                # Aux config: Use fixed top-down camera attached to world (not ego) so it always sees entire scene
                # This ensures mask_object can always see all objects regardless of robot position
                try:
                    C_aux.delFrame("cam0")
                except:
                    pass
                # Top-down camera at height 10, looking down, attached to world (fixed position)
                C_aux.addFrame("cam0", "world", f"Q:'t(0 0 10) d(180 1 0 0)' shape:camera, width:300, height:300")

            elif tile == OBJECT:
                # Add goal object (movable_go) - use object index
                obj_idx = object_indices.get((r, c), 1)
                obj_name = f"obj{obj_idx}"
                joint_name = f"{obj_name}Joint"
                
                # Color palette for objects (up to 5 different colors)
                # Colors: Blue, Green, Red, Yellow, Magenta
                object_colors = [
                    [0, 0, 1],      # Blue (object 1)
                    [0, 1, 0],      # Green (object 2)
                    [1, 0, 0],      # Red (object 3)
                    [1, 1, 0],      # Yellow (object 4)
                    [1, 0, 1],      # Magenta (object 5)
                ]
                color_idx = min(obj_idx - 1, len(object_colors) - 1)  # 0-indexed, wrap at 5
                obj_color = object_colors[color_idx]
                color_str = f"[{obj_color[0]} {obj_color[1]} {obj_color[2]}]"
                
                f = C.addFrame(joint_name, "world")
                f.setRelativePosition([x_pos, y_pos, 0.1])
                C.addFrame(obj_name, joint_name, 
                          f"shape:ssBox, size:[{ob_s*.6}, {ob_s*.6}, 0.2, 0.01], color:{color_str}, contact:1, joint:rigid, logical:{{movable_go}}")

                f_aux = C_aux.addFrame(joint_name, "world")
                f_aux.setRelativePosition([x_pos, y_pos, 0.1])
                C_aux.addFrame(obj_name, joint_name, 
                              f"shape:ssBox, size:[{ob_s*.6}, {ob_s*.6}, 0.2, 0.01], color:{color_str}, contact:1, logical:{{movable_go}}")
                C_aux.addFrame(f"{obj_name}_cam", obj_name, f"Q:'t(0 0 7) d(180 1 0 0)' shape:camera, width:300, height:300")

            elif tile == GOAL:
                # Add goal position - use goal index
                goal_idx = goal_indices.get((r, c), 1)
                goal_name = f"goal{goal_idx}"
                
                # Color palette for goals (up to 5 different colors, same as objects)
                # Colors: Blue, Green, Red, Yellow, Magenta
                goal_colors = [
                    [0, 0, 1],      # Blue (goal 1)
                    [0, 1, 0],      # Green (goal 2)
                    [1, 0, 0],      # Red (goal 3)
                    [1, 1, 0],      # Yellow (goal 4)
                    [1, 0, 1],      # Magenta (goal 5)
                ]
                color_idx = min(goal_idx - 1, len(goal_colors) - 1)  # 0-indexed, wrap at 5
                goal_color = goal_colors[color_idx]
                color_str = f"[{goal_color[0]} {goal_color[1]} {goal_color[2]}]"
                color_str_transparent = f"[{goal_color[0]} {goal_color[1]} {goal_color[2]} .3]"  # Transparent for main config
                
                f = C.addFrame(goal_name, "world", 
                             f"shape:ssBox, size:[{ob_s*.6}, {ob_s*.6}, 0.2, 0.01], color:{color_str_transparent}, contact:0, logical:{{goal}}")
                f.setRelativePosition([x_pos, y_pos, 0.1])

                f_aux = C_aux.addFrame(goal_name, "world", 
                                     f"shape:ssBox, size:[{ob_s*.6}, {ob_s*.6}, 0.2, 0.01], color:{color_str}, contact:0, logical:{{goal}}")
                f_aux.setRelativePosition([x_pos, y_pos, 0.1])

            elif tile == MOVABLE:
                # Add movable obstacle (movable_o)
                # Name starts with "ob" so MOSeGMan's find_critical_objects can detect it
                movable_count += 1
                obj_name = f"ob_movable_{movable_count}"
                
                f = C.addFrame(f"{obj_name}Joint", "world")
                f.setRelativePosition([x_pos, y_pos, 0.1])
                C.addFrame(obj_name, f"{obj_name}Joint", 
                          f"shape:ssBox, size:[{ob_s*.6}, {ob_s*.6}, 0.2, 0.01], color:[1 1 0], contact:1, joint:rigid, logical:{{movable_o}}")

                f_aux = C_aux.addFrame(f"{obj_name}Joint", "world")
                f_aux.setRelativePosition([x_pos, y_pos, 0.1])
                C_aux.addFrame(obj_name, f"{obj_name}Joint", 
                              f"shape:ssBox, size:[{ob_s*.6}, {ob_s*.6}, 0.2, 0.01], color:[1 1 0], contact:1, logical:{{movable_o}}")
                # Add camera for MOSeGMan's mask_object function (needed for object weight calculation)
                C_aux.addFrame(f"{obj_name}_cam", obj_name, f"Q:'t(0 0 7) d(180 1 0 0)' shape:camera, width:300, height:300")

    return C, C_aux


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Generate PCG environments using 2-phase training")
    ap.add_argument("--phase1_model", type=str, required=True, help="Path to Phase 1 model (.zip)")
    ap.add_argument("--phase2_model", type=str, required=True, help="Path to Phase 2 model (.zip)")
    ap.add_argument("--n", type=int, default=16, help="Number of samples")
    ap.add_argument("--size", type=int, default=13)
    ap.add_argument("--phase1_max_steps", type=int, default=200, help="Max steps for Phase 1")
    ap.add_argument("--phase2_max_steps", type=int, default=80, help="Max steps for Phase 2")
    ap.add_argument("--wall_target", type=float, default=0.35, help="Target wall ratio for Phase 1")
    ap.add_argument("--n_objects", type=int, default=1, help="Number of objects (and goals). 1 = single-object, >1 = multi-object (MO-SeGMaN)")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--deterministic", action="store_true", help="Greedy actions (WARNING: Often produces empty grids)")
    ap.add_argument("--stochastic", action="store_true", help="Sample actions (overrides --deterministic, default)")
    ap.add_argument("--phase1_deterministic", action="store_true", help="Use Phase 1 model deterministically when generating base puzzles (WARNING: Often produces empty grids)")
    ap.add_argument("--critical_only", action="store_true", help="Only generate movable-critical puzzles (requires movables to solve)")
    ap.add_argument("--max_critical_attempts", type=int, default=1000, help="Max attempts to find critical puzzle when --critical_only is used")
    args = ap.parse_args()

    # Determine deterministic mode: default to stochastic (False) unless explicitly requested
    # Stochastic is the default because deterministic often produces empty grids
    deterministic = args.deterministic and not args.stochastic
    phase1_deterministic = args.phase1_deterministic and not args.stochastic
    
    if deterministic or phase1_deterministic:
        print("WARNING: Using deterministic mode - this often produces empty or low-quality grids!")
        print("         Consider using --stochastic for better results.")

    # Create Phase 1 environment and load model
    print(f"Loading Phase 1 model from {args.phase1_model}...")
    phase1_env = DummyVecEnv([make_phase1_env(args.size, args.phase1_max_steps, args.seed, args.wall_target, args.n_objects)])
    phase1_model = PPO.load(args.phase1_model, env=phase1_env, device="auto")

    # Create Phase 2 environment and load model
    print(f"Loading Phase 2 model from {args.phase2_model}...")
    phase2_env = DummyVecEnv([make_phase2_env(
        args.phase1_model,  # Phase 2 needs Phase 1 model path to generate base puzzles
        args.size,
        args.phase2_max_steps,
        args.seed,
        phase1_deterministic,  # Use computed value (respects --stochastic override)
        args.n_objects
    )])
    phase2_model = PPO.load(args.phase2_model, env=phase2_env, device="auto")

    # Create output directory
    case_id = random.randint(0, 10000)
    os.makedirs(f"ry_config/case_run_id_{case_id}", exist_ok=True)
    print(f"Output directory: ry_config/case_run_id_{case_id}/")

    # Generate n samples
    valid_count = 0
    max_retries = 50  # Maximum retries per phase before giving up
    
    for i in range(args.n):
        print(f"\nGenerating sample {i+1}/{args.n}...")
        
        # Step 1: Generate base puzzle with Phase 1 (retry until valid)
        print("  Phase 1: Generating base puzzle...")
        phase1_valid = False
        base_grid = None
        phase1_meta = None
        
        for retry in range(max_retries):
            base_grid, phase1_meta = rollout_phase1(phase1_env, phase1_model, args.phase1_max_steps, deterministic)
            phase1_valid = phase1_meta[3]
            
            if phase1_valid:
                break
            else:
                print(f"    Retry {retry + 1}/{max_retries}: Phase 1 generated invalid puzzle, retrying...")
        
        if not phase1_valid:
            print(f"  Error: Failed to generate valid Phase 1 puzzle after {max_retries} retries, skipping sample...")
            continue
        
        # Step 2: Add movables with Phase 2 (retry until valid, and critical if requested)
        print("  Phase 2: Adding movable obstacles...")
        phase2_valid = False
        phase2_critical = False
        final_grid = None
        phase2_meta = None
        phase2_metrics = None
        
        critical_attempts = 0
        max_critical_attempts = args.max_critical_attempts if args.critical_only else max_retries
        
        for retry in range(max_retries):
            result = rollout_phase2(phase2_env, phase2_model, args.phase2_max_steps, deterministic)
            final_grid, phase2_meta = result[0], result[1]
            phase2_valid = phase2_meta[3]
            phase2_metrics = phase2_meta[4] if len(phase2_meta) > 4 else decode_phase2_metrics([{}])
            
            # Check if valid and (if critical_only, also check if critical)
            if phase2_valid:
                if args.critical_only:
                    phase2_critical = phase2_metrics.get("movable_critical", False)
                    critical_attempts += 1
                    if phase2_critical:
                        break
                    else:
                        if critical_attempts < max_critical_attempts:
                            print(f"    Retry {critical_attempts}/{max_critical_attempts}: Phase 2 puzzle not movable-critical, retrying...")
                        else:
                            print(f"    Warning: Reached max critical attempts ({max_critical_attempts}), using non-critical puzzle")
                            break
                else:
                    break
            else:
                print(f"    Retry {retry + 1}/{max_retries}: Phase 2 generated invalid puzzle, retrying...")
        
        if not phase2_valid:
            print(f"  Error: Failed to generate valid Phase 2 puzzle after {max_retries} retries, skipping sample...")
            continue
        
        if args.critical_only and not phase2_critical:
            print(f"  Warning: Generated puzzle is not movable-critical (attempted {critical_attempts} times)")
        
        # Both phases succeeded, proceed with conversion
        valid_count += 1
        
        # Step 3: Convert to rai Config files
        print("  Converting to .g files...")
        C, C_aux = grid_to_rai_config(final_grid, case_id, i, args.size, args.n_objects)
        
        # Save config files
        os.makedirs(f"ry_config/case_run_id_{case_id}/pcg-{i}", exist_ok=True)
        new_C = C.write()
        new_C_aux = C_aux.write()
        
        with open(f"ry_config/case_run_id_{case_id}/pcg-{i}/pcg-{i}.g", "w") as f:
            f.write(new_C)
        with open(f"ry_config/case_run_id_{case_id}/pcg-{i}/pcg-{i}-aux.g", "w") as f:
            f.write(new_C_aux)
        
        print(f"  Saved: pcg-{i}.g and pcg-{i}-aux.g")
        print(f"  Phase 1 metrics: wall_ratio={phase1_meta[2]:.3f}, L1={phase1_meta[0]}, L2={phase1_meta[1]}")
        if phase2_metrics:
            critical_str = "✓ CRITICAL" if phase2_metrics.get("movable_critical", False) else "not critical"
            print(f"  Phase 2 metrics: valid={phase2_valid}, {critical_str}, n_movable={phase2_metrics.get('n_movable', 0)}")
        else:
            print(f"  Phase 2 metrics: valid={phase2_valid}")

    print(f"\n=== Generation Complete ===")
    print(f"Valid samples: {valid_count}/{args.n}")
    print(f"Output directory: ry_config/case_run_id_{case_id}/")


if __name__ == "__main__":
    main()

