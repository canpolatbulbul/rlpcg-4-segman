# eval_phase2.py
# Evaluation script for Phase 2 model

import argparse
import numpy as np
from stable_baselines3 import PPO
from phase2_env import Phase2Env


def eval_once(env, model, deterministic=False):
    """Run one evaluation episode."""
    obs, _ = env.reset()
    ep_rew = 0.0
    info_out = {}
    
    for step in range(env.max_steps):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, r, term, trunc, info = env.step(int(action))
        ep_rew += float(r)
        if term or trunc:
            info_out = info
            break
    
    return ep_rew, info_out


def main():
    ap = argparse.ArgumentParser(description="Evaluate Phase 2 model")
    ap.add_argument("--model", required=True, help="Path to Phase 2 model (.zip)")
    ap.add_argument("--phase1_model", required=True, help="Path to Phase 1 model (.zip)")
    ap.add_argument("--episodes", type=int, default=256)
    ap.add_argument("--size", type=int, default=13)
    ap.add_argument("--max_steps", type=int, default=80)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--phase1_deterministic", action="store_true",
                    help="Use Phase 1 model deterministically")
    args = ap.parse_args()
    
    env = Phase2Env(
        phase1_model_path=args.phase1_model,
        size=args.size,
        max_steps=args.max_steps,
        phase1_deterministic=args.phase1_deterministic
    )
    model = PPO.load(args.model, env=env, device="auto")
    
    keys = [
        "valid", "relaxed_solvable", "strict_solvable", "movable_critical",
        "n_movable", "L1_relaxed", "L2_relaxed", "L1_strict", "L2_strict"
    ]
    acc = {k: [] for k in ["reward"] + keys}
    
    for _ in range(args.episodes):
        rew, info = eval_once(env, model, args.deterministic)
        acc["reward"].append(rew)
        for k in keys:
            if k in info:
                acc[k].append(info[k])
            elif k in ["relaxed_solvable", "strict_solvable", "movable_critical"]:
                acc[k].append(False)
    
    print(f"Episodes: {args.episodes}  deterministic={args.deterministic}")
    print(f"avg_reward = {np.mean(acc['reward']):.3f}  ± {np.std(acc['reward']):.3f}")
    
    # Solvability and movable-critical statistics
    if len(acc.get("valid", [])) > 0:
        valid_count = sum(acc["valid"])
        valid_pct = 100.0 * valid_count / len(acc["valid"])
        print(f"\n=== Phase 2 Metrics ===")
        print(f"Valid levels: {valid_count}/{len(acc['valid'])} ({valid_pct:.1f}%)")
        
        if len(acc.get("relaxed_solvable", [])) > 0:
            relaxed_count = sum(int(x) for x in acc["relaxed_solvable"])
            relaxed_pct = 100.0 * relaxed_count / len(acc["relaxed_solvable"])
            print(f"Relaxed solvable: {relaxed_count}/{len(acc['relaxed_solvable'])} ({relaxed_pct:.1f}%)")
        
        if len(acc.get("strict_solvable", [])) > 0:
            strict_count = sum(int(x) for x in acc["strict_solvable"])
            strict_pct = 100.0 * strict_count / len(acc["strict_solvable"])
            print(f"Strict solvable: {strict_count}/{len(acc['strict_solvable'])} ({strict_pct:.1f}%)")
        
        if len(acc.get("movable_critical", [])) > 0:
            critical_count = sum(int(x) for x in acc["movable_critical"])
            critical_pct = 100.0 * critical_count / max(1, valid_count)
            print(f"Movable-critical: {critical_count}/{valid_count} ({critical_pct:.1f}% of valid levels)")
    
    print(f"\n=== Movable Statistics ===")
    if len(acc.get("n_movable", [])) > 0:
        print(f"avg_n_movable = {np.mean(acc['n_movable']):.1f}  (min {np.min(acc['n_movable']):.0f}, max {np.max(acc['n_movable']):.0f})")
    
    # Path lengths
    l1_relaxed_valid = [x for x in acc.get("L1_relaxed", []) if x > 0]
    l2_relaxed_valid = [x for x in acc.get("L2_relaxed", []) if x > 0]
    if len(l1_relaxed_valid) > 0 or len(l2_relaxed_valid) > 0:
        print(f"\n=== Path Lengths (Relaxed) ===")
        if len(l1_relaxed_valid) > 0:
            print(f"avg_L1_relaxed = {np.mean(l1_relaxed_valid):.1f} (from {len(l1_relaxed_valid)} solvable paths)")
        if len(l2_relaxed_valid) > 0:
            print(f"avg_L2_relaxed = {np.mean(l2_relaxed_valid):.1f} (from {len(l2_relaxed_valid)} solvable paths)")


if __name__ == "__main__":
    main()

