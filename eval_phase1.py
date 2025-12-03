# eval_phase1.py
# Evaluation script for Phase 1 model

import argparse
import numpy as np
from stable_baselines3 import PPO
from phase1_env import Phase1Env


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
    ap = argparse.ArgumentParser(description="Evaluate Phase 1 model")
    ap.add_argument("--model", required=True, help="Path to Phase 1 model (.zip)")
    ap.add_argument("--episodes", type=int, default=256)
    ap.add_argument("--size", type=int, default=13)
    ap.add_argument("--max_steps", type=int, default=150)
    ap.add_argument("--wall_target", type=float, default=0.25)
    ap.add_argument("--deterministic", action="store_true")
    args = ap.parse_args()
    
    env = Phase1Env(size=args.size, max_steps=args.max_steps, wall_target=args.wall_target)
    model = PPO.load(args.model, env=env, device="auto")
    
    keys = ["wall_ratio", "adj_per_wall", "iso_frac", "L1", "L2", "valid"]
    acc = {k: [] for k in ["reward"] + keys}
    
    for _ in range(args.episodes):
        rew, info = eval_once(env, model, args.deterministic)
        acc["reward"].append(rew)
        for k in keys:
            if k in info:
                acc[k].append(info[k])
    
    print(f"Episodes: {args.episodes}  deterministic={args.deterministic}")
    print(f"avg_reward = {np.mean(acc['reward']):.3f}  ± {np.std(acc['reward']):.3f}")
    
    # Statistics
    if len(acc.get("valid", [])) > 0:
        valid_count = sum(acc["valid"])
        valid_pct = 100.0 * valid_count / len(acc["valid"])
        print(f"\n=== Phase 1 Metrics ===")
        print(f"Valid levels: {valid_count}/{len(acc['valid'])} ({valid_pct:.1f}%)")
        print(f"(Note: In Phase 1, valid levels should be solvable since no movables)")
    
    print(f"\n=== Structural Metrics ===")
    for k in ["wall_ratio", "adj_per_wall", "iso_frac", "L1", "L2"]:
        if len(acc.get(k, [])):
            print(f"avg_{k:15s} = {np.mean(acc[k]):.3f}  (min {np.min(acc[k]):.3f}, max {np.max(acc[k]):.3f})")


if __name__ == "__main__":
    main()

