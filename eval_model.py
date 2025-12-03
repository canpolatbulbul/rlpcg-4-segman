import argparse, numpy as np
from stable_baselines3 import PPO
from grid_pcg_env import GridPCGEnv

def eval_once(env, model, max_steps=48, deterministic=False):
    obs, _ = env.reset()
    ep_rew = 0.0
    info_out = None
    for _ in range(max_steps):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, r, term, trunc, info = env.step(int(action))
        ep_rew += float(r)
        if term or trunc:
            info_out = info
            break
    return ep_rew, info_out or {}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--episodes", type=int, default=128)
    ap.add_argument("--size", type=int, default=13)
    ap.add_argument("--max_steps", type=int, default=48)
    ap.add_argument("--deterministic", action="store_true")
    args = ap.parse_args()

    env = GridPCGEnv(size=args.size, max_steps=args.max_steps)
    model = PPO.load(args.model, env=env, device="auto")

    keys = ["wall_ratio","adj_per_wall","iso_frac","L1","L2","valid","n_movable","movable_ratio","solid_ratio",
            "relaxed_solvable","strict_solvable","movable_critical","L1_relaxed","L2_relaxed"]
    acc = {k: [] for k in ["reward"]+keys}

    for _ in range(args.episodes):
        rew, info = eval_once(env, model, args.max_steps, args.deterministic)
        acc["reward"].append(rew)
        for k in keys:
            if k in info:
                acc[k].append(info[k])

    print(f"Episodes: {args.episodes}  deterministic={args.deterministic}")
    print(f"avg_reward   = {np.mean(acc['reward']):.3f}  ± {np.std(acc['reward']):.3f}")
    
    # Solvability and movable-critical statistics
    if len(acc.get("valid", [])) > 0:
        valid_count = sum(acc["valid"])
        valid_pct = 100.0 * valid_count / len(acc["valid"])
        print(f"\n=== Solvability Metrics ===")
        print(f"Valid levels: {valid_count}/{len(acc['valid'])} ({valid_pct:.1f}%)")
        
        if len(acc.get("relaxed_solvable", [])) > 0:
            relaxed_count = sum(acc["relaxed_solvable"])
            relaxed_pct = 100.0 * relaxed_count / len(acc["relaxed_solvable"])
            print(f"Relaxed solvable: {relaxed_count}/{len(acc['relaxed_solvable'])} ({relaxed_pct:.1f}%)")
        
        if len(acc.get("strict_solvable", [])) > 0:
            strict_count = sum(acc["strict_solvable"])
            strict_pct = 100.0 * strict_count / len(acc["strict_solvable"])
            print(f"Strict solvable: {strict_count}/{len(acc['strict_solvable'])} ({strict_pct:.1f}%)")
        
        if len(acc.get("movable_critical", [])) > 0:
            critical_count = sum(acc["movable_critical"])
            critical_pct = 100.0 * critical_count / max(1, valid_count)  # % of valid levels
            print(f"Movable-critical: {critical_count}/{valid_count} ({critical_pct:.1f}% of valid levels)")
    
    print(f"\n=== Structural Metrics ===")
    for k in ["wall_ratio","adj_per_wall","iso_frac","L1","L2","n_movable","movable_ratio","solid_ratio"]:
        if len(acc.get(k, [])):
            print(f"avg_{k:15s}= {np.mean(acc[k]):.3f}  (min {np.min(acc[k]):.3f}, max {np.max(acc[k]):.3f})")
    
    if len(acc.get("L1_relaxed", [])) > 0:
        print(f"\n=== Path Lengths (Relaxed) ===")
        print(f"avg_L1_relaxed = {np.mean([x for x in acc['L1_relaxed'] if x > 0]):.1f}")
        print(f"avg_L2_relaxed = {np.mean([x for x in acc['L2_relaxed'] if x > 0]):.1f}")

if __name__ == "__main__":
    main()
