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

    keys = ["wall_ratio","adj_per_wall","iso_frac","L1","L2","valid","n_movable","movable_ratio","solid_ratio"]
    acc = {k: [] for k in ["reward"]+keys}

    for _ in range(args.episodes):
        rew, info = eval_once(env, model, args.max_steps, args.deterministic)
        acc["reward"].append(rew)
        for k in keys:
            if k in info:
                acc[k].append(info[k])

    print(f"Episodes: {args.episodes}  deterministic={args.deterministic}")
    print(f"avg_reward   = {np.mean(acc['reward']):.3f}  ± {np.std(acc['reward']):.3f}")
    for k in keys:
        if len(acc[k]):
            print(f"avg_{k:11s}= {np.mean(acc[k]):.3f}  (min {np.min(acc[k]):.3f}, max {np.max(acc[k]):.3f})")

if __name__ == "__main__":
    main()
