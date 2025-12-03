# train_phase1.py
# Training script for Phase 1: Structure Learning

import os
import argparse
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.logger import configure

from phase1_env import Phase1Env
from small_cnn import SmallGridCNN


def make_env(seed: int, size: int, max_steps: int, wall_target: float):
    def _thunk():
        return Phase1Env(
            size=size,
            max_steps=max_steps,
            seed=seed,
            wall_target=wall_target
        )
    return _thunk


class MetricsCallback(BaseCallback):
    """Tracks and logs Phase 1 metrics."""
    def __init__(self, log_freq: int = 500, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.episode_infos = []
        
    def _on_step(self) -> bool:
        if self.locals.get("infos") is not None:
            for info in self.locals["infos"]:
                if isinstance(info, dict) and info.get("valid") is not None:
                    if "final_grid" in info:
                        self.episode_infos.append(info)
        
        if len(self.episode_infos) >= self.log_freq:
            self._log_metrics()
            self.episode_infos.clear()
            
        return True
    
    def _log_metrics(self):
        if not self.episode_infos:
            return
            
        n = len(self.episode_infos)
        valid_count = sum(1 for info in self.episode_infos if info.get("valid", 0) == 1)
        
        # Solvability (all should be solvable if valid, since no movables)
        solvable_count = valid_count  # In Phase 1, if valid, should be solvable
        
        # Average metrics
        wall_ratio_avg = sum(info.get("wall_ratio", 0.0) for info in self.episode_infos) / n
        l1_avg = sum(info.get("L1", 0) for info in self.episode_infos) / max(1, n)
        l2_avg = sum(info.get("L2", 0) for info in self.episode_infos) / max(1, n)
        
        self.logger.record("metrics/valid_pct", 100.0 * valid_count / max(1, n))
        self.logger.record("metrics/solvable_pct", 100.0 * solvable_count / max(1, n))
        self.logger.record("metrics/avg_wall_ratio", wall_ratio_avg)
        self.logger.record("metrics/avg_l1", l1_avg)
        self.logger.record("metrics/avg_l2", l2_avg)


def main():
    p = argparse.ArgumentParser(description="Train Phase 1: Structure Learning")
    
    # Environment
    p.add_argument("--size", type=int, default=13)
    p.add_argument("--max_steps", type=int, default=150)
    p.add_argument("--min_steps", type=int, default=50)
    p.add_argument("--n_envs", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wall_target", type=float, default=0.25)
    
    # Training
    p.add_argument("--total_timesteps", type=int, default=1_000_000)
    p.add_argument("--n_steps", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--ent_coef", type=float, default=0.05)
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--clip_range", type=float, default=0.2)
    
    # I/O
    p.add_argument("--logdir", type=str, default="runs/phase1_structure")
    p.add_argument("--no_tb", action="store_true")
    p.add_argument("--resume", type=str, default=None, help="Path to .zip model to resume from")
    p.add_argument("--eval_every", type=int, default=50_000)
    p.add_argument("--eval_episodes", type=int, default=32)
    
    args = p.parse_args()
    
    os.makedirs(args.logdir, exist_ok=True)
    print("Using cuda device" if torch.cuda.is_available() else "Using cpu device")
    
    # Vectorized environments
    venv_fns = [make_env(args.seed + i, args.size, args.max_steps, args.wall_target)
                for i in range(args.n_envs)]
    train_env = SubprocVecEnv(venv_fns) if args.n_envs > 1 else DummyVecEnv(venv_fns)
    train_env = VecMonitor(train_env, filename=os.path.join(args.logdir, "monitor.csv"))
    
    eval_env = DummyVecEnv([make_env(10_000, args.size, args.max_steps, args.wall_target)])
    eval_env = VecMonitor(eval_env)
    
    # Logger
    tb_log = None if args.no_tb else args.logdir
    if tb_log:
        configure(tb_log, ["stdout", "tensorboard"])
    
    # Policy
    policy_kwargs = dict(
        features_extractor_class=SmallGridCNN,
        features_extractor_kwargs={"features_dim": 256},
        normalize_images=False,
    )
    
    # Create or resume model
    if args.resume:
        print(f"Resuming from: {args.resume}")
        model = PPO.load(args.resume, env=train_env, device="auto", print_system_info=False)
        model.n_steps = args.n_steps
        model.batch_size = args.batch_size
        model.learning_rate = args.lr
        model.gamma = args.gamma
        model.ent_coef = args.ent_coef
        model.tensorboard_log = tb_log
        model.policy_kwargs.update(policy_kwargs)
    else:
        model = PPO(
            "CnnPolicy",
            env=train_env,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            clip_range=args.clip_range,
            gamma=args.gamma,
            ent_coef=args.ent_coef,
            verbose=1,
            tensorboard_log=tb_log,
            policy_kwargs=policy_kwargs,
        )
    
    # Callbacks
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=args.logdir,
        log_path=args.logdir,
        eval_freq=max(args.eval_every // args.n_envs, 1),
        n_eval_episodes=args.eval_episodes,
        deterministic=False,
    )
    
    metrics_cb = MetricsCallback(log_freq=500)
    
    callbacks = [eval_cb, metrics_cb]
    
    # Train
    print(f"\n=== Phase 1 Training: Structure Learning ===")
    print(f"Goal: Learn to place walls to create well-structured, solvable puzzles")
    print(f"Action space: WALL and EMPTY only")
    print(f"Entities: Pre-placed randomly at reset\n")
    
    model.learn(total_timesteps=args.total_timesteps, callback=callbacks, reset_num_timesteps=not args.resume)
    model.save(os.path.join(args.logdir, "phase1_final"))
    print(f"\nSaved final model to: {args.logdir}/phase1_final.zip")
    print(f"Use this model for Phase 2 training!")
    
    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()

