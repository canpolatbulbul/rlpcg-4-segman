# train_ppo.py
import os
import argparse
import math
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.logger import configure

from grid_pcg_env import GridPCGEnv
from small_cnn import SmallGridCNN  # your existing extractor


def make_env(seed: int, size: int, max_steps: int, use_curriculum: bool, wall_target: float):
    def _thunk():
        return GridPCGEnv(size=size, max_steps=max_steps, seed=seed,
                          use_curriculum=use_curriculum, wall_target=wall_target)
    return _thunk


# ---- Optional entropy annealing schedule ----
def make_ent_coef_schedule(start: float, end: float, total_steps: int):
    def schedule(progress_remaining: float):
        # progress_remaining goes 1 -> 0 over training
        step = (1.0 - progress_remaining) * total_steps
        # cosine from start -> end
        cos = 0.5 * (1 + math.cos(math.pi * min(max(step / total_steps, 0.0), 1.0)))
        return end + (start - end) * cos
    return schedule


class StopAfterSteps(BaseCallback):
    """Hard stop after N environment steps (for quick sanity checks)."""
    def __init__(self, max_env_steps: int, verbose: int = 0):
        super().__init__(verbose)
        self.max_env_steps = max_env_steps
    def _on_step(self) -> bool:
        return self.model.num_timesteps < self.max_env_steps


def main():
    p = argparse.ArgumentParser()
    # env
    p.add_argument("--size", type=int, default=13)
    p.add_argument("--max_steps", type=int, default=192)        # large to allow walls
    p.add_argument("--n_envs", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--use_curriculum", action="store_true", help="Start resets with 0-2 entities")
    p.add_argument("--wall_target", type=float, default=0.25)

    # train
    p.add_argument("--total_timesteps", type=int, default=1_000_000)
    p.add_argument("--n_steps", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--ent_coef", type=float, default=0.05)  # you said this worked better
    p.add_argument("--ent_coef_final", type=float, default=None, help="If set, cosine-anneal ent_coef -> this value")
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--clip_range", type=float, default=0.2)

    # io
    p.add_argument("--logdir", type=str, default="runs/ppo_grid_corridors_v2")
    p.add_argument("--no_tb", action="store_true")
    p.add_argument("--resume", type=str, default=None, help="Path to a .zip model to resume from")
    p.add_argument("--checkpoint_every", type=int, default=200_000)
    p.add_argument("--eval_every", type=int, default=50_000)
    p.add_argument("--eval_episodes", type=int, default=32)
    p.add_argument("--pause_after", type=int, default=0, help="Hard stop after this many env steps (0=off)")
    args = p.parse_args()

    os.makedirs(args.logdir, exist_ok=True)
    print("Using cuda device" if torch.cuda.is_available() else "Using cpu device")

    # --------- Vec envs ---------
    # SubprocVecEnv is much faster when episodes are long
    venv_fns = [make_env(args.seed + i, args.size, args.max_steps, args.use_curriculum, args.wall_target)
                for i in range(args.n_envs)]
    train_env = SubprocVecEnv(venv_fns) if args.n_envs > 1 else DummyVecEnv(venv_fns)
    train_env = VecMonitor(train_env, filename=os.path.join(args.logdir, "monitor.csv"))

    eval_env = DummyVecEnv([make_env(10_000, args.size, args.max_steps, args.use_curriculum, args.wall_target)])
    eval_env = VecMonitor(eval_env)

    # --------- logger / TB ---------
    tb_log = None if args.no_tb else args.logdir
    if tb_log:
        configure(tb_log, ["stdout", "tensorboard"])

    # --------- entropy schedule (optional) ---------
    ent_coef = args.ent_coef
    if args.ent_coef_final is not None:
        ent_coef = make_ent_coef_schedule(args.ent_coef, args.ent_coef_final, args.total_timesteps)

    policy_kwargs = dict(
        features_extractor_class=SmallGridCNN,
        features_extractor_kwargs={"features_dim": 256},
        normalize_images=False,
    )

    # --------- create / resume model ---------
    if args.resume:
        print(f"Resuming from: {args.resume}")
        model = PPO.load(args.resume, env=train_env, device="auto", print_system_info=False)
        # You can still override some trainer-level hyperparams:
        model.n_steps = args.n_steps
        model.batch_size = args.batch_size
        model.learning_rate = args.lr
        model.gamma = args.gamma
        model.ent_coef = ent_coef
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
            ent_coef=ent_coef,
            verbose=1,
            tensorboard_log=tb_log,
            policy_kwargs=policy_kwargs,
        )

    # --------- callbacks (eval + checkpoints + optional pause) ---------
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=args.logdir,
        log_path=args.logdir,
        eval_freq=max(args.eval_every // args.n_envs, 1),
        n_eval_episodes=args.eval_episodes,
        deterministic=False,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=max(args.checkpoint_every // args.n_envs, 1),
        save_path=args.logdir,
        name_prefix="ppo_ckpt",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    callbacks = [eval_cb, ckpt_cb]
    if args.pause_after > 0:
        callbacks.append(StopAfterSteps(args.pause_after))

    # --------- learn ---------
    model.learn(total_timesteps=args.total_timesteps, callback=callbacks, reset_num_timesteps=not args.resume)
    model.save(os.path.join(args.logdir, "ppo_grid_movable"))
    print(f"Saved final model to: {args.logdir}/ppo_grid_movable.zip")

    # close cleanly
    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
