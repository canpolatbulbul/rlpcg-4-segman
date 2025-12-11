#!/usr/bin/env python3
"""
Quick convergence check from monitor.csv
Usage: python check_convergence.py [logdir]

Note: monitor.csv from VecMonitor only has basic metrics (r, l, t).
Custom metrics (wall_ratio, valid_pct) are in TensorBoard.
This script uses available monitor.csv data for convergence checking.
"""
import pandas as pd
import sys
import os

def main():
    logdir = sys.argv[1] if len(sys.argv) > 1 else "runs/phase1_2obj_17x17_wall0.32"
    csv_path = os.path.join(logdir, "monitor.csv")
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found")
        print("Usage: python check_convergence.py [logdir]")
        return
    
    try:
        df = pd.read_csv(csv_path, skiprows=1)
        
        if len(df) < 1000:
            print(f"Not enough data yet ({len(df)} rows). Need at least 1000 steps.")
            return
        
        # Check what columns are available
        available_cols = df.columns.tolist()
        print(f"Available columns: {available_cols}")
        print()
        
        # VecMonitor typically logs: r (reward), l (episode length), t (time)
        # Map to standard names
        if 'r' in df.columns:
            reward_col = 'r'
        elif 'ep_rew_mean' in df.columns:
            reward_col = 'ep_rew_mean'
        else:
            print("Error: No reward column found. Expected 'r' or 'ep_rew_mean'")
            return
        
        if 'l' in df.columns:
            ep_len_col = 'l'
        elif 'ep_len_mean' in df.columns:
            ep_len_col = 'ep_len_mean'
        else:
            print("Error: No episode length column found. Expected 'l' or 'ep_len_mean'")
            return
        
        # Get last 50k steps (or all if less)
        n_recent = min(50000, len(df))
        recent = df.tail(n_recent)
        
        # Calculate metrics from available data
        reward = recent[reward_col].mean()
        reward_std = recent[reward_col].std()
        reward_min = recent[reward_col].min()
        reward_max = recent[reward_col].max()
        
        # Calculate trend (compare last 1000 vs first 1000 of recent window)
        if len(recent) >= 2000:
            reward_trend = recent[reward_col].iloc[-1000:].mean() - recent[reward_col].iloc[:1000].mean()
        else:
            reward_trend = recent[reward_col].iloc[-len(recent)//2:].mean() - recent[reward_col].iloc[:len(recent)//2].mean()
        
        ep_len = recent[ep_len_col].mean()
        ep_len_std = recent[ep_len_col].std()
        ep_len_min = recent[ep_len_col].min()
        ep_len_max = recent[ep_len_col].max()
        
        # Check for custom metrics (may not be present)
        has_wall_ratio = 'wall_ratio' in df.columns
        has_valid_pct = 'valid_pct' in df.columns
        
        wall_ratio = None
        wall_std = None
        valid_pct = None
        
        if has_wall_ratio:
            wall_ratio = recent['wall_ratio'].mean()
            wall_std = recent['wall_ratio'].std()
        
        if has_valid_pct:
            valid_pct = recent['valid_pct'].mean()
        
        print("=" * 70)
        print(f"Convergence Check: Last {n_recent:,} episodes")
        print("=" * 70)
        print(f"Reward:         {reward:.2f} (std: {reward_std:.2f}, range: [{reward_min:.2f}, {reward_max:.2f}])")
        print(f"Reward trend:   {reward_trend:+.2f} (last 1k vs first 1k of recent window)")
        print(f"Episode length: {ep_len:.1f} (std: {ep_len_std:.1f}, range: [{ep_len_min:.1f}, {ep_len_max:.1f}])")
        if wall_ratio is not None:
            print(f"Wall ratio:     {wall_ratio:.3f} (std: {wall_std:.3f})")
        else:
            print(f"Wall ratio:     N/A (check TensorBoard for metrics/avg_wall_ratio)")
        if valid_pct is not None:
            print(f"Validity:       {valid_pct:.1f}%")
        else:
            print(f"Validity:       N/A (check TensorBoard for metrics/valid_pct)")
        print("=" * 70)
        print()
        
        # Convergence criteria (using available metrics)
        # For reward: use relative std (coefficient of variation) for high rewards
        reward_cv = reward_std / abs(reward) if reward != 0 else float('inf')
        reward_stable = (reward_cv < 0.15 or reward_std < 0.5) and abs(reward_trend) < 0.2
        # For episode length: std is expected to be higher when early termination works
        ep_len_stable = ep_len_std < 50  # More lenient for early termination variance
        reward_good = reward > 5.0
        ep_len_good = ep_len < 200  # Should show early termination
        
        # Wall ratio and validity checks (if available)
        wall_stable = None
        wall_good = None
        valid_good = None
        
        if wall_ratio is not None:
            wall_stable = wall_std < 0.02
            wall_good = wall_ratio > 0.28
        
        if valid_pct is not None:
            valid_good = valid_pct > 95.0
        
        # Convergence decision
        converged = reward_stable and ep_len_stable and reward_good and ep_len_good
        if wall_good is not None:
            converged = converged and wall_good
        if valid_good is not None:
            converged = converged and valid_good
        
        print("Convergence Status:")
        print(f"  Reward stable:         {'✅' if reward_stable else '❌'} (CV < 15% or std < 0.5, trend < 0.2)")
        if reward != 0:
            print(f"    Reward CV: {reward_cv*100:.1f}% (std={reward_std:.2f} / mean={reward:.2f})")
        print(f"  Episode length stable:  {'✅' if ep_len_stable else '❌'} (std < 50)")
        print(f"  Reward good:           {'✅' if reward_good else '❌'} (> 5.0)")
        print(f"  Episode length good:   {'✅' if ep_len_good else '❌'} (< 200, early termination)")
        if wall_stable is not None:
            print(f"  Wall ratio stable:     {'✅' if wall_stable else '❌'} (std < 0.02)")
        if wall_good is not None:
            print(f"  Wall ratio good:       {'✅' if wall_good else '❌'} (> 0.28)")
        if valid_good is not None:
            print(f"  Validity good:         {'✅' if valid_good else '❌'} (> 95%)")
        print()
        
        if converged:
            print("=" * 70)
            print("✅ CONVERGED - Safe to stop training!")
            print("=" * 70)
            print("All available metrics are stable and meet quality thresholds.")
            print("Further training is unlikely to improve significantly.")
            print()
            print("Note: For wall_ratio and validity, check TensorBoard:")
            print(f"  tensorboard --logdir {logdir}")
            print("  Look for: metrics/avg_wall_ratio and metrics/valid_pct")
        else:
            print("=" * 70)
            print("⏳ Still learning - Keep training")
            print("=" * 70)
            if not reward_stable:
                if reward != 0:
                    print(f"   - Reward still varying (CV={reward_cv*100:.1f}%, std={reward_std:.2f}, trend={reward_trend:+.2f})")
                else:
                    print(f"   - Reward still varying (std={reward_std:.2f}, trend={reward_trend:+.2f})")
            if not reward_good:
                print(f"   - Reward too low ({reward:.2f}, should be >5.0)")
            if not ep_len_stable:
                print(f"   - Episode length still varying (std={ep_len_std:.1f})")
            if not ep_len_good:
                print(f"   - Episode length too high ({ep_len:.1f}, should be <200 for early termination)")
            if wall_good is not None and not wall_good:
                print(f"   - Wall ratio too low ({wall_ratio:.3f}, should be >0.28)")
            if valid_good is not None and not valid_good:
                print(f"   - Validity too low ({valid_pct:.1f}%, should be >95%)")
            print()
            print("Recommendation: Check again after 200k more steps")
            print()
            print("For detailed metrics (wall_ratio, validity), check TensorBoard:")
            print(f"  tensorboard --logdir {logdir}")
        
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

