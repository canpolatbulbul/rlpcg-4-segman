# Early Stopping Guide for Phase 1 Training

## When to Stop Training (Convergence Indicators)

### ✅ **STOP if these conditions are met** (typically after 1-1.5M steps):

1. **Wall Ratio Stabilized**
   - `metrics/avg_wall_ratio` has been **stable** (within ±0.02) for **>200k steps**
   - Example: 0.30-0.32 for last 200k steps, not increasing further
   - **Check**: Look at last 200k-300k steps, should be flat line

2. **Validity & Solvability Stable**
   - `metrics/valid_pct` >95% and **stable** (not fluctuating)
   - `metrics/solvable_pct` >95% and **stable**
   - Both have been consistent for **>200k steps**

3. **Reward Plateaued**
   - `rollout/ep_rew_mean` has **stopped increasing** for **>200k steps**
   - Should be in range 6-10 and stable
   - **Check**: Last 200k steps should show flat or very slow increase (<0.1 per 100k steps)

4. **Episode Length Stabilized**
   - `rollout/ep_len_mean` has **stopped decreasing** for **>200k steps**
   - Should be in range 100-180 (early termination working)
   - **Check**: Should be relatively flat, not trending down further

5. **Learning Metrics Stable**
   - `train/explained_variance` >0.8 and **stable**
   - `train/approx_kl` <0.1 and **stable** (not decreasing further)
   - `train/policy_loss` and `train/value_loss` **not decreasing** (converged)

### 🎯 **"Good Enough" Thresholds** (Can stop even if not perfect):

If after **1M steps** you have:
- ✅ Wall ratio: **>0.28** (close enough to 0.32)
- ✅ Validity: **>95%**
- ✅ Solvability: **>95%**
- ✅ Reward: **>5.0** and stable
- ✅ Episode length: **<200** (early termination working)

**→ You can stop. The model is good enough for Phase 2.**

### ⚠️ **Keep Training If:**

1. **Wall ratio still increasing** (hasn't plateaued yet)
   - If it's at 0.25 and still trending up → keep going
   - If it's at 0.30 and flat → can stop

2. **Reward still increasing** (>0.2 per 100k steps)
   - Agent still learning → keep going

3. **Metrics still fluctuating** (not stable)
   - High variance in metrics → keep training

4. **Early termination not working** (ep_len_mean always 250)
   - Agent hasn't learned quality thresholds → keep training

## How to Check Convergence in TensorBoard

### Method 1: Visual Inspection
1. Open TensorBoard: `tensorboard --logdir runs/phase1_2obj_17x17_wall0.32`
2. Look at last **200k-300k steps** of these metrics:
   - `metrics/avg_wall_ratio` → Should be flat line
   - `rollout/ep_rew_mean` → Should be flat or very slow increase
   - `rollout/ep_len_mean` → Should be flat
3. If all are flat → **Converged, can stop**

### Method 2: Check Monitor CSV
```bash
# On server, check recent metrics
tail -100 runs/phase1_2obj_17x17_wall0.32/monitor.csv | \
  awk -F',' '{print $4, $5, $6}'  # wall_ratio, reward, episode_length
```

Look for:
- **Wall ratio**: Consistent values (e.g., all around 0.30-0.32)
- **Reward**: Consistent or very slow increase
- **Episode length**: Consistent (e.g., all around 120-150)

### Method 3: Statistical Check
If you have Python on server:
```python
import pandas as pd
import numpy as np

df = pd.read_csv('runs/phase1_2obj_17x17_wall0.32/monitor.csv', skiprows=1)
# Get last 50k steps
recent = df.tail(50000)

# Check if metrics are stable
wall_ratio_std = recent['wall_ratio'].std()
reward_std = recent['ep_rew_mean'].std()
ep_len_std = recent['ep_len_mean'].std()

# If std dev is small, metrics are stable
if wall_ratio_std < 0.02 and reward_std < 0.5 and ep_len_std < 20:
    print("CONVERGED - Can stop training")
```

## Typical Convergence Timeline

### Expected Progress:
- **0-500k steps**: Rapid learning, metrics changing quickly
- **500k-1M steps**: Slowing down, metrics stabilizing
- **1M-1.5M steps**: **Convergence zone** - metrics plateau
- **1.5M-2M steps**: Fine-tuning, minimal improvement

### When to Check:
- **After 1M steps**: First convergence check
- **After 1.5M steps**: Second check (likely converged)
- **After 2M steps**: Final check (definitely converged)

## Decision Tree

```
After 1M steps:
├─ Wall ratio >0.28 AND stable? 
│  ├─ Yes → Check other metrics
│  └─ No → Keep training
│
├─ Validity >95% AND stable?
│  ├─ Yes → Check other metrics
│  └─ No → Keep training
│
├─ Reward stable (not increasing)?
│  ├─ Yes → Check other metrics
│  └─ No → Keep training
│
└─ All metrics stable for >200k steps?
   ├─ Yes → ✅ STOP (Converged)
   └─ No → Keep training, check again at 1.5M
```

## Red Flags (Don't Stop Yet)

1. **Wall ratio still increasing** → Keep training
2. **Reward still increasing** (>0.2 per 100k) → Keep training
3. **High variance in metrics** → Keep training
4. **Metrics below thresholds** → Keep training

## Green Lights (Safe to Stop)

1. ✅ **Wall ratio stable** at 0.28-0.32 for >200k steps
2. ✅ **Validity stable** at >95% for >200k steps
3. ✅ **Reward stable** (increase <0.1 per 100k steps)
4. ✅ **Episode length stable** (early termination working)
5. ✅ **All metrics plateaued** for >200k steps

## Quick Check Script

Save this as `check_convergence.py` on your server:

```python
#!/usr/bin/env python3
"""Quick convergence check from monitor.csv"""
import pandas as pd
import sys

logdir = sys.argv[1] if len(sys.argv) > 1 else "runs/phase1_2obj_17x17_wall0.32"
csv_path = f"{logdir}/monitor.csv"

try:
    df = pd.read_csv(csv_path, skiprows=1)
    recent = df.tail(50000)  # Last 50k steps
    
    wall_ratio = recent['wall_ratio'].mean()
    wall_std = recent['wall_ratio'].std()
    reward = recent['ep_rew_mean'].mean()
    reward_std = recent['ep_rew_mean'].std()
    ep_len = recent['ep_len_mean'].mean()
    ep_len_std = recent['ep_len_mean'].std()
    
    print(f"Last 50k steps:")
    print(f"  Wall ratio: {wall_ratio:.3f} ± {wall_std:.3f}")
    print(f"  Reward: {reward:.2f} ± {reward_std:.2f}")
    print(f"  Episode length: {ep_len:.1f} ± {ep_len_std:.1f}")
    print()
    
    # Convergence check
    converged = (
        wall_std < 0.02 and
        reward_std < 0.5 and
        ep_len_std < 20 and
        wall_ratio > 0.28 and
        reward > 5.0
    )
    
    if converged:
        print("✅ CONVERGED - Safe to stop training!")
    else:
        print("⏳ Still learning - Keep training")
        if wall_std >= 0.02:
            print(f"   - Wall ratio still varying (std={wall_std:.3f})")
        if reward_std >= 0.5:
            print(f"   - Reward still varying (std={reward_std:.2f})")
        if wall_ratio < 0.28:
            print(f"   - Wall ratio too low ({wall_ratio:.3f} < 0.28)")
        if reward < 5.0:
            print(f"   - Reward too low ({reward:.2f} < 5.0)")
            
except Exception as e:
    print(f"Error: {e}")
    print("Make sure monitor.csv exists and training has started")
```

Usage:
```bash
python check_convergence.py runs/phase1_2obj_17x17_wall0.32
```

## Summary: When to Stop

**Stop training if:**
- ✅ All metrics **stable** (flat) for **>200k steps** AND
- ✅ Wall ratio **>0.28** AND
- ✅ Validity **>95%** AND
- ✅ Reward **>5.0**

**Typical stopping point:** **1-1.5M steps** (often converges before 2M)

**Don't wait for 2M if metrics have plateaued!** You're just wasting compute.

