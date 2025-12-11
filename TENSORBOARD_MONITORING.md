# TensorBoard Monitoring Guide for Phase 1 (2-Object Training)

## Critical Metrics to Watch

### 1. **Wall Ratio** (`metrics/avg_wall_ratio`)
- **Target**: Should approach **0.32** (your wall_target)
- **Acceptable range**: 0.28-0.32
- **Warning signs**:
  - ❌ Stuck below **0.25**: Agent not placing enough walls
  - ❌ Stuck at **0.20-0.22**: Same issue, needs intervention
  - ❌ Decreasing over time: Agent learning to avoid walls (bad)
  - ✅ Increasing toward 0.32: Good, learning correctly
  - ✅ Stable at 0.28-0.32: Acceptable (may be slightly below target)

**Action if stuck low**: Increase `wall_ratio_penalty` in `phase1_env.py` (currently 30.0)

### 2. **Validity** (`metrics/valid_pct`)
- **Target**: **>95%** (ideally >98%)
- **Warning signs**:
  - ❌ Below **90%**: Too many invalid puzzles (missing entities)
  - ❌ Decreasing over time: Agent breaking entity placement
  - ❌ Stuck at **<85%**: Serious problem, check entity placement logic
  - ✅ Stable at >95%: Good

**Action if low**: Check entity placement constraints, may need to adjust `entity_min_distance`

### 3. **Solvability** (`metrics/solvable_pct`)
- **Target**: **>95%** (should match validity since no movables in Phase 1)
- **Warning signs**:
  - ❌ Below **90%**: Agent placing walls that block paths
  - ❌ Decreasing: Agent learning to break solvability (very bad)
  - ❌ Much lower than validity: Walls are blocking paths
  - ✅ Matches validity: Good (all valid puzzles are solvable)

**Action if low**: May need to reduce `wall_target` or increase per-step solvability rewards

### 4. **Episode Reward** (`rollout/ep_rew_mean`)
- **Target**: **Positive and increasing** (typically 5-10 range)
- **Warning signs**:
  - ❌ Negative: Agent getting heavy penalties (invalid/unsolvable)
  - ❌ Decreasing over time: Learning wrong behavior
  - ❌ Stuck at low value (<2): Not learning effectively
  - ❌ High variance (large std dev): Unstable learning
  - ✅ Increasing toward 6-10: Good learning
  - ✅ Stable positive: Acceptable if other metrics good

**Action if negative/decreasing**: Check reward structure, may need tuning

### 5. **Episode Length** (`rollout/ep_len_mean`)
- **Target**: **<200** (should show early termination when quality met)
- **Expected**: 100-180 steps (early termination when thresholds met)
- **Warning signs**:
  - ❌ Always **250** (max_steps): Never early terminating
    - Means: Quality thresholds not being met
    - Check: Wall ratio, solvability, path lengths
  - ❌ Very short (<50): Terminating too early (before min_steps)
    - Shouldn't happen (min_steps=60 blocks this)
  - ✅ Decreasing over time: Learning to achieve quality faster
  - ✅ Stable at 100-180: Good, early termination working

**Action if always max_steps**: Check early termination conditions, may need to relax thresholds

### 6. **Path Lengths** (`metrics/avg_l1`, `metrics/avg_l2`)
- **Target**: **Non-trivial** (not all adjacent)
- **Expected**: L1+L2 total ~40-60 (sum across both object-goal pairs)
- **Warning signs**:
  - ❌ Very short (<5 each): Entities too close (trivial puzzles)
  - ❌ Decreasing to very low: Agent learning to place entities closer (bad)
  - ✅ Stable at reasonable values: Good
  - ✅ Increasing slightly: Acceptable (more complex puzzles)

**Note**: These are averages across both pairs, so expect higher values than single-object

### 7. **Learning Metrics**

#### **Explained Variance** (`train/explained_variance`)
- **Target**: **>0.8** (ideally >0.9)
- **Warning signs**:
  - ❌ Below **0.5**: Poor learning signal
  - ❌ Negative: Very bad, rewards not predictive
  - ✅ >0.8: Good learning

#### **Policy Loss** (`train/policy_loss`)
- **Target**: **Decreasing and stabilizing** (typically -0.1 to -0.5)
- **Warning signs**:
  - ❌ Very large negative (<-2): Over-optimizing, may collapse
  - ❌ Increasing: Learning wrong direction
  - ✅ Stabilizing around -0.2 to -0.5: Good

#### **Value Loss** (`train/value_loss`)
- **Target**: **Decreasing and stabilizing** (typically 0.01-0.1)
- **Warning signs**:
  - ❌ Very high (>1.0): Poor value estimation
  - ❌ Increasing: Value function not learning
  - ✅ Decreasing and low: Good

#### **Entropy** (`train/entropy`)
- **Target**: **Stable** (typically -5 to -3)
- **Warning signs**:
  - ❌ Very low (<-8): Over-exploiting, may collapse
  - ❌ Very high (>0): Too much exploration
  - ✅ Stable around -4 to -6: Good balance

#### **KL Divergence** (`train/approx_kl`)
- **Target**: **<0.1** (ideally <0.05)
- **Warning signs**:
  - ❌ Very high (>0.5): Policy changing too fast, unstable
  - ❌ Spiking: Training instability
  - ✅ Stable and low: Good

## Red Flags (Stop Training)

1. **Wall ratio stuck <0.20** for >500k steps
2. **Validity <85%** consistently
3. **Solvability <80%** consistently
4. **Reward negative** and not improving
5. **Explained variance <0.3** consistently
6. **KL divergence >1.0** (training instability)

## Good Signs (Continue Training)

1. ✅ Wall ratio increasing toward 0.32
2. ✅ Validity >95% and stable
3. ✅ Solvability >95% and stable
4. ✅ Reward positive and increasing
5. ✅ Episode length decreasing (early termination working)
6. ✅ Explained variance >0.8
7. ✅ All metrics stabilizing after 1M+ steps

## Monitoring Schedule

- **First 100k steps**: Check if learning starts (reward increasing, validity >80%)
- **500k steps**: Should see wall ratio >0.25, validity >90%
- **1M steps**: Should see wall ratio >0.28, validity >95%, early termination working
- **2M steps**: Should see convergence (metrics stabilizing)

## Quick Check Command

On your server, you can quickly check metrics:
```bash
# Check latest metrics from monitor.csv
tail -20 runs/phase1_2obj_17x17_wall0.32/monitor.csv
```

## What to Do If Things Go Wrong

### Wall Ratio Too Low (<0.25)
1. Increase `wall_ratio_penalty` in `phase1_env.py` (try 40.0 or 50.0)
2. Or reduce `wall_target` to 0.30 (easier target)
3. Check if early termination is preventing wall placement

### Validity Too Low (<90%)
1. Check entity placement constraints (`entity_min_distance`)
2. May need to reduce constraints for 17×17 grid
3. Check if walls are overwriting entities (shouldn't happen with hard-block)

### Solvability Too Low (<90%)
1. Reduce `wall_target` to 0.30 (less walls = easier to maintain paths)
2. Check if per-step solvability rewards are working
3. May need to increase solvability bonus in rewards

### No Early Termination
1. Check early termination conditions in `phase1_env.py`
2. May need to relax wall ratio tolerance (±0.05)
3. Check if path length requirement is too strict

### Training Instability
1. Reduce learning rate (try 1e-4)
2. Increase batch size (try 4096)
3. Reduce clip range (try 0.1)
4. Check for reward scaling issues

