# Analysis: Training with Reduced Wall Penalty (30.0) - Still Not Working

## Results Summary

### Metrics
- **avg_wall_ratio_valid_only**: 0.325 → 0.305 (still decreasing)
- **valid_pct**: ~63.6% (decent but not great)
- **ep_rew_mean**: ~81 (stable, no improvement)
- **ep_len_mean**: 250 (early stopping removed - working ✓)

### The Problem Persists
Even with reduced wall penalty (30.0), the agent still learns to avoid higher ratios. The "start high, drop later" pattern continues.

## Root Cause: Statistical Reality vs Reward Structure

The agent is learning a **statistical truth**: Lower wall ratios ARE more likely to be valid.

**The Reward Math:**
- Per-step rewards (0.30-0.80 per wall) over 250 steps = **+75 to +200 total**
- At 0.30 vs target 0.40: terminal penalty = `-30.0 * 0.10 = -3.0`
- If agent tries 0.40 and breaks solvability: terminal penalty = `-10.0`

**Agent learns:**
- "Stay at 0.30: +75-200 per-step, -3.0 terminal = **net +72 to +197**"
- "Try 0.40: risk -10.0 terminal = **net negative**"
- "Better to stay safe at 0.30"

## The Real Issue

The reward structure **penalizes deviation** but doesn't **reward reaching the target** enough. We need to make reaching the target MORE valuable than staying safe.

## Solutions

### Option 1: Large Bonus for Reaching Target (Recommended)
Add a large bonus when wall ratio is within target range:
```python
if abs(wr - self.wall_target) < 0.05:
    R += 20.0  # Large bonus for reaching target
```

**Rationale**: Makes reaching target highly valuable, outweighing the risk

### Option 2: Asymmetric Penalty
Penalize being below target MORE than being above:
```python
if wr < self.wall_target:
    wall_term = -self.wall_ratio_penalty * (wall_dev * 1.5)  # 1.5x penalty for being below
else:
    wall_term = -self.wall_ratio_penalty * wall_dev  # Normal penalty for being above
```

**Rationale**: Pushes agent toward target more aggressively

### Option 3: Increase Per-Step Rewards When Far Below Target
Make per-step rewards even stronger when far below:
```python
if wr < self.wall_target:
    target_gap = self.wall_target - wr
    # Increase from 0.30-0.80 to 0.50-1.20 per wall
    reward += 0.50 + 0.70 * min(1.0, target_gap / 0.2)
```

**Rationale**: Stronger incentive to place walls when far below target

### Option 4: Reduce Unsolvable Penalty Further
Change from -10.0 to -5.0 to make exploration less risky:
```python
return -5.0, metrics  # Reduced from -10.0
```

**Rationale**: Makes failures less catastrophic, encourages exploration

## Recommended Fix: Combine Options 1 + 4

1. **Add large bonus for reaching target** (+20.0)
2. **Reduce unsolvable penalty** (-10.0 → -5.0)

This will:
- Make reaching target highly valuable
- Make failures less catastrophic
- Encourage exploration toward target
