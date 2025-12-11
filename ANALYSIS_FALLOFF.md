# Analysis: "Start Good, Then Fall Off" Pattern

## Observations

### Metrics Behavior
- **avg_wall_ratio_valid_only**: 
  - Starts at ~0.32
  - **Peaks at ~0.345-0.35** around 200k-250k steps (good!)
  - **Drops to ~0.295-0.30** at 400k-500k steps (fall off)
  - Stabilizes around 0.31

- **valid_pct**: 
  - Peaks at ~62-63% around 800k-1M steps
  - Declines to ~60% by end

- **ep_rew_mean**: 
  - Starts high (peaking near 87)
  - Declines to ~83-84 by end

## Root Cause: Target Bonus Too Narrow

The +20.0 target bonus only triggers when `abs(wr - target) < 0.05`.

**For target 0.40:**
- Bonus triggers: [0.35, 0.45]
- Agent peaks at 0.345-0.35 (just at the edge!)
- But then learns: "Hitting this narrow range is hard, and if I miss, I get penalized"
- Agent learns to stay at 0.30-0.31 (safer, more consistent)

## The Problem

The agent initially explores and finds the target range, but then learns:
1. "Hitting [0.35, 0.45] is difficult and inconsistent"
2. "Staying at 0.30-0.31 gives me consistent rewards without risk"
3. "The wall penalty at 0.30 vs 0.40 is only -3.0, which is acceptable"

## Solutions

### Option 1: Wider Target Bonus Range (Recommended)
Change from `±0.05` to `±0.10`:
```python
if abs(wr - self.wall_target) < 0.10:  # Wider range
    target_bonus = 20.0
```

**Rationale**: Makes it easier to hit the bonus, more consistent rewards

### Option 2: Gradient Bonus (Better)
Instead of all-or-nothing, give scaled bonus based on distance:
```python
# Bonus scales from +20.0 at target to 0 at ±0.10 deviation
wall_dev = abs(wr - self.wall_target)
if wall_dev < 0.10:
    target_bonus = 20.0 * (1.0 - wall_dev / 0.10)  # Linear scaling
else:
    target_bonus = 0.0
```

**Rationale**: Rewards getting closer to target, not just hitting exact range

### Option 3: Asymmetric Target Bonus
Reward being above target more than being below:
```python
if wr >= self.wall_target - 0.05:  # At or above target
    target_bonus = 20.0
elif wr >= self.wall_target - 0.10:  # Close below target
    target_bonus = 10.0
```

**Rationale**: Since agent is consistently below target, reward getting closer

### Option 4: Reduce Wall Penalty Further
Change from 30.0 to 20.0:
```python
self.wall_ratio_penalty = 20.0  # Further reduced
```

**Rationale**: Makes deviation less costly, encourages exploration

## Recommended Fix: Option 2 (Gradient Bonus)

Replace the all-or-nothing bonus with a gradient that rewards getting closer to target. This will:
1. Reward progress toward target, not just hitting exact range
2. Make rewards more consistent and less "risky"
3. Encourage continuous improvement rather than binary success/failure
