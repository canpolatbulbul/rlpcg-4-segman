# Training Analysis: wall_target=0.40 (No Early Stopping)

## Key Observations

### ✅ Early Stopping Removal Worked
- **ep_len_mean**: Consistently 250 (max_steps) throughout training
- Episodes now run to completion, giving agent full opportunity to explore

### ⚠️ The "Start High, Drop Later" Pattern Persists
- **avg_wall_ratio_valid_only**: 
  - Starts at ~0.32 (good!)
  - Drops sharply to ~0.28 at 400k-600k steps
  - Ends at ~0.27-0.26 (well below target of 0.40)
- **Same pattern as v6/v7**: Agent initially explores, then learns to avoid higher ratios

### ✅ Valid Percentage Improving
- **valid_pct / solvable_pct**: 55% → 86% (excellent improvement!)
- Agent is learning to generate more valid grids

### ❌ But Wall Ratio in Valid Grids is Decreasing
- Agent learns: "Lower wall ratios = higher probability of being valid"
- This is a **reward structure problem**, not an exploration problem

## Root Cause: Wall Ratio Penalty is Too Strong

### The Reward Math Problem

**Current reward structure:**
- Wall ratio penalty: `-100.0 * abs(wr - wall_target)`
- Solvability bonus: `+6.0` (for 2 objects)
- Unsolvable penalty: `-10.0`

**Example calculations for target 0.40:**

1. **Agent at 0.27 (current behavior):**
   - Wall penalty: `-100.0 * 0.13 = -13.0`
   - Solvability bonus: `+6.0`
   - Net: `-7.0` from wall ratio deviation
   - But grid is valid, so total reward is positive

2. **Agent at 0.40 (target):**
   - Wall penalty: `-100.0 * 0.0 = 0.0`
   - Solvability bonus: `+6.0`
   - Net: `+6.0` from wall ratio (perfect!)
   - But risk: if it breaks solvability, gets `-10.0` penalty

3. **Agent tries 0.40 but breaks solvability:**
   - Wall penalty: `0.0` (at target)
   - Unsolvable penalty: `-10.0`
   - Net: `-10.0` (catastrophic)

**The agent learns:**
- "Staying at 0.27 = safe, consistent rewards"
- "Trying to reach 0.40 = risky, might get -10.0 penalty"
- "The -13.0 wall penalty at 0.27 is acceptable compared to the risk of -10.0 unsolvable penalty"

### The Problem

The **wall ratio penalty (100.0) is too strong** relative to:
1. The solvability bonus (+6.0)
2. The unsolvable penalty (-10.0)

The agent optimizes for **validity** (which it achieves: 86% valid) at the cost of **wall ratio** (which decreases over time).

## Solutions

### Option 1: Reduce Wall Ratio Penalty (Recommended)
**Change**: `wall_ratio_penalty = 100.0` → `30.0` or `50.0`

**Rationale**: 
- Makes the penalty proportional to other rewards
- At 0.27 vs 0.40: penalty would be `-30.0 * 0.13 = -3.9` instead of `-13.0`
- This makes reaching the target more attractive relative to staying safe

### Option 2: Increase Solvability Bonus
**Change**: `solvability_bonus = 3.0 * n_objects` → `5.0 * n_objects` or `10.0 * n_objects`

**Rationale**:
- Makes valid grids more valuable
- But doesn't directly incentivize higher wall ratios

### Option 3: Make Wall Ratio Penalty Asymmetric
**Change**: Penalize being below target MORE than being above target

**Rationale**:
- Agent is consistently below target
- Asymmetric penalty would push agent toward target

### Option 4: Add Bonus for Reaching Target
**Change**: Add `+20.0` bonus if `abs(wr - target) < 0.05`

**Rationale**:
- Strong incentive to reach the target range
- Outweighs the risk of breaking solvability

## Recommended Fix

**Reduce wall_ratio_penalty from 100.0 to 30.0-50.0**

This will:
1. Make the penalty proportional to other rewards
2. Make reaching the target more attractive
3. Still penalize deviation, but not so severely that agent avoids exploration

## Current State Summary

- ✅ Early stopping removed (working as intended)
- ✅ Valid percentage improving (86%)
- ❌ Wall ratio decreasing over time (0.32 → 0.26)
- ❌ Agent learning "safe" strategy at lower ratios
- **Root cause**: Wall ratio penalty too strong relative to other rewards
