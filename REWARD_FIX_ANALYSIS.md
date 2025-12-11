# Reward Structure Fix Analysis

## Problem Diagnosis

### Key Insight from Montage Analysis
The montage images revealed a critical insight: **The agent CAN generate high wall ratios (0.30-0.34)**, but TensorBoard shows an average of only 0.24. This discrepancy indicates:

1. **The agent has learned the capability** to reach higher wall ratios
2. **The problem is consistency**, not capability
3. **Many episodes fail** when attempting higher ratios, dragging down the average
4. **The reward structure is too risk-averse**, causing the agent to prefer the "safe" 0.24 strategy

### Root Cause
The agent learned a risk-averse policy:
- **Staying at 0.24**: Low risk, consistent rewards (~+14.0 per episode)
- **Trying to reach 0.32**: High risk, occasional success (+12.6) but frequent failures (-25.4)
- **Result**: Agent prefers the safe strategy, even though it can occasionally succeed

## Changes Implemented

### 1. Increased Per-Step Rewards (Line 473)
**Before**: `0.10 + 0.20 * min(1.0, target_gap / 0.2)` → **0.10 to 0.30 per wall**
**After**: `0.30 + 0.50 * min(1.0, target_gap / 0.2)` → **0.30 to 0.80 per wall**

**Rationale**: Make exploration more attractive by providing stronger incentives when below target. This should encourage the agent to push towards 0.32 instead of staying at 0.24.

### 2. Reduced Terminal Penalties (Lines 361, 527)
**Before**: `-30.0` for unsolvable/invalid grids
**After**: `-10.0` for unsolvable/invalid grids

**Rationale**: Reduce the catastrophic penalty that was preventing exploration. The agent was too afraid to try higher ratios because failures were too costly. With -10.0 penalty, failures are still punished but not so severely that exploration is discouraged.

### 3. Enhanced Metrics Logging (train_phase1.py)
**Added**: `metrics/avg_wall_ratio_valid_only` - Average wall ratio over **valid episodes only**

**Rationale**: The original `avg_wall_ratio` averages over ALL episodes (including failures with wall_ratio=0.0), which drags down the average. The new metric shows what the agent achieves when it succeeds, providing better insight into the agent's actual capability.

## Expected Outcomes

1. **Higher average wall ratio**: With stronger per-step rewards, the agent should be more motivated to push towards 0.32
2. **More consistent high ratios**: Reduced penalties allow more exploration, leading to more successful high-ratio episodes
3. **Better metrics visibility**: The `avg_wall_ratio_valid_only` metric will show the agent's true capability separate from failure rate

## Why This Should Work

The montage evidence shows the agent **can** reach 0.30-0.34 wall ratios. The problem was:
- **Weak incentives** to explore (low per-step rewards)
- **Strong disincentives** to explore (high failure penalties)
- **Result**: Agent learned to stay safe at 0.24

With the new reward structure:
- **Stronger incentives** (0.30-0.80 per wall) make exploration more attractive
- **Reduced disincentives** (-10.0 instead of -30.0) make failures less catastrophic
- **Result**: Agent should be more willing to explore and reach higher ratios consistently

## Next Steps

1. **Retrain** with the new reward structure
2. **Monitor** both `avg_wall_ratio` (all episodes) and `avg_wall_ratio_valid_only` (valid only)
3. **Check** if the valid-only average is higher than the overall average (indicating the agent can do better but fails often)
4. **If still stuck**: Consider curriculum learning (start with target=0.20, gradually increase to 0.32)

## Additional Considerations

### Is 0.32 Physically Achievable?
With 2 objects in a 17×17 grid (289 cells), 0.32 wall ratio means ~93 walls. The montage shows grids with 0.30-0.34 ratios that are solvable, so **0.32 is definitely achievable**.

### Early Termination Range
Current range: `[target - 0.05, target + 0.05]` = `[0.27, 0.37]` for target 0.32
- Agent at 0.24 is below this range, so early termination never triggers
- This is fine - the agent should learn to reach the target range first
- Once it consistently reaches 0.27-0.37, early termination will kick in

### Curriculum Learning Alternative
If the fixes don't work, consider:
1. Train with `wall_target=0.20` until agent reaches ~0.18-0.20 consistently
2. Increase to `wall_target=0.25` and retrain
3. Gradually increase to `wall_target=0.32`
This would help the agent learn incrementally rather than trying to jump from 0.24 to 0.32.
