# Reward Tuning Notes

## Current Evaluation Results (After Initial Implementation)

- **Movable-critical: 14.3%** (target: >50%)
- **Strict solvable: 77.0%** (too high - movables not blocking)
- **Relaxed solvable: 90.6%** (good)

## Problem Analysis

The agent is placing movables (avg 8.6 per level) but they're not critical blockers. The reward structure needs adjustment.

## Reward Changes Made

### Version 1 (Initial):
- Movable-critical: +3.0
- Both solvable (non-critical): -2.0
- **Gap: 5.0**

### Version 2 (Current - after tuning):
- Movable-critical: **+5.0** (increased from +3.0)
- Both solvable (non-critical): **-4.0** (increased from -2.0)
- **Gap: 9.0** (much larger signal)

## Rationale

1. **Larger gap** between critical and non-critical makes the signal clearer
2. **Stronger penalty** for non-critical movables discourages decorative placement
3. **Stronger bonus** for critical movables makes it more rewarding to learn the pattern

## Expected Impact

With these changes, the agent should:
- Learn to place movables in positions that actually block paths
- Avoid placing movables where they don't affect solvability
- Increase movable-critical percentage toward >50%

## Next Steps

1. Retrain with new reward structure
2. Monitor TensorBoard for `metrics/movable_critical_pct`
3. Re-evaluate after training
4. If still low (<30%), consider:
   - Further increasing the gap (e.g., +6.0 / -5.0)
   - Adding per-step penalties for placing movables off-path
   - Adjusting curriculum schedule for movables

