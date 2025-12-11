# Phase 2 Analysis: n-Object Support & Movable-Critical Logic

## Issues Found

### 🐛 BUG #1: `_generate_base_puzzle()` Only Checks for 1 Object

**Location**: `phase2_env.py` line 149

**Problem**:
```python
if n_robot == 1 and n_object == 1 and n_goal == 1:  # ❌ Only checks for 1 object!
```

**Should be**:
```python
if n_robot == 1 and n_object == self.n_objects and n_goal == self.n_objects:
```

**Impact**: For n_objects=2, this will reject valid base puzzles that have 2 objects and 2 goals!

### ✅ n-Object Support: Mostly Correct

The rest of the code correctly handles n_objects:
- ✅ `_valid_final()` checks `self._count(OBJECT) == self.n_objects`
- ✅ Object-goal pairing logic handles multiple pairs correctly
- ✅ Path checking loops through all pairs
- ✅ Path length averaging divides by `self.n_objects`

## Movable-Critical Logic Analysis

### How It Works Currently

**Terminal Reward (Episode End)**:
1. **Checks ALL object-goal pairs** for:
   - Relaxed solvability (with movables treated as empty)
   - Strict solvability (with movables blocking)

2. **Movable-critical condition**:
   ```python
   movable_critical = relaxed_solvable AND at_least_one_strict_unsolvable
   ```
   - ALL pairs must be relaxed-solvable
   - AT LEAST ONE pair must be strict-unsolvable

3. **Reward**:
   - If movable_critical: **+10.0** (single bonus, NOT per-path)
   - If not movable_critical: **-5.0** (single penalty)
   - **NOT scaled by number of blocked paths**

### Per-Step Shaping Reward

**Location**: `phase2_env.py` lines 403-459

**Problem**: Only checks the **first object-goal pair**:
```python
oy, ox = object_positions[0]  # ❌ Only first pair!
```

**Current behavior**:
- If movable blocks R→O1 or O1→G1: +0.3 per blocked path
- If movable blocks R→O2 or O2→G2: **NOT CHECKED** (ignored!)

**Impact**: Agent gets no per-step reward for blocking paths of object 2!

## Answer to Your Question

**Q: "Does it check how many paths the movable blocks? Like if it blocks object1→goal1 path + object2→goal2 path at the same time, it gets twice the reward?"**

**A: NO, it does NOT work that way currently:**

1. **Terminal reward**: Single +10.0 bonus if ANY path is blocked (not scaled)
2. **Per-step reward**: Only checks first object-goal pair (ignores others)

## Recommended Fixes

### Fix #1: Correct Base Puzzle Validation
```python
# Line 149
if n_robot == 1 and n_object == self.n_objects and n_goal == self.n_objects:
```

### Fix #2: Check ALL Pairs in Per-Step Shaping
Instead of only checking `object_positions[0]`, loop through ALL pairs and sum rewards:
```python
# For each object-goal pair:
for (oy, ox), (gy, gx) in object_goal_pairs:
    # Check if movable blocks this pair's paths
    # Reward for each blocked path
```

### Fix #3 (Optional): Scale Terminal Reward by Number of Blocked Paths
```python
# Count how many pairs are strict-unsolvable
n_blocked_pairs = sum(1 for ... if strict_unsolvable)
R = self.movable_critical_bonus * (1.0 + 0.2 * (n_blocked_pairs - 1))  # Bonus for blocking multiple
```
