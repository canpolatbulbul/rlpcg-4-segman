# PCGRL for SeGMaN: Analysis and Improvements

## Executive Summary

This document analyzes the current PCGRL setup for generating puzzle grids for SeGMaN (a robotic motion planner) and proposes concrete improvements to ensure the RL agent generates **challenging but solvable** puzzles with **critical movable obstacles**.

---

## 1. Current Environment Definition

### State Space
- **Observation**: `(H, W, 6)` one-hot encoded planes for tile types: `[EMPTY, WALL, ROBOT, OBJECT, GOAL, MOVABLE]`
- **Grid size**: Default 13×13 (configurable)

### Action Space
- **Discrete**: `H × W × 6` actions
- Each action selects a cell `(y, x)` and a tile type to place
- **Unique entity semantics**: Placing ROBOT/OBJECT/GOAL moves (or creates) that entity
- WALL and MOVABLE cannot overwrite entities

### Episode Structure
- Episodes last exactly `max_steps` (default 192) - no SUBMIT action
- Terminal reward computed at episode end
- Small per-step shaping rewards guide learning

---

## 2. Intended Behavior

The agent should generate grids that:

1. **Are valid**: Exactly 1 ROBOT, 1 OBJECT, 1 GOAL
2. **Are solvable** (relaxed): When MOVABLE obstacles are treated as pushable (like EMPTY), there exist paths:
   - ROBOT → OBJECT
   - OBJECT → GOAL
3. **Are challenging** (strict): When MOVABLE obstacles are treated as solid (like WALL), at least one path is blocked
4. **Have structure**: Walls form corridors, not blobs; appropriate wall density (~25%)
5. **Use MOVABLEs critically**: MOVABLE obstacles must be "critical blockers" - they block the naive shortest path but the puzzle remains solvable if they can be pushed

---

## 3. Current Reward Structure Analysis

### Current Reward Components (from `_evaluate_grid()`)

| Component | Weight/Value | Purpose | Issue |
|-----------|-------------|---------|-------|
| Base validity | -1.0 if invalid | Ensure 3 entities exist | ✅ Works |
| Path length (L1+L2) | +0.06 × (L1+L2) | Encourage non-trivial paths | ⚠️ Uses static path (ignores movables) |
| Wall ratio | -5.0 × |deviation| | Force ~25% walls | ✅ Works |
| Corridor quality | +1.5 × adj_per_wall - 0.9 × iso_frac | Encourage corridors | ✅ Works |
| Movable quantity | +0.5 if in range | Target ~3% density | ⚠️ Doesn't ensure criticality |
| Movable on-path | +2.0 × n_on_path | Reward movables on static path | ⚠️ Doesn't check if they're critical |
| Movable off-path | -0.5 × n_off_path | Penalize irrelevant movables | ⚠️ Too weak |
| Obstruction bonus | +1.0 if blocked > static | Reward path obstruction | ❌ **Doesn't ensure strict unsolvable** |
| Adjacency penalty | -2.0 × adj_trivial | Avoid trivial puzzles | ✅ Works |
| Border penalty | -0.5 × border_pen | Keep entities away from edges | ✅ Works |

### Critical Issues Identified

1. **Missing "Movable-Critical" Check**: 
   - Current code checks if `path_len_blocked > path_len_static`, but this doesn't guarantee that strict pathfinding is **unsolvable**
   - A puzzle where both strict and relaxed are solvable (but paths are longer) gets a small bonus, but movables aren't truly critical

2. **Reward Magnitude Imbalance**:
   - Movable-critical condition should be a **dominant** reward signal (similar to validity)
   - Current obstruction bonus (+1.0) is too small compared to wall ratio penalties (-5.0 × deviation)
   - The agent can satisfy wall ratio and ignore movable-criticality

3. **No Penalty for Non-Critical Movables**:
   - If both strict and relaxed are solvable, movables are decorative, not functional
   - Should penalize this case strongly

4. **Path Length Reward Uses Static Path**:
   - Current: `path_len_static = L1_static + L2_static` (treats movables as empty)
   - This is correct for encouraging solvability, but we need to separately reward the challenge aspect

---

## 4. Proposed Improvements

### 4.1 Explicit Movable-Critical Reward Logic

**Definition**: A puzzle is "movable-critical" if:
- **Relaxed** (MOVABLE = empty): Both paths exist (ROBOT→OBJECT, OBJECT→GOAL)
- **Strict** (MOVABLE = wall): At least one path is broken

**Implementation**:
```python
# Check relaxed solvability
L1_relaxed = shortest_path_len(grid, ROBOT, OBJECT, treat_movable_as_empty=True)
L2_relaxed = shortest_path_len(grid, OBJECT, GOAL, treat_movable_as_empty=True)
relaxed_solvable = (L1_relaxed is not None) and (L2_relaxed is not None)

# Check strict solvability  
L1_strict = shortest_path_len(grid, ROBOT, OBJECT, treat_movable_as_empty=False)
L2_strict = shortest_path_len(grid, OBJECT, GOAL, treat_movable_as_empty=False)
strict_solvable = (L1_strict is not None) and (L2_strict is not None)

# Movable-critical condition
movable_critical = relaxed_solvable and (not strict_solvable)
```

**Reward Structure**:
- **Base reward for relaxed solvability**: +2.0 (essential)
- **Movable-critical bonus**: +3.0 (dominant signal)
- **Penalty if both solvable**: -2.0 (movables irrelevant)
- **Penalty if relaxed unsolvable**: -3.0 (bad level)

### 4.2 Reward Rebalancing

Keep existing structural rewards but adjust magnitudes:
- Wall ratio: Keep strong (-5.0 × deviation) - structure is important
- Corridor quality: Keep as is
- Path length: Use relaxed path for base reward (encourages solvability)
- Movable-critical: Make it the **primary** movable reward (replaces current on-path/obstruction logic)

### 4.3 Enhanced Logging

Add metrics to `info` dict:
- `relaxed_solvable`: bool
- `strict_solvable`: bool  
- `movable_critical`: bool
- `L1_relaxed`, `L2_relaxed`: path lengths (relaxed)
- `L1_strict`, `L2_strict`: path lengths (strict, or None)

### 4.4 Evaluation Script Updates

Update `eval_model.py` and `eval_and_select.py` to:
- Report % of levels that are movable-critical
- Report % that are relaxed-solvable
- Report distributions of strict vs relaxed path lengths
- Filter keepers by movable-critical flag

---

## 5. Implementation Plan

1. ✅ Update `_evaluate_grid()` with explicit strict/relaxed checks
2. ✅ Implement movable-critical reward logic
3. ✅ Rebalance reward coefficients
4. ✅ Add comprehensive comments
5. ✅ Update training script logging
6. ✅ Enhance evaluation scripts

---

## 6. Expected Outcomes

After these changes:

- **Training**: Agent should learn to place MOVABLE obstacles in critical positions
- **Evaluation**: >50% of generated levels should be movable-critical (vs current ~5-10%)
- **Quality**: Generated grids should resemble hand-crafted SeGMaN puzzles with pushable blocks creating interesting pathfinding challenges

---

## 7. Notes on PCGRL Paper Alignment

The PCGRL paper emphasizes:
- **Playability constraints**: Levels must be solvable
- **Diversity**: Reward for varied structures
- **Challenge**: Reward for non-trivial paths

Our implementation aligns with this by:
- Enforcing relaxed solvability (playability)
- Rewarding movable-critical condition (challenge)
- Using wall ratio and corridor quality (structure/diversity)

The key addition is the **explicit movable-critical constraint**, which is specific to SeGMaN's pushable obstacle semantics.

