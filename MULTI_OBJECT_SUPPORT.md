# Multi-Object Support (MO-SeGMaN Integration)

## Overview

The system now supports **multiple objects and goals**, enabling integration with MO-SeGMaN (Multi-Objective Selective Guided Manipulation). This allows generating puzzles with N objects and N goals (one goal per object), matching the supervisor's MO-SeGMaN planner requirements.

## Key Changes

### 1. **Environment Updates**

#### Phase1Env (`phase1_env.py`)
- Added `n_objects` parameter (default: 1 for backward compatibility)
- **Auto-adjusts grid size**: When `n_objects > 1`, automatically uses 17×17 grid (instead of 13×13)
- **Entity placement**: Places N objects and N goals randomly with constraints
- **Validation**: Checks for exactly N objects and N goals
- **Solvability**: Verifies paths for all object-goal pairs (greedy pairing by proximity)
- **Rewards**: Scales solvability bonus by number of objects

#### Phase2Env (`phase2_env.py`)
- Added `n_objects` parameter (must match Phase 1 model)
- **Base puzzle generation**: Uses Phase 1 model with matching `n_objects`
- **Validation**: Checks for N objects and N goals
- **Movable-critical**: All pairs must be relaxed solvable, at least one pair must be strict unsolvable
- **Per-step shaping**: Simplified checks for efficiency (full check at episode end)

### 2. **Visualization Updates**

All montage scripts now support multi-object visualization:
- **`montage_phase1.py`**: Distinct colors per object-goal pair
- **`montage_phase2.py`**: Distinct colors per object-goal pair
- **`montage_phase2_critical_only.py`**: Distinct colors per object-goal pair

**Color Scheme** (for up to 5 objects):
- Object 1: Orange/Yellow → Goal 1: Magenta
- Object 2: Green → Goal 2: Cyan
- Object 3: Red → Goal 3: Pink
- Object 4: Blue → Goal 4: Purple
- Object 5: Brown → Goal 5: Rose

### 3. **Generation Updates**

#### `rai_env_gen.py`
- Added `n_objects` parameter
- **Object-goal pairing**: Pairs objects with goals using greedy closest-distance matching
- **.g file generation**: Creates `obj1`, `obj2`, ..., `objN` and `goal1`, `goal2`, ..., `goalN` in rai config
- All objects have `movable_go` logical attribute (for MO-SeGMaN)

### 4. **Training Scripts**

#### `train_phase1.py` & `train_phase2.py`
- Added `--n_objects` argument (default: 1)
- Passes `n_objects` to environment factories

## Usage

### Training Phase 1 (Multi-Object)

```bash
python train_phase1.py \
  --size 17 \
  --n_objects 2 \
  --wall_target 0.35 \
  --total_timesteps 1_500_000 \
  --logdir runs/phase1_multiobj_2
```

**Note**: Grid size auto-adjusts to 17×17 when `n_objects > 1`, but you can override with `--size`.

### Training Phase 2 (Multi-Object)

```bash
python train_phase2.py \
  --phase1_model runs/phase1_multiobj_2/phase1_final.zip \
  --n_objects 2 \
  --size 17 \
  --total_timesteps 1_000_000 \
  --logdir runs/phase2_multiobj_2
```

**Important**: `n_objects` must match the Phase 1 model!

### Generating .g Files (Multi-Object)

```bash
python rai_env_gen.py \
  --phase1_model runs/phase1_multiobj_2/phase1_final.zip \
  --phase2_model runs/phase2_multiobj_2/phase2_final.zip \
  --n_objects 2 \
  --n 16 \
  --stochastic
```

### Visualization

```bash
# Phase 1 montage
python montage_phase1.py \
  --model runs/phase1_multiobj_2/phase1_final.zip \
  --n_objects 2 \
  --n 16 \
  --stochastic

# Phase 2 montage (all grids)
python montage_phase2.py \
  --model runs/phase2_multiobj_2/phase2_final.zip \
  --phase1_model runs/phase1_multiobj_2/phase1_final.zip \
  --n_objects 2 \
  --n 16 \
  --stochastic

# Phase 2 montage (critical only)
python montage_phase2_critical_only.py \
  --model runs/phase2_multiobj_2/phase2_final.zip \
  --phase1_model runs/phase1_multiobj_2/phase1_final.zip \
  --n_objects 2 \
  --n 16 \
  --stochastic
```

## Grid Size Recommendations

- **1 object**: 13×13 (default)
- **2-3 objects**: 17×17 (auto-adjusted when `n_objects > 1`)
- **4-5 objects**: 19×19 or 21×21 (manually specify with `--size`)
- **6+ objects**: 21×21 or larger

## Object-Goal Pairing

The system uses **greedy closest-distance pairing**:
1. For each object, find the closest unused goal
2. Pair them together
3. Repeat until all objects are paired

This matches MO-SeGMaN's approach where each object has a corresponding goal position.

## Solvability Checks

### Phase 1
- Checks paths: R→O₁, O₁→G₁, R→O₂, O₂→G₂, ... for all pairs
- All paths must exist for puzzle to be valid

### Phase 2
- **Relaxed mode**: All object-goal pairs must be solvable (movables treated as empty)
- **Strict mode**: At least one pair must be unsolvable (movables treated as walls)
- **Movable-critical**: All relaxed solvable AND at least one strict unsolvable

## Backward Compatibility

- **Default `n_objects=1`**: All existing code works without changes
- **Single-object mode**: Original behavior preserved
- **Multi-object mode**: Activated when `n_objects > 1`

## Testing with MO-SeGMaN

The generated `.g` files are compatible with MO-SeGMaN:
- Objects are named `obj1`, `obj2`, ..., `objN` with `movable_go` logical attribute
- Goals are named `goal1`, `goal2`, ..., `goalN` with `goal` logical attribute
- Movable obstacles are named `movable_1`, `movable_2`, ... with `movable_o` logical attribute

Test with:
```bash
python segman_test.py --case_id <case_id>
```

## Notes

- **Grid size**: Multi-object puzzles need more space. 17×17 is recommended for 2-3 objects.
- **Training time**: Multi-object puzzles are more complex, may need more training steps
- **Movable count**: Target remains 5-6 movables regardless of number of objects
- **Pairing**: Object-goal pairing is deterministic (closest distance), ensuring consistency

