# 2-Phase Training System Summary

## Overview

We moved from a single-phase training system to a **2-phase curriculum learning approach** that mirrors how humans design puzzle levels:

1. **Phase 1: Structure Learning** - Learn to place walls to create well-structured, solvable puzzles
2. **Phase 2: Challenge Learning** - Learn to add movable obstacles to make puzzles challenging (movable-critical)

This separation makes training more stable and allows the agent to focus on one skill at a time.

---

## Phase 1: Structure Learning (`phase1_env.py`)

### Goal
Train an agent to place **walls** to create structured, solvable puzzles with appropriate wall density.

### Key Features

**Environment Setup:**
- Grid size: 13×13 (default)
- Max steps: 200
- Min steps before early termination: 50
- **Entities (ROBOT, OBJECT, GOAL) are randomly placed at reset** with constraints:
  - Minimum 5-block distance between entities
  - No border margin (entities can be at edges)
  - Agent cannot modify entities (hard-blocked)

**Action Space:**
- Agent can only place `EMPTY` or `WALL`
- Cannot place entities or movables
- Actions that would overwrite entities are blocked

**Reward Structure:**
- **Validity**: Must have exactly 1 robot, 1 object, 1 goal
- **Solvability**: Paths R→O and O→G must exist
- **Path length**: Encourages non-trivial paths (`alpha * (L1 + L2)`)
- **Wall ratio**: Strong penalty for deviation from target (default 0.35)
- **Corridor quality**: Rewards structured walls, penalizes isolated walls and 2×2 blocks
- **Early termination bonus**: Scaled bonus for terminating early when quality thresholds are met

**Early Termination:**
- Allowed after `min_steps` if:
  - Grid is valid
  - Solvable
  - Wall ratio within ±0.05 of target
  - Path lengths are reasonable

**Training Script:** `train_phase1.py`

**Evaluation Script:** `eval_phase1.py`

**Montage Script:** `montage_phase1.py`

---

## Phase 2: Challenge Learning (`phase2_env.py`)

### Goal
Train an agent to place **movable obstacles** on base puzzles to make them challenging (movable-critical).

### Key Features

**Environment Setup:**
- Grid size: 13×13 (default)
- Max steps: 40 (reduced from 80 - only need to place 5-6 movables)
- Min steps before early termination: 30
- **Base puzzle generated using trained Phase 1 model** at each reset
- Retries up to 10 times to ensure valid, solvable base puzzle

**Action Space:**
- Agent can only place `EMPTY` or `MOVABLE`
- **Walls and entities are frozen** (cannot be modified)
- Actions on frozen cells are blocked

**Reward Structure:**
- **Movable-critical bonus**: +10.0 (dominant reward)
  - Puzzle must be solvable in relaxed mode (movables treated as empty)
  - Puzzle must be unsolvable in strict mode (movables treated as walls)
- **Penalties**:
  - Both solvable: -5.0 (movables irrelevant)
  - Relaxed unsolvable: -3.0 (bad level)
- **Movable count penalty**: Strong linear penalty for excess movables (target: 5-6)
- **Per-step shaping**: Immediate rewards for placing movables that block paths
- **Early termination bonus**: For achieving movable-critical condition

**Movable-Critical Condition:**
- **Strict mode**: Treat movables as walls → puzzle should be unsolvable
- **Relaxed mode**: Treat movables as empty → puzzle should be solvable
- A good puzzle requires the robot to push movables out of the way

**Training Script:** `train_phase2.py`

**Evaluation Script:** `eval_phase2.py`

**Montage Scripts:**
- `montage_phase2.py` - All generated grids
- `montage_phase2_critical_only.py` - Only movable-critical grids

---

## Training Workflow

### Step 1: Train Phase 1
```bash
python train_phase1.py \
  --size 13 \
  --max_steps 200 \
  --wall_target 0.35 \
  --total_timesteps 1_500_000 \
  --logdir runs/phase1_v1
```

**What to look for:**
- Wall ratio approaching target (e.g., 0.35)
- High validity rate (>95%)
- High solvability rate (>95%)
- Structured wall patterns in montages

**Common Issues & Fixes:**
- **Wall ratio too low**: Increase `wall_ratio_penalty` (currently 20.0)
- **Wall ratio stuck**: Check early termination bounds match `wall_target`
- **No structure**: Check corridor quality rewards

### Step 2: Train Phase 2
```bash
python train_phase2.py \
  --phase1_model runs/phase1_v1/ppo_phase1_final.zip \
  --size 13 \
  --max_steps 40 \
  --total_timesteps 1_000_000 \
  --logdir runs/phase2_v1
```

**What to look for:**
- Movable-critical percentage increasing (>70% target)
- Average movable count around 5-6
- Strict solvable percentage decreasing (movables blocking effectively)

**Common Issues & Fixes:**
- **Too many movables**: Increase `movable_count_penalty_coeff` (currently 1.5)
- **Low movable-critical**: Check base puzzle quality (may need better Phase 1 model)
- **Both solvable**: Increase per-step shaping rewards for blocking paths

---

## Key Design Decisions

### Why Two Phases?

1. **Credit Assignment**: Easier to learn "place walls" vs "place walls AND movables"
2. **Stability**: Phase 1 creates consistent base puzzles for Phase 2
3. **Curriculum**: Learn structure first, then challenge
4. **Mirrors Human Design**: Humans design structure first, then add obstacles

### Why Random Entity Placement?

- Entity placement is not the learning goal
- Random placement provides diversity
- Agent learns to work with any entity configuration
- Constraints ensure reasonable separation

### Why Early Termination?

- Efficiency: Don't waste steps when quality is achieved
- Signal: Agent learns when to stop
- Scaled bonus: Rewards being closer to exact target

### Why Frozen Cells in Phase 2?

- Prevents agent from breaking base puzzle
- Focuses learning on movable placement
- Ensures base puzzle quality is maintained

---

## File Structure

```
rlpcg-4-segman/
├── phase1_env.py              # Phase 1 environment
├── phase2_env.py              # Phase 2 environment
├── train_phase1.py            # Phase 1 training script
├── train_phase2.py            # Phase 2 training script
├── eval_phase1.py             # Phase 1 evaluation
├── eval_phase2.py             # Phase 2 evaluation
├── montage_phase1.py          # Phase 1 visualization
├── montage_phase2.py          # Phase 2 visualization
├── montage_phase2_critical_only.py  # Phase 2 critical-only montage
├── grid_pcg_env.py            # Shared utilities (BFS, tile IDs)
└── small_cnn.py               # CNN feature extractor (shared)
```

---

## Current Status & Known Issues

### Phase 1
- ✅ Random entity placement working
- ✅ Early termination working
- ✅ Wall ratio tracking target (with tuning)
- ⚠️ Wall ratio may still be slightly below target (0.27-0.30 vs 0.35)
  - Can increase `wall_ratio_penalty` further if needed

### Phase 2
- ✅ Movable-critical condition implemented
- ✅ Base puzzle generation with retry logic
- ✅ Movable count control (target 5-6)
- ⚠️ Movable-critical percentage can be low if base puzzles have too few walls
  - Solution: Ensure Phase 1 model has good wall ratio

---

## Next Steps

1. **Train Phase 1** until satisfied with wall ratio and structure
2. **Train Phase 2** using the Phase 1 model
3. **Evaluate** using `eval_phase1.py` and `eval_phase2.py`
4. **Generate montages** to visually inspect quality
5. **Generate .g files** using `rai_env_gen.py` for SeGMaN testing
6. **Test with SeGMaN** using `segman_test.py`

---

## Quick Reference: Key Hyperparameters

### Phase 1
- `wall_target`: 0.35 (target wall ratio)
- `wall_ratio_penalty`: 20.0 (penalty for deviation)
- `max_steps`: 200
- `min_steps`: 50

### Phase 2
- `movable_critical_bonus`: 10.0
- `movable_count_penalty_coeff`: 1.5
- `max_steps`: 40
- `min_steps`: 30

---

## Tips for Training

1. **Start with Phase 1**: Get good base puzzles first
2. **Monitor TensorBoard**: Watch wall ratio, validity, solvability
3. **Check montages**: Visual inspection is crucial
4. **Tune gradually**: Don't change too many things at once
5. **Save checkpoints**: Keep intermediate models for comparison

