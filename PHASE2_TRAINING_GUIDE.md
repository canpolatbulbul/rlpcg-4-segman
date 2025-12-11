# Phase 2 Training Guide

## Overview

Phase 2 builds on Phase 1 by:
1. Using Phase 1 model to generate base puzzles (walls + entities)
2. Learning to place MOVABLE obstacles to create "movable-critical" puzzles
3. Movable-critical: Puzzle is solvable with movables, but unsolvable without them

## Requirements

- **Phase 1 model**: Trained Phase 1 model (v10 best_model)
- **n_objects**: Must match Phase 1 model (2 for your case)
- **size**: Must match Phase 1 model (17 for 2-object mode)

## Training Command

```bash
python train_phase2.py \
  --phase1_model <path_to_v10_best_model> \
  --n_objects 2 \
  --size 13 \
  --max_steps 80 \
  --total_timesteps 1_500_000 \
  --logdir runs/phase2_2obj_v10_base
```

## Finding Your v10 Model

The v10 model should be in a run directory. Common locations:
- `runs/phase1_2obj_v10/best_model.zip`
- `runs/phase1_2obj_wall040_*/best_model.zip`
- Or wherever you saved the v10 training run

## Key Parameters

- `--phase1_model`: **REQUIRED** - Path to your v10 Phase 1 model
- `--n_objects 2`: Must match Phase 1 (2 objects)
- `--size 13`: Base size (auto-adjusts to 17 for 2 objects, matching Phase 1)
- `--phase1_deterministic`: Optional - Use Phase 1 deterministically (default: stochastic)
- `--max_steps 80`: Steps for placing movables (shorter than Phase 1's 250)

## What Phase 2 Does

1. **Reset**: Phase 1 model generates a base puzzle (walls + entities)
2. **Agent actions**: Place MOVABLE or EMPTY only (walls/entities frozen)
3. **Goal**: Create movable-critical puzzles
4. **Reward**: 
   - High reward for movable-critical condition
   - Solvability bonuses
   - Path length rewards

## Expected Metrics

Monitor in TensorBoard:
- `metrics/movable_critical_pct`: Percentage of puzzles that are movable-critical
- `metrics/relaxed_solvable_pct`: Puzzles solvable with movables
- `metrics/strict_solvable_pct`: Puzzles solvable without movables
- `metrics/avg_n_movable`: Average number of movables placed

## Notes

- Phase 1 model is loaded once and used to generate base puzzles
- Phase 2 agent only learns to place movables, not walls
- This two-phase approach allows specialization: Phase 1 = structure, Phase 2 = challenge
