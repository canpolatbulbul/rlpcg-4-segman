# Two-Phase Training Guide

## Overview

This project now uses a **two-phase training approach** that mirrors how humans build puzzle levels:

1. **Phase 1**: Learn to place walls and create well-structured, solvable puzzles
2. **Phase 2**: Learn to add movable obstacles to make puzzles challenging (movable-critical)

## Why Two Phases?

- **Simpler learning**: Each phase focuses on one task
- **Better credit assignment**: Agent gets clearer feedback
- **No catastrophic forgetting**: Phase 1 skills remain intact
- **Smaller action spaces**: Phase 1 uses 2 actions (WALL/EMPTY), Phase 2 uses 2 actions (MOVABLE/EMPTY)

## Phase 1: Structure Learning

### Environment (`phase1_env.py`)
- **Entities**: Pre-placed randomly at reset with constraints (min distance, border margin)
- **Actions**: Only WALL and EMPTY (no entity placement, no MOVABLE)
- **Goal**: Create well-structured, solvable puzzles with ~25% wall density
- **Early termination**: Allowed after 50 steps if quality thresholds met

### Training
```bash
python train_phase1.py \
  --size 13 --max_steps 150 --n_envs 8 \
  --total_timesteps 1_000_000 \
  --wall_target 0.25 \
  --logdir runs/phase1_structure
```

### Evaluation
```bash
python eval_phase1.py \
  --model runs/phase1_structure/phase1_final.zip \
  --episodes 256
```

### Success Criteria
- >90% valid levels (all entities present)
- >85% solvable (relaxed solvability)
- Wall ratio ~0.25
- Good corridor structure (low isolated walls)

## Phase 2: Challenge Learning

### Environment (`phase2_env.py`)
- **Base puzzles**: Generated using trained Phase 1 model at each reset
- **Frozen cells**: Walls and entities cannot be modified
- **Actions**: Only MOVABLE and EMPTY
- **Goal**: Place movables to achieve movable-critical condition
- **Early termination**: Allowed after 30 steps if movable-critical achieved

### Training
```bash
python train_phase2.py \
  --phase1_model runs/phase1_structure/phase1_final.zip \
  --size 13 --max_steps 80 --n_envs 8 \
  --total_timesteps 1_000_000 \
  --logdir runs/phase2_challenge
```

### Evaluation
```bash
python eval_phase2.py \
  --model runs/phase2_challenge/phase2_final.zip \
  --phase1_model runs/phase1_structure/phase1_final.zip \
  --episodes 256
```

### Success Criteria
- >50% movable-critical (relaxed solvable AND strict unsolvable)
- >90% relaxed solvable
- 3-7 movables per level on average

## Workflow

1. **Train Phase 1** until satisfied with structure quality
2. **Evaluate Phase 1** to verify it generates good base puzzles
3. **Train Phase 2** using Phase 1 model
4. **Evaluate Phase 2** to check movable-critical percentage
5. **Use Phase 2 model** for final puzzle generation

## Key Features

### Phase 1
- Random entity placement with constraints
- Hard-blocking: Cannot place walls on entities
- Early termination with quality checks
- Focused reward on structure quality

### Phase 2
- Dynamic base puzzle generation (fresh each episode)
- Frozen walls/entities (agent can't break structure)
- Dominant reward signal for movable-critical condition
- Early termination when goal achieved

## Files

- `phase1_env.py`: Phase 1 environment
- `phase2_env.py`: Phase 2 environment (uses Phase 1 model)
- `train_phase1.py`: Phase 1 training script
- `train_phase2.py`: Phase 2 training script
- `eval_phase1.py`: Phase 1 evaluation script
- `eval_phase2.py`: Phase 2 evaluation script

## Notes

- Phase 1 model is frozen during Phase 2 training
- Each Phase 2 episode gets a fresh base puzzle from Phase 1
- Phase 1 can use deterministic or stochastic generation for Phase 2
- Both phases support early termination to encourage efficiency

