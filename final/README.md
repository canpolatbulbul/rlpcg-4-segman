# RL-PCG Environment Generation - Lab Directory

This directory contains the trained models and scripts needed to generate procedural content generation (PCG) environments for robotic planning tasks.

## Contents

### Models
- `phase1_model.zip` - Phase 1 trained model (generates base puzzles with walls + entities)
- `phase2_model.zip` - Phase 2 trained model (adds movable obstacles to create challenging puzzles)

### Scripts
- `montage_phase1.py` - Visualize Phase 1 model outputs (grid montages)
- `montage_phase2_critical_only.py` - Visualize Phase 2 model outputs (only movable-critical puzzles)
- `rai_env_gen.py` - Generate environments and convert to rai Config (.g) files
- `segman_test.py` - Test generated environments with MOSeGMan planner

### Dependencies
- `phase1_env.py` - Phase 1 environment definition
- `phase2_env.py` - Phase 2 environment definition
- `grid_pcg_env.py` - Shared utilities (pathfinding, etc.)
- `small_cnn.py` - CNN policy architecture
- `MOSeGMan/` - MOSeGMan planner (required for segman_test.py)
- `ry_config/` - Base rai configuration files (base.g, base-aux.g)

## Quick Start
Models were trained with size 17, n_objects 2, wall_target 0.40, so these scripts are intended to be used with those args, otherwise unexpected outcomes may occur.
### 1. Generate Environments (rai Config files)

```bash
python rai_env_gen.py \
  --phase1_model phase1_model.zip \
  --phase2_model phase2_model.zip \
  --n 16 \
  --size 17 \
  --n_objects 2 \
  --wall_target 0.40 \
  --stochastic
```

**Arguments:**
- `--phase1_model`: Path to Phase 1 model (required)
- `--phase2_model`: Path to Phase 2 model (required)
- `--n`: Number of environments to generate (default: 16)
- `--size`: Grid size (13 for single-object, 17 for 2-object) (default: 13)
- `--n_objects`: Number of objects/goals (1 or 2) (default: 1)
- `--wall_target`: Wall ratio target (should match Phase 1 training) (default: 0.35)
- `--stochastic`: Use stochastic sampling (recommended, default)
- `--deterministic`: Use greedy actions (often produces empty grids, not recommended)

**Output:**
- Creates `ry_config/case_run_id_XXXX/` directory
- Each environment saved as `pcg-N/pcg-N.g` and `pcg-N-aux.g`

### 2. Visualize Phase 1 Outputs

```bash
python montage_phase1.py \
  --model phase1_model.zip \
  --n 16 \
  --size 17 \
  --n_objects 2 \
  --wall_target 0.40 \
  --out montage_phase1.png \
  --stochastic
```

**Optional but Strongly Recommended:**
- `--valid_only`: Only show valid (solvable) grids

### 3. Visualize Phase 2 Outputs (Critical Only)

```bash
python montage_phase2_critical_only.py \
  --model phase2_model.zip \
  --phase1_model phase1_model.zip \
  --n 16 \
  --size 17 \
  --n_objects 2 \
  --out montage_phase2_critical.png \
  --stochastic
```

**Note:** This script keeps sampling until it finds N movable-critical grids. May take time if critical rate is low.

### 4. Test Generated Environments with MOSeGMan

```bash
python segman_test.py --case_id <case_id>
```

**Arguments:**
- `--case_id`: Case ID from rai_env_gen output (the number in `case_run_id_XXXX`)

**Output:**
- Creates `data/tracks_id_<case_id>.csv` with robot trajectories

## Model Specifications

### Phase 1 Model
- **Grid size**: 17×17 (for 2-object mode)
- **Wall target**: 0.40
- **Objects**: 2 objects, 2 goals
- **Action space**: Place WALL or EMPTY
- **Output**: Base puzzle with walls and entities (no movables)

### Phase 2 Model
- **Grid size**: 17×17 (matches Phase 1)
- **Objects**: 2 objects, 2 goals (matches Phase 1)
- **Action space**: Place MOVABLE or EMPTY
- **Input**: Base puzzle from Phase 1
- **Output**: Puzzle with movables (movable-critical condition)

## Environment Format

### Grid Tiles
- `EMPTY` (0): Empty space (light gray)
- `WALL` (1): Wall (brown)
- `ROBOT` (2): Robot start position (blue)
- `OBJECT` (3): Movable object to push (yellow/orange)
- `GOAL` (4): Goal position (magenta/pink)
- `MOVABLE` (5): Movable obstacle (green)

### Movable-Critical Condition
A puzzle is "movable-critical" if:
- **Relaxed solvable**: All paths (R→O1, O1→G1, R→O2, O2→G2) are solvable when movables are treated as pushable
- **Strict unsolvable**: At least one path is unsolvable when movables are treated as walls (blocking)

This means the puzzle requires pushing movables to solve.

## File Structure

```
lab_directory/
├── phase1_model.zip
├── phase2_model.zip
├── montage_phase1.py
├── montage_phase2_critical_only.py
├── rai_env_gen.py
├── segman_test.py
├── phase1_env.py
├── phase2_env.py
├── grid_pcg_env.py
├── small_cnn.py
├── MOSeGMan/
│   ├── MOSeGMan.py
│   └── SeGManv2.py
├── ry_config/
│   ├── base.g
│   └── base-aux.g
└── README.md (this file)
```

## Requirements (you can use requirements.txt)

- Python 3.8+
- `stable-baselines3` (for loading models)
- `numpy`, `matplotlib` (for visualization)
- `robotic` (rai library, for rai_env_gen and segman_test)
- `gymnasium` (for environments)

Install dependencies:
```bash
pip install stable-baselines3 numpy matplotlib gymnasium
# Plus robotic library (rai) - follow your lab's installation instructions
```

## Troubleshooting

### Empty or Invalid Grids
- **Solution**: Use `--stochastic` flag (default). Deterministic mode often produces empty grids.

### Low Critical Rate in Phase 2
- **Normal**: Movable-critical puzzles are harder to generate. The script will retry automatically.
- **Solution**: Increase `--max_attempts` in montage script, or generate more samples.

### Size Mismatch Errors
- **Ensure**: `--size` and `--n_objects` match the training configuration
- **For 2-object mode**: Use `--size 17 --n_objects 2`

### Model Loading Errors
- **Check**: Model paths are correct and files exist
- **Ensure**: Models were trained with matching `n_objects` configuration

## Notes

- Models are trained for **2-object mode** (n_objects=2)
- Grid size **17×17** is used for 2-object mode (auto-adjusted from base size 13)
- **Stochastic sampling** is recommended for better diversity
- **Wall target 0.40** was used during Phase 1 training
- Generated environments are saved in `ry_config/case_run_id_XXXX/` directories

## Contact

For questions or issues, refer to the main project documentation or contact the development team.
