# Merge Summary: Supervisor's SeGMaN Integration

This document summarizes the merge of the supervisor's SeGMaN integration code into the 2-phase training system.

## What Was Merged

### 1. SeGMaN Motion Planner (`MOSeGMan/`)
- **`MOSeGMan.py`**: Multi-Objective Selective Guided Manipulation class
- **`SeGManv2.py`**: Selective Guided Manipulation v2 implementation
- These files provide the motion planning functionality that operates on `.g` config files.

### 2. Base Configuration Files (`ry_config/`)
- **`base.g`**: Base rai configuration file
- **`base-aux.g`**: Auxiliary rai configuration file
- These serve as templates for generating new environments.

### 3. Updated Generation Script (`rai_env_gen.py`)
- **Original**: Used single-phase `GridPCGEnv` with old model
- **Updated**: Now uses 2-phase system:
  - Phase 1: Generates base puzzle (walls + entities)
  - Phase 2: Adds movable obstacles
  - Converts final grid (including MOVABLE tiles) to `.g` files

### 4. Testing Script (`segman_test.py`)
- Tests SeGMaN motion planner on generated `.g` files
- Tracks robot movement and saves to CSV
- Updated to accept `--case_id` argument

## Key Changes

### `rai_env_gen.py` Updates

1. **Two-Phase Generation**:
   - Loads both Phase 1 and Phase 2 models
   - Generates base puzzle with Phase 1
   - Adds movables with Phase 2
   - Combines results into final grid

2. **MOVABLE Tile Support**:
   - Added handling for `MOVABLE` tiles in `grid_to_rai_config()`
   - Movables are converted to `movable_o` objects in the `.g` file
   - Yellow color coding for movables

3. **Improved Error Handling**:
   - Skips invalid puzzles from either phase
   - Reports metrics for each phase
   - Tracks valid sample count

## Usage

### Generate Environments

```bash
python rai_env_gen.py \
  --phase1_model <path_to_phase1_model.zip> \
  --phase2_model <path_to_phase2_model.zip> \
  --n 16 \
  --size 13 \
  --stochastic
```

**Arguments**:
- `--phase1_model`: Path to trained Phase 1 model (required)
- `--phase2_model`: Path to trained Phase 2 model (required)
- `--n`: Number of samples to generate (default: 16)
- `--size`: Grid size (default: 13)
- `--phase1_max_steps`: Max steps for Phase 1 (default: 200)
- `--phase2_max_steps`: Max steps for Phase 2 (default: 40)
- `--wall_target`: Target wall ratio for Phase 1 (default: 0.35)
- `--deterministic`: Use greedy actions (default: False)
- `--stochastic`: Use stochastic sampling (overrides --deterministic)
- `--phase1_deterministic`: Use Phase 1 deterministically when generating base puzzles

**Output**:
- Creates `ry_config/case_run_id_<ID>/` directory
- Each sample saved as `pcg-<i>/pcg-<i>.g` and `pcg-<i>/pcg-<i>-aux.g`

### Test SeGMaN on Generated Files

```bash
python segman_test.py --case_id <case_id>
```

**Arguments**:
- `--case_id`: Case ID to test (default: 79)

**Output**:
- Creates `data/tracks_id_<case_id>.csv` with robot movement tracks

## Tile Mapping

| Grid Tile | .g File Object | Color | Logical Attribute |
|-----------|----------------|-------|-------------------|
| `EMPTY` | (none) | - | - |
| `WALL` | `block_<r>_<c>` | Brown (0.6953, 0.515625, 0.453125) | - |
| `ROBOT` | `ego` | - | `agent` |
| `OBJECT` | `obj1` | Blue | `movable_go` |
| `GOAL` | `goal1` | Blue (transparent) | `goal` |
| `MOVABLE` | `movable_<n>` | Yellow | `movable_o` |

## Integration Notes

- The 2-phase system ensures base puzzles are always valid and solvable before adding movables
- Movable obstacles are properly integrated into the SeGMaN planning system
- The conversion maintains spatial relationships and tile semantics
- Both deterministic and stochastic generation modes are supported

## Next Steps

1. Train Phase 1 and Phase 2 models if not already done
2. Generate test environments using `rai_env_gen.py`
3. Test motion planning using `segman_test.py`
4. Analyze results and tune models as needed

