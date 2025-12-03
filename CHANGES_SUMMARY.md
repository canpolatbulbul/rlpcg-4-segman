# Implementation Summary: Movable-Critical Reward Logic

## What Was Changed

### 1. Core Environment (`grid_pcg_env.py`)

**Major Changes:**
- **Replaced** the old movable reward logic (on-path bonuses, obstruction checks) with **explicit movable-critical condition**
- **Added** strict/relaxed BFS checks that explicitly verify:
  - **Relaxed** (MOVABLE = empty): Puzzle must be solvable
  - **Strict** (MOVABLE = wall): Puzzle should be unsolvable
  - **Movable-critical**: Relaxed solvable AND strict unsolvable = ideal puzzle

**Reward Structure (Terminal):**
```
Base solvability (relaxed): +2.0 (essential)
Movable-critical bonus: +3.0 (dominant signal)
Penalty if both solvable: -2.0 (movables irrelevant)
Penalty if relaxed unsolvable: -3.0 (bad level)
Path length: +0.06 × (L1_relaxed + L2_relaxed)
Wall ratio: -5.0 × |deviation| (structural)
Corridor quality: +1.5 × adj_per_wall - 0.9 × iso_frac (structural)
```

**New Metrics in `info` dict:**
- `relaxed_solvable`: bool
- `strict_solvable`: bool
- `movable_critical`: bool
- `L1_relaxed`, `L2_relaxed`: path lengths (relaxed)
- `L1_strict`, `L2_strict`: path lengths (strict, or None)

**Comments:** Added comprehensive docstrings explaining the reward design and movable-critical logic.

---

### 2. Training Script (`train_ppo.py`)

**Added:**
- `MetricsCallback`: Aggregates and logs custom metrics to TensorBoard
  - Tracks movable-critical percentage
  - Tracks relaxed/strict solvability rates
  - Logs average path lengths, wall ratios, etc.
- Logs metrics every 500 episodes to TensorBoard under `metrics/` namespace

**TensorBoard Metrics:**
- `metrics/movable_critical_pct`: % of valid levels that are movable-critical
- `metrics/relaxed_solvable_pct`: % of levels that are relaxed-solvable
- `metrics/strict_solvable_pct`: % of levels that are strict-solvable
- `metrics/valid_pct`: % of levels that are valid
- `metrics/avg_l1_relaxed`, `metrics/avg_l2_relaxed`: Average path lengths
- `metrics/avg_wall_ratio`, `metrics/avg_n_movable`: Structural metrics

---

### 3. Evaluation Scripts

#### `eval_model.py`
**Enhanced to report:**
- Solvability statistics (valid, relaxed, strict, movable-critical percentages)
- Path length distributions (relaxed)
- Structural metrics (wall ratio, movable count, etc.)

#### `eval_and_select.py`
**Enhanced to:**
- Track and save movable-critical flag in CSV
- Include relaxed/strict solvability flags in CSV
- Include path lengths (both relaxed and strict) in CSV
- Print summary statistics at the end (if pandas is available)
- Optionally filter keepers by movable-critical condition (commented out, can be enabled)

---

## How to Use

### Training

```bash
python train_ppo.py \
  --size 13 --max_steps 192 --n_envs 8 \
  --total_timesteps 1_500_000 \
  --n_steps 1024 --batch_size 1024 \
  --lr 3e-4 --ent_coef 0.05 --use_curriculum \
  --logdir runs/ppo_grid_movable_critical
```

**Monitor Training:**
- Check TensorBoard: `tensorboard --logdir runs/ppo_grid_movable_critical`
- Look for `metrics/movable_critical_pct` - should increase over training
- Target: >50% of valid levels should be movable-critical by end of training

### Evaluation

**Quick evaluation:**
```bash
python eval_model.py \
  --model runs/ppo_grid_movable_critical/ppo_grid_movable.zip \
  --episodes 256 --max_steps 192
```

**Comprehensive evaluation with selection:**
```bash
python eval_and_select.py \
  --model runs/ppo_grid_movable_critical/ppo_grid_movable.zip \
  --episodes 2000 --max_steps 192 \
  --w_min 0.20 --w_max 0.30 \
  --movable_min 3 --movable_max 7 \
  --adj_min 0.15 --iso_max 0.30 \
  --min_Lsum 14 --min_L1 4 --min_L2 4 \
  --out_dir eval_movable_critical --use_curriculum
```

**To require movable-critical condition for keepers**, edit `eval_and_select.py` line ~163:
```python
keep = (
    (final_info["valid"] == 1) and
    final_info.get("movable_critical", False) and  # Add this line
    ...
)
```

---

## Expected Outcomes

### Before (Old Implementation)
- ~5-10% of valid levels are movable-critical
- Most movables are decorative (both strict and relaxed solvable)
- Agent doesn't learn to place movables in critical positions

### After (New Implementation)
- **Target: >50% of valid levels should be movable-critical**
- Movables are placed to block naive paths but puzzle remains solvable
- Generated grids resemble hand-crafted SeGMaN puzzles with pushable blocks

---

## Key Design Decisions

1. **Dominant Reward Signal**: Movable-critical condition (+3.0) is the primary reward for good puzzles, outweighing other shaping terms
2. **Explicit Checks**: Separate strict/relaxed BFS checks make the condition unambiguous
3. **Penalty for Non-Critical**: Strong penalty (-2.0) if both strict and relaxed are solvable (movables are irrelevant)
4. **Backward Compatibility**: Old metrics (`L1`, `L2`, `n_movable_on_path`) still present but deprecated

---

## Files Modified

1. `grid_pcg_env.py` - Core reward logic
2. `train_ppo.py` - Added metrics callback
3. `eval_model.py` - Enhanced reporting
4. `eval_and_select.py` - Enhanced CSV and summary
5. `ANALYSIS_AND_IMPROVEMENTS.md` - Analysis document (new)
6. `CHANGES_SUMMARY.md` - This file (new)

---

## Next Steps

1. **Train from scratch** with the new reward structure
2. **Monitor TensorBoard** to see if movable-critical percentage increases
3. **Evaluate** after training to verify >50% movable-critical rate
4. **Adjust hyperparameters** if needed (reward magnitudes, curriculum schedule)
5. **Compare** generated grids to hand-crafted SeGMaN puzzles

---

## Troubleshooting

**If movable-critical percentage stays low:**
- Check if relaxed solvability is high (good) but strict solvability is also high (bad)
- May need to increase movable-critical bonus (+3.0) or decrease penalty for both solvable (-2.0)
- Check curriculum schedule - movables are only introduced after progress > 0.5

**If training is unstable:**
- The reward magnitudes are tuned for stability, but you may need to adjust:
  - Reduce movable-critical bonus if reward variance is too high
  - Increase wall ratio penalty if structure is poor
  - Adjust entropy coefficient if exploration is insufficient

