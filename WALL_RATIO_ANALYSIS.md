# Wall Ratio Analysis for 2-Object 17×17 Grid

## Grid Space Analysis

### Grid Dimensions
- **Total cells**: 17 × 17 = **289 cells**
- **Entities**: 1 robot + 2 objects + 2 goals = **5 cells**
- **Available for walls/empty/movables**: **284 cells**

### Comparison with Single-Object 13×13
- **Single-object**: 13×13 = 169 cells, 3 entities = 166 available
- **2-object**: 17×17 = 289 cells, 5 entities = 284 available
- **Scale factor**: ~1.71× more cells, but only 1.67× more entities
- **Relative space**: Slightly more space per entity in 2-object mode

## Wall Ratio Options

### Option 1: 0.30 Wall Ratio
- **Walls**: ~87 cells (30% of 289)
- **Empty**: ~197 cells (after walls and entities)
- **Pros**:
  - More flexibility for maintaining 4 paths (R→O1, O1→G1, R→O2, O2→G2)
  - Easier to place 5-6 movables strategically
  - Less risk of over-constraining paths
- **Cons**:
  - Might be too sparse, less structured
  - Agent might struggle to create interesting corridors

### Option 2: 0.32 Wall Ratio (RECOMMENDED)
- **Walls**: ~93 cells (32% of 289)
- **Empty**: ~191 cells
- **Pros**:
  - **Balanced**: Good structure while maintaining flexibility
  - Enough walls for interesting corridors
  - Enough open space for 5-6 movables
  - Paths remain maintainable with 4 object-goal pairs
- **Cons**:
  - Slightly tighter than 0.30, but still manageable

### Option 3: 0.35 Wall Ratio
- **Walls**: ~101 cells (35% of 289)
- **Empty**: ~183 cells
- **Pros**:
  - More structured, denser corridors
  - Closer to single-object ratio (0.35)
- **Cons**:
  - **Risky**: With 4 paths to maintain, might be too dense
  - Less flexibility for movable placement
  - Agent might struggle to maintain all paths
  - Phase 2 might have difficulty placing movables strategically

## Path Complexity Analysis

### Single-Object (13×13)
- **Paths to maintain**: 2 (R→O, O→G)
- **Path cells**: ~20-30 cells typically
- **Available for walls**: ~136 cells (after entities and paths)
- **Wall ratio achieved**: ~0.20-0.27 (below 0.35 target)

### 2-Object (17×17)
- **Paths to maintain**: 4 (R→O1, O1→G1, R→O2, O2→G2)
- **Path cells**: ~40-60 cells typically (2× more paths)
- **Available for walls**: ~224 cells (after entities and paths)
- **Wall ratio needed**: Lower than single-object due to more paths

## Phase 2 Considerations

### Movable Placement Requirements
- **Target**: 5-6 movables
- **Purpose**: Strategically block paths in strict mode
- **Constraint**: Must maintain relaxed solvability

### Space Requirements
- **With 0.30 wall ratio**: ~197 empty cells → Plenty of space for 5-6 movables
- **With 0.32 wall ratio**: ~191 empty cells → Good space for 5-6 movables
- **With 0.35 wall ratio**: ~183 empty cells → Tighter, but still feasible

### Strategic Placement
- Movables need to be placed in "choke points" between paths
- With 4 paths, there are more potential choke points
- But paths might be more spread out, requiring more movables to block effectively
- **Conclusion**: 0.30-0.32 provides better flexibility for strategic placement

## Recommendation: 0.32 Wall Ratio

### Justification
1. **Balanced structure**: Enough walls for interesting corridors (93 walls)
2. **Path flexibility**: 191 empty cells provide flexibility for 4 paths
3. **Movable space**: Sufficient space for 5-6 strategic movables
4. **Learning stability**: Not too sparse (agent learns structure) or too dense (maintains paths)
5. **Phase 2 compatibility**: Good balance for movable-critical condition

### Alternative: 0.30 if Struggling
- If agent struggles to maintain all 4 paths with 0.32
- If Phase 2 has difficulty achieving movable-critical
- Start with 0.32, reduce to 0.30 if needed

## Training Recommendations

### Phase 1 Training
```bash
# Recommended configuration
--size 17
--n_objects 2
--wall_target 0.32
--max_steps 250  # More steps for larger grid
--total_timesteps 2_000_000  # More training for complex task
```

### Phase 2 Training
- Use same `wall_target` (0.32) when generating base puzzles
- Movable count target remains 5-6 (regardless of number of objects)
- Movable-critical condition: All 4 paths relaxed solvable, at least one strict unsolvable

## Expected Outcomes

### Phase 1 (wall_target=0.32)
- **Wall ratio achieved**: ~0.28-0.32 (may be slightly below target, as with single-object)
- **Validity**: >95%
- **Solvability**: >95% (all 4 paths)
- **Path lengths**: L1+L2 total ~40-60 (sum across both pairs)

### Phase 2
- **Movable-critical**: >70% target
- **Movable count**: 5-6 average
- **Strict solvable**: <30% (movables blocking effectively)

## Monitoring

### Key Metrics to Watch
1. **Wall ratio**: Should approach 0.32 (may stabilize at 0.28-0.30)
2. **Validity**: Should be >95%
3. **Solvability**: All 4 paths should be solvable >95% of the time
4. **Path lengths**: Should be non-trivial (not all adjacent)
5. **Early termination**: Should occur when quality thresholds met

### Warning Signs
- Wall ratio stuck at <0.25: Increase `wall_ratio_penalty` or reduce `wall_target`
- Solvability <90%: Too many walls, reduce `wall_target` to 0.30
- Paths too short: Increase path length requirements
- No early termination: Check early termination conditions

