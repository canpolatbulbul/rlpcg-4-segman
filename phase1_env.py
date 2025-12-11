# phase1_env.py
# Phase 1: Structure Learning Environment
# Agent learns to place walls to create good puzzle structure
# Entities (ROBOT, OBJECT, GOAL) are pre-placed randomly at reset

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque

# Import shared utilities from grid_pcg_env
from grid_pcg_env import (
    EMPTY, WALL, ROBOT, OBJECT, GOAL, MOVABLE,
    shortest_path_len
)

# Phase 1 only uses EMPTY and WALL
PHASE1_TILES = (EMPTY, WALL)
PHASE1_N_TILES = len(PHASE1_TILES)


class Phase1Env(gym.Env):
    """
    Phase 1 Environment: Structure Learning
    
    - Entities (ROBOT, OBJECT, GOAL) are randomly placed at reset with constraints
    - Agent can only place WALL or EMPTY (no entity placement, no MOVABLE)
    - Goal: Learn to create well-structured, solvable puzzles with appropriate wall density
    - Early termination allowed after min_steps if quality thresholds are met
    """
    
    metadata = {"render_modes": []}
    
    def __init__(
        self,
        size: int = 13,
        max_steps: int = 200,  # Increased from 150 to give more time for wall placement
        min_steps: int = 50,  # Minimum steps before early termination allowed
        seed: int | None = None,
        wall_target: float = 0.35,
        entity_min_distance: int = 5,  # Minimum distance between entities
        entity_border_margin: int = 0,  # Entities can be placed at edges (0 = no margin)
        n_objects: int = 1,  # Number of objects (and goals). 1 = single-object mode, >1 = multi-object mode
    ):
        super().__init__()
        self.h = int(size)
        self.w = int(size)
        self.size = int(size)
        self.max_steps = int(max_steps)
        self.min_steps = int(min_steps)
        self.wall_target = float(wall_target)
        self.entity_min_distance = int(entity_min_distance)
        self.entity_border_margin = int(entity_border_margin)
        self.n_objects = int(n_objects)
        
        # Auto-adjust size for multi-object mode
        if self.n_objects > 1 and size < 17:
            # Multi-object needs more space
            self.h = 17
            self.w = 17
            self.size = 17
        
        self.steps = 0
        self.rng = np.random.RandomState(seed if seed is not None else 42)
        
        # Reward hyperparameters
        self.alpha = 0.06       # Path length weight
        self.gamma = 2.0        # Trivial adjacency penalty
        self.delta = 0.5        # Border penalty weight
        self.lambda_corridor_term = 1.5    # Corridor quality
        self.lambda_isolated_term = 0.9    # Isolated walls penalty
        self.lambda_block_term = 0.3       # 2x2 block penalty
        self.wall_ratio_penalty = 100.0    # Wall ratio deviation penalty (very strong to force agent toward target)
        
        # Early termination bonus
        self.early_term_bonus = 0.2
        
        # Grid
        self.grid = np.full((self.h, self.w), EMPTY, dtype=np.int32)
        
        # Observation: H x W x 6 (one-hot for all tile types, even though we only use EMPTY/WALL)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.h, self.w, 6), dtype=np.float32
        )
        # Action: Discrete(H * W * 2) for (y, x, tile_type) where tile_type ∈ {EMPTY, WALL}
        self.action_space = spaces.Discrete(self.h * self.w * PHASE1_N_TILES)
        
        # Track entity positions (frozen after reset)
        self.entity_positions = set()
    
    def _obs(self) -> np.ndarray:
        """Generate one-hot observation planes."""
        planes = np.zeros((self.h, self.w, 6), dtype=np.float32)
        for t in [EMPTY, WALL, ROBOT, OBJECT, GOAL, MOVABLE]:
            planes[:, :, t] = (self.grid == t)
        return planes
    
    def _count(self, t: int) -> int:
        return int(np.sum(self.grid == t))
    
    def _pos(self, t: int):
        """Get position of a tile. For entities that can appear multiple times, returns first occurrence."""
        ys, xs = np.where(self.grid == t)
        if ys.size == 0:
            return None
        return int(ys[0]), int(xs[0])
    
    def _all_positions(self, t: int):
        """Get all positions of a tile type. Returns list of (y, x) tuples."""
        ys, xs = np.where(self.grid == t)
        return [(int(y), int(x)) for y, x in zip(ys, xs)]
    
    def _manhattan_dist(self, pos1, pos2):
        y1, x1 = pos1
        y2, x2 = pos2
        return abs(y1 - y2) + abs(x1 - x2)
    
    def _place_entities_randomly(self):
        """
        Randomly place ROBOT, N OBJECTs, N GOALs with constraints:
        - Minimum distance between any two entities
        - Can be placed at edges (no border margin restriction)
        - For multi-object mode: places n_objects OBJECTs and n_objects GOALs
        """
        self.entity_positions.clear()
        
        # Build entity list: 1 robot + n_objects objects + n_objects goals
        entities = [ROBOT]
        for _ in range(self.n_objects):
            entities.append(OBJECT)
        for _ in range(self.n_objects):
            entities.append(GOAL)
        
        self.rng.shuffle(entities)
        
        placed = []
        margin = self.entity_border_margin  # Now 0, so entities can be anywhere
        
        for entity in entities:
            attempts = 0
            while attempts < 1000:  # Safety limit
                # Allow placement anywhere in grid (margin=0 means full range)
                y = self.rng.randint(0, self.h)
                x = self.rng.randint(0, self.w)
                
                # Check minimum distance from already placed entities
                too_close = False
                for py, px in placed:
                    if self._manhattan_dist((y, x), (py, px)) < self.entity_min_distance:
                        too_close = True
                        break
                
                if not too_close:
                    self.grid[y, x] = entity
                    self.entity_positions.add((y, x))
                    placed.append((y, x))
                    break
                
                attempts += 1
            
            if attempts >= 1000:
                # Fallback: place anywhere if we can't find a good spot
                y = self.rng.randint(0, self.h)
                x = self.rng.randint(0, self.w)
                self.grid[y, x] = entity
                self.entity_positions.add((y, x))
                placed.append((y, x))
    
    def _valid_final(self) -> bool:
        """Check if all entities are present."""
        return (self._count(ROBOT) == 1 and
                self._count(OBJECT) == self.n_objects and
                self._count(GOAL) == self.n_objects)
    
    def _border_penalty(self, margin: int = 1) -> float:
        """Penalty for entities near borders."""
        pen = 0.0
        H, W = self.h, self.w
        for t in (ROBOT, OBJECT, GOAL):
            p = self._pos(t)
            if p is None:
                continue
            y, x = p
            if y <= margin or y >= H - 1 - margin or x <= margin or x >= W - 1 - margin:
                pen += 1.0
        return pen
    
    def _wall_stats(self):
        """Compute wall statistics."""
        n_walls = int(np.sum(self.grid == WALL))
        if n_walls == 0:
            return dict(
                n_walls=0, ratio=0.0,
                n_isolated=0, n_adj_pairs=0, n_2x2=0
            )
        
        g_walls = (self.grid == WALL).astype(np.int32)
        
        # Adjacency
        up = g_walls[:-1, :] * g_walls[1:, :]
        left = g_walls[:, :-1] * g_walls[:, 1:]
        n_adj_pairs = int(up.sum() + left.sum())
        
        # Degree (neighbors)
        deg = np.zeros_like(g_walls, dtype=np.int32)
        deg[1:,  :] += g_walls[:-1, :]  # Top neighbor
        deg[:-1, :] += g_walls[1:,  :]  # Bottom neighbor
        deg[:, 1:]  += g_walls[:, :-1]  # Left neighbor
        deg[:, :-1] += g_walls[:, 1:]   # Right neighbor (BUG FIX: was [:, :-1])
        n_isolated = int(((g_walls == 1) & (deg == 0)).sum())
        
        # 2x2 blocks
        n_2x2 = int((g_walls[:-1, :-1] * g_walls[1:, :-1] * 
                     g_walls[:-1, 1:] * g_walls[1:, 1:]).sum())
        
        ratio = n_walls / float(self.h * self.w)
        
        return dict(
            n_walls=n_walls, ratio=ratio,
            n_isolated=n_isolated, n_adj_pairs=n_adj_pairs, n_2x2=n_2x2
        )
    
    def _can_early_terminate(self) -> bool:
        """Check if early termination conditions are met."""
        if self.steps < self.min_steps:
            return False
        
        if not self._valid_final():
            return False
        
        # Check relaxed solvability (no movables, so just walls)
        robot_pos = self._pos(ROBOT)
        if robot_pos is None:
            return False
        ry, rx = robot_pos
        
        object_positions = self._all_positions(OBJECT)
        goal_positions = self._all_positions(GOAL)
        
        if len(object_positions) != self.n_objects or len(goal_positions) != self.n_objects:
            return False
        
        # Pair objects with goals (same logic as _evaluate_grid)
        used_goals = set()
        object_goal_pairs = []
        for oy, ox in object_positions:
            best_goal = None
            best_dist = float('inf')
            for gy, gx in goal_positions:
                if (gy, gx) in used_goals:
                    continue
                dist = self._manhattan_dist((oy, ox), (gy, gx))
                if dist < best_dist:
                    best_dist = dist
                    best_goal = (gy, gx)
            if best_goal is None:
                return False
            object_goal_pairs.append(((oy, ox), best_goal))
            used_goals.add(best_goal)
        
        # Check all paths are solvable
        for (oy, ox), (gy, gx) in object_goal_pairs:
            L1 = shortest_path_len(self.grid, (ry, rx), (oy, ox), treat_movable_as_empty=True)
            L2 = shortest_path_len(self.grid, (oy, ox), (gy, gx), treat_movable_as_empty=True)
            if L1 is None or L2 is None:
                return False
        
        # Check wall ratio (use dynamic target with tolerance)
        ws = self._wall_stats()
        wr = ws["ratio"]
        # Allow termination if within ±0.05 of target
        target_lo = max(0.0, self.wall_target - 0.05)
        target_hi = min(1.0, self.wall_target + 0.05)
        if not (target_lo <= wr <= target_hi):
            return False
        
        # Check path length (non-trivial) - sum across all pairs
        total_path_length = 0
        for (oy, ox), (gy, gx) in object_goal_pairs:
            L1 = shortest_path_len(self.grid, (ry, rx), (oy, ox), treat_movable_as_empty=True)
            L2 = shortest_path_len(self.grid, (oy, ox), (gy, gx), treat_movable_as_empty=True)
            total_path_length += (L1 + L2)
        
        # Minimum path length scales with number of objects
        min_path_length = 10 * self.n_objects
        if total_path_length < min_path_length:
            return False
        
        return True
    
    def _evaluate_grid(self):
        """Evaluate final grid and compute terminal reward."""
        metrics = {
            "valid": 0, "L1": 0, "L2": 0,
            "wall_ratio": 0.0, "adj_per_wall": 0.0, "iso_frac": 0.0,
        }
        
        if not self._valid_final():
            return -1.0, metrics
        
        # Get robot position
        robot_pos = self._pos(ROBOT)
        if robot_pos is None:
            return -1.0, metrics
        ry, rx = robot_pos
        
        # Get all object and goal positions
        object_positions = self._all_positions(OBJECT)
        goal_positions = self._all_positions(GOAL)
        
        if len(object_positions) != self.n_objects or len(goal_positions) != self.n_objects:
            return -1.0, metrics
        
        # Pair objects with goals (greedy: pair each object with closest goal)
        # This matches MO-SeGMaN's approach where each object has a corresponding goal
        used_goals = set()
        object_goal_pairs = []
        total_path_length = 0
        
        for oy, ox in object_positions:
            # Find closest unused goal
            best_goal = None
            best_dist = float('inf')
            for gy, gx in goal_positions:
                if (gy, gx) in used_goals:
                    continue
                dist = self._manhattan_dist((oy, ox), (gy, gx))
                if dist < best_dist:
                    best_dist = dist
                    best_goal = (gy, gx)
            
            if best_goal is None:
                return -1.0, metrics  # Should not happen if counts match
            
            object_goal_pairs.append(((oy, ox), best_goal))
            used_goals.add(best_goal)
        
        # Check solvability for all pairs: R→O_i and O_i→G_i for each pair
        all_paths_valid = True
        L1_total = 0  # Sum of all R→O path lengths
        L2_total = 0  # Sum of all O→G path lengths
        
        for (oy, ox), (gy, gx) in object_goal_pairs:
            # Path from robot to object
            L1 = shortest_path_len(self.grid, (ry, rx), (oy, ox), treat_movable_as_empty=True)
            # Path from object to goal
            L2 = shortest_path_len(self.grid, (oy, ox), (gy, gx), treat_movable_as_empty=True)
            
            if L1 is None or L2 is None:
                all_paths_valid = False
                break
            
            L1_total += L1
            L2_total += L2
        
        if not all_paths_valid:
            # STRONG penalty for unsolvable grids
            # Must outweigh per-step rewards (~0.20 * n_walls = ~20 for 100 walls)
            return -30.0, metrics
        
        # Average path lengths for metrics (backward compatibility)
        L1_avg = L1_total / self.n_objects
        L2_avg = L2_total / self.n_objects
        
        # Trivial adjacency (check all object-goal pairs)
        adj_trivial = 0.0
        for (oy, ox), (gy, gx) in object_goal_pairs:
            if max(abs(ry - oy), abs(rx - ox)) <= 1:
                adj_trivial = 1.0
            if max(abs(oy - gy), abs(ox - gx)) <= 1:
                adj_trivial = 1.0
        
        # Wall stats
        ws = self._wall_stats()
        wr = ws["ratio"]
        wall_dev = abs(wr - self.wall_target)
        
        # Wall ratio term
        wall_term = -self.wall_ratio_penalty * wall_dev
        
        # Corridor quality
        if ws["n_walls"] > 0:
            adj_per_wall = ws["n_adj_pairs"] / float(ws["n_walls"])
            iso_frac = ws["n_isolated"] / float(ws["n_walls"])
        else:
            adj_per_wall, iso_frac = 0.0, 0.0
        
        corridor_term = (
            + self.lambda_corridor_term * adj_per_wall
            - self.lambda_isolated_term * iso_frac
            - self.lambda_block_term * (ws["n_2x2"] / max(1, ws["n_walls"]))
        )
        
        # Additional penalties
        border_pen = self._border_penalty(margin=1)
        
        # Final reward
        # Scale solvability bonus by number of objects (more objects = harder)
        solvability_bonus = 3.0 * self.n_objects
        R = (
            + 2.0  # Base validity bonus
            + solvability_bonus  # Solvability bonus (scaled by n_objects)
            + self.alpha * (L1_total + L2_total)  # Total path length across all pairs
            + wall_term  # Wall ratio
            + corridor_term  # Corridor quality
            - self.gamma * adj_trivial  # Trivial adjacency
            - self.delta * border_pen  # Border penalty
        )
        
        if ws["n_walls"] == 0:
            R -= 0.5  # Penalty for no structure
        
        metrics.update({
            "valid": 1,
            "L1": int(L1_avg),  # Average for backward compatibility
            "L2": int(L2_avg),  # Average for backward compatibility
            "wall_ratio": wr,
            "adj_per_wall": adj_per_wall,
            "iso_frac": iso_frac,
            "n_objects": self.n_objects,  # Track number of objects
            "final_grid": self.grid.copy()
        })
        
        return float(R), metrics
    
    def reset(self, *, seed: int | None = None, options=None):
        """Reset environment: place entities randomly, clear walls."""
        if seed is not None:
            self.rng.seed(seed)
        
        self.steps = 0
        self.grid.fill(EMPTY)
        self.entity_positions.clear()
        
        # Place entities randomly
        self._place_entities_randomly()
        
        return self._obs(), {}
    
    def step(self, action: int):
        """Execute one step."""
        # Decode action: (y, x, tile_type)
        idx = int(action) // PHASE1_N_TILES
        t_idx = int(action) % PHASE1_N_TILES
        t = PHASE1_TILES[t_idx]
        y, x = divmod(idx, self.w)
        
        reward = 0.0
        
        # Hard-block: Cannot place on entities
        if (y, x) in self.entity_positions:
            reward -= 0.01  # Small penalty for invalid attempt
            # No grid change
        elif t == WALL:
            if self.grid[y, x] == WALL:
                reward -= 0.01  # Redundant placement
            else:
                # Place the wall - NO per-step solvability check
                # Solvability is only checked at episode END via terminal reward
                # This removes the "solvability ceiling" that was preventing higher wall ratios
                self.grid[y, x] = WALL
                
                # Simple per-step shaping based on wall ratio
                ws = self._wall_stats()
                wr = ws["ratio"]
                
                if wr < self.wall_target:
                    # Below target: reward placing walls
                    # Stronger bonus the further below target we are
                    target_gap = self.wall_target - wr
                    reward += 0.10 + 0.20 * min(1.0, target_gap / 0.2)  # 0.10 to 0.30 per wall
                elif wr > self.wall_target + 0.05:
                    # Above target: penalize placing walls
                    reward -= 0.15
                else:
                    # At target: small bonus
                    reward += 0.05
        else:  # EMPTY
            if self.grid[y, x] == EMPTY:
                reward -= 0.01  # Redundant
            elif self.grid[y, x] == WALL:
                self.grid[y, x] = EMPTY  # Remove wall
                reward -= 0.02  # Small penalty for removing wall
            # If entity, we already handled it above
        
        # Per-step shaping (small bonuses for good wall placement)
        if t == WALL and self.grid[y, x] == WALL:
            g_walls = (self.grid == WALL).astype(np.int32)
            deg = 0
            for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
                ny, nx = y+dy, x+dx
                if 0 <= ny < self.h and 0 <= nx < self.w and g_walls[ny, nx]:
                    deg += 1
            
            if deg == 1:
                reward += 0.03  # Good: extending a chain
            elif deg == 2:
                reward += 0.02  # Good: corridor-like
            elif deg == 0:
                reward -= 0.03  # Bad: isolated
        
        self.steps += 1
        
        # Check termination
        terminated = False
        truncated = False
        early_terminated = False
        
        if self.steps >= self.max_steps:
            terminated = True
        elif self._can_early_terminate():
            terminated = True
            early_terminated = True
            # Scale early termination bonus based on how close to target
            ws = self._wall_stats()
            wall_dev = abs(ws["ratio"] - self.wall_target)
            # Bonus scales from full (0.2) at target to 0 at ±0.05 deviation
            early_bonus_scaled = self.early_term_bonus * max(0.0, 1.0 - wall_dev / 0.05)
            reward += early_bonus_scaled
        
        info = {}
        if terminated:
            if not self._valid_final():
                wr = float(np.mean(self.grid == WALL))
                reward += -30.0  # Strong penalty for invalid grid (matches unsolvable penalty)
                info = {"valid": 0, "L1": 0, "L2": 0, "wall_ratio": wr, "final_grid": self.grid.copy()}
            else:
                r_eval, metrics = self._evaluate_grid()
                reward += r_eval
                metrics["final_grid"] = self.grid.copy()
                metrics["early_terminated"] = early_terminated  # Track if episode ended early
                metrics["episode_length"] = self.steps  # Track episode length
                info = metrics
        
        return self._obs(), float(reward), terminated, truncated, info
    
    def render(self):
        """Render grid as text."""
        chars = {EMPTY: ".", WALL: "#", ROBOT: "R", OBJECT: "O", GOAL: "G", MOVABLE: "M"}
        print("\n".join("".join(chars.get(int(self.grid[y, x]), "?") for x in range(self.w)) for y in range(self.h)))

