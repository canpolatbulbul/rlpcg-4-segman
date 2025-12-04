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
        max_steps: int = 150,
        min_steps: int = 50,  # Minimum steps before early termination allowed
        seed: int | None = None,
        wall_target: float = 0.25,
        entity_min_distance: int = 5,  # Minimum distance between entities
        entity_border_margin: int = 0,  # Entities can be placed at edges (0 = no margin)
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
        
        self.steps = 0
        self.rng = np.random.RandomState(seed if seed is not None else 42)
        
        # Reward hyperparameters
        self.alpha = 0.06       # Path length weight
        self.gamma = 2.0        # Trivial adjacency penalty
        self.delta = 0.5        # Border penalty weight
        self.lambda_corridor_term = 1.5    # Corridor quality
        self.lambda_isolated_term = 0.9    # Isolated walls penalty
        self.lambda_block_term = 0.3       # 2x2 block penalty
        self.wall_ratio_penalty = 5.0      # Wall ratio deviation penalty
        
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
        ys, xs = np.where(self.grid == t)
        if ys.size == 0:
            return None
        return int(ys[0]), int(xs[0])
    
    def _manhattan_dist(self, pos1, pos2):
        y1, x1 = pos1
        y2, x2 = pos2
        return abs(y1 - y2) + abs(x1 - x2)
    
    def _place_entities_randomly(self):
        """
        Randomly place ROBOT, OBJECT, GOAL with constraints:
        - Minimum distance between any two entities
        - Can be placed at edges (no border margin restriction)
        """
        self.entity_positions.clear()
        entities = [ROBOT, OBJECT, GOAL]
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
                self._count(OBJECT) == 1 and
                self._count(GOAL) == 1)
    
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
        deg[1:,  :] += g_walls[:-1, :]
        deg[:-1, :] += g_walls[1:,  :]
        deg[:, 1:]  += g_walls[:, :-1]
        deg[:, :-1] += g_walls[:, :-1]
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
        ry, rx = self._pos(ROBOT)
        oy, ox = self._pos(OBJECT)
        gy, gx = self._pos(GOAL)
        
        L1 = shortest_path_len(self.grid, (ry, rx), (oy, ox), treat_movable_as_empty=True)
        L2 = shortest_path_len(self.grid, (oy, ox), (gy, gx), treat_movable_as_empty=True)
        
        if L1 is None or L2 is None:
            return False
        
        # Check wall ratio
        ws = self._wall_stats()
        wr = ws["ratio"]
        if not (0.20 <= wr <= 0.30):
            return False
        
        # Check path length (non-trivial)
        if (L1 + L2) < 10:
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
        
        ry, rx = self._pos(ROBOT)
        oy, ox = self._pos(OBJECT)
        gy, gx = self._pos(GOAL)
        
        # Check solvability (no movables, so just walls)
        L1 = shortest_path_len(self.grid, (ry, rx), (oy, ox), treat_movable_as_empty=True)
        L2 = shortest_path_len(self.grid, (oy, ox), (gy, gx), treat_movable_as_empty=True)
        
        if L1 is None or L2 is None:
            return -1.0, metrics
        
        # Trivial adjacency
        adj_trivial = 0.0
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
        R = (
            + 2.0  # Base validity bonus
            + 3.0  # Solvability bonus
            + self.alpha * (L1 + L2)  # Path length
            + wall_term  # Wall ratio
            + corridor_term  # Corridor quality
            - self.gamma * adj_trivial  # Trivial adjacency
            - self.delta * border_pen  # Border penalty
        )
        
        if ws["n_walls"] == 0:
            R -= 0.5  # Penalty for no structure
        
        metrics.update({
            "valid": 1,
            "L1": int(L1),
            "L2": int(L2),
            "wall_ratio": wr,
            "adj_per_wall": adj_per_wall,
            "iso_frac": iso_frac,
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
                self.grid[y, x] = WALL
                reward += 0.05  # Bonus for placing wall
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
        
        if self.steps >= self.max_steps:
            terminated = True
        elif self._can_early_terminate():
            terminated = True
            reward += self.early_term_bonus  # Early termination bonus
        
        info = {}
        if terminated:
            if not self._valid_final():
                wr = float(np.mean(self.grid == WALL))
                reward += -2.0
                info = {"valid": 0, "L1": 0, "L2": 0, "wall_ratio": wr, "final_grid": self.grid.copy()}
            else:
                r_eval, metrics = self._evaluate_grid()
                reward += r_eval
                metrics["final_grid"] = self.grid.copy()
                info = metrics
        
        return self._obs(), float(reward), terminated, truncated, info
    
    def render(self):
        """Render grid as text."""
        chars = {EMPTY: ".", WALL: "#", ROBOT: "R", OBJECT: "O", GOAL: "G", MOVABLE: "M"}
        print("\n".join("".join(chars.get(int(self.grid[y, x]), "?") for x in range(self.w)) for y in range(self.h)))

