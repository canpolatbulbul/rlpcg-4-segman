# phase2_env.py
# Phase 2: Challenge Learning Environment
# Agent learns to add movable obstacles to make puzzles challenging
# Base puzzles are generated using a trained Phase 1 model

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque

# Import shared utilities
from grid_pcg_env import (
    EMPTY, WALL, ROBOT, OBJECT, GOAL, MOVABLE,
    shortest_path_len
)

# Phase 2 only uses EMPTY and MOVABLE
PHASE2_TILES = (EMPTY, MOVABLE)
PHASE2_N_TILES = len(PHASE2_TILES)


class Phase2Env(gym.Env):
    """
    Phase 2 Environment: Challenge Learning
    
    - Base puzzle (walls + entities) is generated using Phase 1 model at reset
    - Walls and entities are frozen (agent cannot modify them)
    - Agent can only place MOVABLE or EMPTY
    - Goal: Learn to place movables in critical positions (movable-critical condition)
    - Early termination allowed after min_steps if movable-critical achieved
    """
    
    metadata = {"render_modes": []}
    
    def __init__(
        self,
        phase1_model_path: str,  # Path to trained Phase 1 model
        size: int = 13,
        max_steps: int = 80,
        min_steps: int = 30,  # Minimum steps before early termination
        seed: int | None = None,
        phase1_deterministic: bool = False,  # Whether to use Phase 1 model deterministically
    ):
        super().__init__()
        self.h = int(size)
        self.w = int(size)
        self.size = int(size)
        self.max_steps = int(max_steps)
        self.min_steps = int(min_steps)
        self.phase1_deterministic = bool(phase1_deterministic)
        
        # Load Phase 1 model
        from stable_baselines3 import PPO
        from phase1_env import Phase1Env
        
        # Create a temporary Phase 1 env for the model
        temp_phase1_env = Phase1Env(size=size, max_steps=150, seed=seed)
        self.phase1_model = PPO.load(phase1_model_path, env=temp_phase1_env, device="auto")
        self.phase1_env = temp_phase1_env
        
        self.steps = 0
        self.rng = np.random.RandomState(seed if seed is not None else 42)
        
        # Reward hyperparameters
        self.movable_critical_bonus = 10.0
        self.movable_critical_penalty = -5.0
        self.relaxed_unsolvable_penalty = -3.0
        self.movable_count_bonus = 0.5  # Small bonus for reasonable count (3-7)
        self.structure_preservation_bonus = 0.3
        self.early_term_bonus = 0.3
        
        # Grid
        self.grid = np.full((self.h, self.w), EMPTY, dtype=np.int32)
        
        # Track frozen cells (walls and entities)
        self.frozen_cells = set()
        
        # Observation: H x W x 6 (one-hot for all tile types)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.h, self.w, 6), dtype=np.float32
        )
        # Action: Discrete(H * W * 2) for (y, x, tile_type) where tile_type ∈ {EMPTY, MOVABLE}
        self.action_space = spaces.Discrete(self.h * self.w * PHASE2_N_TILES)
    
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
    
    def _generate_base_puzzle(self) -> np.ndarray:
        """
        Use Phase 1 model to generate a base puzzle.
        Returns the grid with walls and entities placed.
        """
        obs, _ = self.phase1_env.reset(seed=self.rng.randint(0, 2**31))
        done = False
        
        # Run Phase 1 agent for one episode
        for _ in range(self.phase1_env.max_steps):
            action, _ = self.phase1_model.predict(obs, deterministic=self.phase1_deterministic)
            obs, reward, term, trunc, info = self.phase1_env.step(int(action))
            if term or trunc:
                done = True
                break
        
        # Get the final grid
        if done and "final_grid" in info:
            base_grid = info["final_grid"].copy()
        else:
            # Fallback: use current grid from phase1_env
            base_grid = self.phase1_env.grid.copy()
        
        return base_grid
    
    def _valid_final(self) -> bool:
        """Check if all entities are present."""
        return (self._count(ROBOT) == 1 and
                self._count(OBJECT) == 1 and
                self._count(GOAL) == 1)
    
    def _can_early_terminate(self) -> bool:
        """Check if early termination conditions are met (movable-critical achieved)."""
        if self.steps < self.min_steps:
            return False
        
        if not self._valid_final():
            return False
        
        ry, rx = self._pos(ROBOT)
        oy, ox = self._pos(OBJECT)
        gy, gx = self._pos(GOAL)
        
        # Check relaxed solvability
        L1_relaxed = shortest_path_len(self.grid, (ry, rx), (oy, ox), treat_movable_as_empty=True)
        L2_relaxed = shortest_path_len(self.grid, (oy, ox), (gy, gx), treat_movable_as_empty=True)
        
        if L1_relaxed is None or L2_relaxed is None:
            return False
        
        # Check strict solvability (should be unsolvable)
        L1_strict = shortest_path_len(self.grid, (ry, rx), (oy, ox), treat_movable_as_empty=False)
        L2_strict = shortest_path_len(self.grid, (oy, ox), (gy, gx), treat_movable_as_empty=False)
        
        strict_solvable = (L1_strict is not None) and (L2_strict is not None)
        
        # Movable-critical: relaxed solvable AND strict unsolvable
        movable_critical = not strict_solvable
        
        return movable_critical
    
    def _evaluate_grid(self):
        """Evaluate final grid and compute terminal reward."""
        metrics = {
            "valid": 0,
            "relaxed_solvable": False,
            "strict_solvable": False,
            "movable_critical": False,
            "n_movable": 0,
            "L1_relaxed": 0,
            "L2_relaxed": 0,
            "L1_strict": None,
            "L2_strict": None,
        }
        
        if not self._valid_final():
            return -10.0, metrics
        
        ry, rx = self._pos(ROBOT)
        oy, ox = self._pos(OBJECT)
        gy, gx = self._pos(GOAL)
        
        # Check relaxed solvability
        L1_relaxed = shortest_path_len(self.grid, (ry, rx), (oy, ox), treat_movable_as_empty=True)
        L2_relaxed = shortest_path_len(self.grid, (oy, ox), (gy, gx), treat_movable_as_empty=True)
        relaxed_solvable = (L1_relaxed is not None) and (L2_relaxed is not None)
        
        # Check strict solvability
        L1_strict = shortest_path_len(self.grid, (ry, rx), (oy, ox), treat_movable_as_empty=False)
        L2_strict = shortest_path_len(self.grid, (oy, ox), (gy, gx), treat_movable_as_empty=False)
        strict_solvable = (L1_strict is not None) and (L2_strict is not None)
        
        # Movable-critical condition
        movable_critical = relaxed_solvable and (not strict_solvable)
        
        n_movable = self._count(MOVABLE)
        
        # Reward computation
        if not relaxed_solvable:
            # Bad: Even with pushable movables, puzzle is unsolvable
            R = self.relaxed_unsolvable_penalty
        elif movable_critical:
            # IDEAL: Movable-critical achieved
            R = self.movable_critical_bonus
        else:
            # Both solvable: movables are not critical
            R = self.movable_critical_penalty
        
        # Movable count constraint (target: 3-6 movables)
        target_min, target_max = 3, 6
        if target_min <= n_movable <= target_max:
            R += 1.0  # Bonus for being in ideal range
        elif n_movable > target_max:
            # Strong penalty for excess movables (scales with how many extra)
            excess = n_movable - target_max
            R -= 0.5 * excess  # Linear penalty: 10 movables = -2.0, 20 = -7.0
        elif n_movable < target_min and n_movable > 0:
            R -= 1.0  # Penalty for too few (but not as harsh)
        
        R += self.structure_preservation_bonus  # Small bonus for maintaining structure
        
        metrics.update({
            "valid": 1,
            "relaxed_solvable": relaxed_solvable,
            "strict_solvable": strict_solvable,
            "movable_critical": movable_critical,
            "n_movable": n_movable,
            "L1_relaxed": int(L1_relaxed) if L1_relaxed is not None else 0,
            "L2_relaxed": int(L2_relaxed) if L2_relaxed is not None else 0,
            "L1_strict": int(L1_strict) if L1_strict is not None else None,
            "L2_strict": int(L2_strict) if L2_strict is not None else None,
            "final_grid": self.grid.copy()
        })
        
        return float(R), metrics
    
    def reset(self, *, seed: int | None = None, options=None):
        """Reset: Generate new base puzzle using Phase 1 model."""
        if seed is not None:
            self.rng.seed(seed)
        
        self.steps = 0
        self.frozen_cells.clear()
        
        # Generate base puzzle using Phase 1 model
        base_grid = self._generate_base_puzzle()
        self.grid = base_grid.copy()
        
        # Mark frozen cells (walls and entities)
        for y in range(self.h):
            for x in range(self.w):
                if self.grid[y, x] in (WALL, ROBOT, OBJECT, GOAL):
                    self.frozen_cells.add((y, x))
        
        return self._obs(), {}
    
    def step(self, action: int):
        """Execute one step."""
        # Decode action: (y, x, tile_type)
        idx = int(action) // PHASE2_N_TILES
        t_idx = int(action) % PHASE2_N_TILES
        t = PHASE2_TILES[t_idx]
        y, x = divmod(idx, self.w)
        
        reward = 0.0
        
        # Hard-block: Cannot modify frozen cells (walls/entities)
        if (y, x) in self.frozen_cells:
            reward -= 0.01  # Small penalty for invalid attempt
            # No grid change
        elif t == MOVABLE:
            if self.grid[y, x] == MOVABLE:
                reward -= 0.01  # Redundant placement
            else:
                self.grid[y, x] = MOVABLE
                reward += 0.02  # Small bonus for placing movable
        else:  # EMPTY
            if self.grid[y, x] == EMPTY:
                reward -= 0.01  # Redundant
            elif self.grid[y, x] == MOVABLE:
                self.grid[y, x] = EMPTY  # Remove movable
                reward -= 0.01  # Small penalty for removing
        
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
                reward += -10.0
                info = {"valid": 0, "final_grid": self.grid.copy()}
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

