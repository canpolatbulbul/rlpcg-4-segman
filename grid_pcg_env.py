# grid_pcg_env.py
# Python 3.8+, Gymnasium API
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque

# ---- Tile IDs ----
EMPTY, WALL, ROBOT, OBJECT, GOAL, MOVABLE = 0, 1, 2, 3, 4, 5
TILES = (EMPTY, WALL, ROBOT, OBJECT, GOAL, MOVABLE)
N_TILES = len(TILES)

# ---- BFS on 4-connected grid, walls and movable obstacles are blocked ----
# ---- BFS on 4-connected grid ----
def shortest_path_len(grid: np.ndarray, start, goal, treat_movable_as_empty=False) -> int | None:
    H, W = grid.shape
    (sy, sx), (gy, gx) = start, goal
    if (sy, sx) == (gy, gx):
        return 0
    seen = np.zeros_like(grid, dtype=bool)
    q = deque()
    q.append((sy, sx, 0))
    seen[sy, sx] = True
    while q:
        y, x, d = q.popleft()
        for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
            ny, nx = y+dy, x+dx
            if ny < 0 or ny >= H or nx < 0 or nx >= W:
                continue
            if seen[ny, nx]:
                continue
            
            cell = grid[ny, nx]
            # WALL always blocks. MOVABLE blocks unless treated as empty.
            if cell == WALL:
                continue
            if cell == MOVABLE and not treat_movable_as_empty:
                continue
                
            if (ny, nx) == (gy, gx):
                return d + 1
            seen[ny, nx] = True
            q.append((ny, nx, d+1))
    return None

def get_shortest_path_cells(grid: np.ndarray, start, goal, treat_movable_as_empty=False) -> set:
    """
    Returns a set of (y, x) coordinates on the shortest path.
    If multiple shortest paths exist, this returns one of them (arbitrary).
    Returns empty set if no path.
    """
    H, W = grid.shape
    (sy, sx), (gy, gx) = start, goal
    if (sy, sx) == (gy, gx):
        return {(sy, sx)}
        
    # BFS to find distance to start from all reachable cells
    # We'll search backwards from start to fill a dist map, then trace back from goal?
    # Actually, standard BFS from start recording parents is easier for reconstruction.
    
    parent = {}
    q = deque()
    q.append((sy, sx))
    parent[(sy, sx)] = None
    visited = np.zeros_like(grid, dtype=bool)
    visited[sy, sx] = True
    
    found = False
    while q:
        y, x = q.popleft()
        if (y, x) == (gy, gx):
            found = True
            break
        
        for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
            ny, nx = y+dy, x+dx
            if 0 <= ny < H and 0 <= nx < W and not visited[ny, nx]:
                cell = grid[ny, nx]
                is_blocked = (cell == WALL) or (cell == MOVABLE and not treat_movable_as_empty)
                if not is_blocked:
                    visited[ny, nx] = True
                    parent[(ny, nx)] = (y, x)
                    q.append((ny, nx))
    
    path_cells = set()
    if found:
        curr = (gy, gx)
        while curr is not None:
            path_cells.add(curr)
            curr = parent[curr]
    return path_cells


class GridPCGEnv(gym.Env):
    """
    SeGMaN-style PCG environment.

    - No SUBMIT: episodes last exactly `max_steps`.
    - Unique entities with "move semantics": placing ROBOT/OBJECT/GOAL moves (or creates) that entity.
    - WALL and MOVABLE cannot overwrite an entity.
    - Final reward at episode end (solvability + shaping), plus small per-step shaping.
    - Observation: H x W x 6 (one-hot planes for [EMPTY, WALL, ROBOT, OBJECT, GOAL, MOVABLE]).
    - Action: Discrete(H * W * 6): (y, x, tile_type).
    - MOVABLE obstacles block pathfinding like walls, but are tracked separately for reward shaping.
    """

    metadata = {"render_modes": []}

    def __init__(
            self,
            size: int = 13,
            max_steps: int = 192,         # give enough action budget to place walls
            seed: int | None = None,
            use_curriculum: bool = True,  # 0-2 entities pre-placed at reset
            wall_target: float = 0.25,    # target wall ratio
    ):
        super().__init__()
        self.h = int(size)
        self.w = int(size)
        self.size = int(size)
        self.max_steps = int(max_steps)
        self.use_curriculum = bool(use_curriculum)
        self.wall_target = float(wall_target)
        
        # Curriculum state
        self.progress = 0.0 # 0.0 to 1.0
        self._cur_wall_target = 0.05
        self._cur_movable_target = 0.0

        self.steps = 0
        self.rng = np.random.RandomState(seed if seed is not None else 42)

        # -------- reward hyperparams (tuned for stability) --------
        # Primary reward signals (terminal):
        # - Base solvability (relaxed): +2.0
        # - Movable-critical: +3.0 (dominant) / -2.0 (both solvable) / -3.0 (relaxed unsolvable)
        # - Path length: alpha * (L1 + L2) - encourages non-trivial but solvable paths
        # - Wall ratio: -5.0 * |deviation| - strong structural constraint
        # - Corridor quality: encourages structured walls
        
        self.alpha = 0.06       # weight on L1+L2 (path lengths, using relaxed paths)
        self.gamma = 2.0        # adjacency (trivial) penalty
        self.delta = 0.5        # border penalty weight (terminal)

        # corridor / structure shaping
        self.lambda_corridor_term = 1.5    # terminal connectedness reward
        self.lambda_corridor_step = 0.25   # tiny incremental reward for more adjacency per wall
        self.lambda_isolated_term = 0.9    # terminal penalty for isolated walls
        self.lambda_block_term = 0.3       # terminal penalty for 2x2 wall blocks
        
        # movable obstacle reward shaping (secondary to movable-critical condition)
        self.lambda_movable = 0.5  # bonus for having movable obstacles in target range (terminal)
        # Scale target count with grid size (approx 3% density)
        # 13x13 (169) -> ~5.0
        # 16x16 (256) -> ~7.6
        # 20x20 (400) -> ~12.0
        self.movable_desired_count = max(3.0, (self.w * self.h) * 0.03)
        
        # Note: Removed lambda_movable_on_path, lambda_movable_off_path, lambda_boxed, lambda_path_obstruction
        # These are replaced by the explicit movable-critical check in _evaluate_grid()

        # per-step coax toward "some walls" after all 3 entities exist
        self.wall_step_coax = 0.30 # V4: Massive boost (was 0.15)
        # useful band for final wall ratio bonus
        self._wall_band_lo = 0.20
        self._wall_band_hi = 0.50
        # grid
        self.grid = np.full((self.h, self.w), EMPTY, dtype=np.int32)

        # spaces
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.h, self.w, N_TILES), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.h * self.w * N_TILES)

        # small per-step costs/bonuses
        self.step_cost = 0.0002
        self.first_valid_bonus = 0.15
        self._was_valid = False

    def set_progress(self, p: float):
        """
        Update curriculum progress (0.0 -> 1.0).
        Schedule:
          0.0 - 0.2: Basics (Walls 0.05, Movables 0)
          0.2 - 0.5: Walls Ramp (0.05 -> 0.25)
          0.5 - 0.8: Movables Ramp (0 -> 7)
          0.8 - 1.0: Polish (Full constraints)
        """
        self.progress = np.clip(p, 0.0, 1.0)
        
        # Wall Target Schedule
        if self.progress < 0.2:
            self._cur_wall_target = 0.05
        elif self.progress < 0.5:
            # Linear ramp 0.05 -> 0.25
            ratio = (self.progress - 0.2) / 0.3
            self._cur_wall_target = 0.05 + ratio * (0.25 - 0.05)
        else:
            self._cur_wall_target = 0.25
            
        # Movable Target Schedule
        if self.progress < 0.5:
            self._cur_movable_target = 0.0
        elif self.progress < 0.8:
            # Linear ramp 0 -> target
            ratio = (self.progress - 0.5) / 0.3
            self._cur_movable_target = 0.0 + ratio * self.movable_desired_count
        else:
            self._cur_movable_target = self.movable_desired_count

    # ---------- helpers ----------
    def _obs(self) -> np.ndarray:
        planes = np.zeros((self.h, self.w, N_TILES), dtype=np.float32)
        for t in TILES:
            planes[:, :, t] = (self.grid == t)
        return planes

    def _count(self, t: int) -> int:
        return int(np.sum(self.grid == t))

    def _pos(self, t: int):
        ys, xs = np.where(self.grid == t)
        if ys.size == 0:
            return None
        return int(ys[0]), int(xs[0])

    def _place_unique(self, t: int, y: int, x: int):
        assert t in (ROBOT, OBJECT, GOAL)
        cur = self._pos(t)
        if cur is not None:
            cy, cx = cur
            self.grid[cy, cx] = EMPTY
        self.grid[y, x] = t

    def _seed_small_segment(self, length: int):
        y = int(self.rng.randint(1, self.h - 1))
        x = int(self.rng.randint(1, self.w - 1))
        if self.rng.rand() < 0.5:
            # horizontal
            for k in range(length):
                xx = x + k
                if 0 < xx < self.w - 1 and self.grid[y, xx] not in (ROBOT, OBJECT, GOAL):
                    self.grid[y, xx] = WALL
        else:
            # vertical
            for k in range(length):
                yy = y + k
                if 0 < yy < self.h - 1 and self.grid[yy, x] not in (ROBOT, OBJECT, GOAL):
                    self.grid[yy, x] = WALL

    def _seed_walls_bootstrap(self):
        # 70% chance to pre-seed 1–3 short segments (length 2–4)
        if self.rng.rand() < 0.7:
            n = int(self.rng.randint(1, 4))
            for _ in range(n):
                L = int(self.rng.randint(2, 5))
                self._seed_small_segment(L)

    def _valid_final(self) -> bool:
        return (self._count(ROBOT) == 1 and
                self._count(OBJECT) == 1 and
                self._count(GOAL) == 1)

    def _border_penalty(self, margin: int = 1) -> float:
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
        """
        Compute statistics for walls and movable obstacles.
        Returns stats for combined solid mask (walls + movables) and separately for movables.
        """
        # Combined solid mask (walls + movables) for corridor structure
        g_solid = ((self.grid == WALL) | (self.grid == MOVABLE)).astype(np.int32)
        n_solid = int(g_solid.sum())
        
        # Separate counts
        n_walls = int(np.sum(self.grid == WALL))
        n_movable = int(np.sum(self.grid == MOVABLE))
        
        if n_solid == 0:
            return dict(
                n_walls=n_walls, n_movable=n_movable, n_solid=0,
                ratio=0.0, solid_ratio=0.0, movable_ratio=0.0,
                n_isolated=0, n_adj_pairs=0, n_2x2=0
            )

        # 4-neighborhood adjacency on combined solid mask
        up = g_solid[:-1, :] * g_solid[1:, :]
        left = g_solid[:, :-1] * g_solid[:, 1:]
        n_adj_pairs = int(up.sum() + left.sum())

        deg = np.zeros_like(g_solid, dtype=np.int32)
        deg[1:,  :] += g_solid[:-1, :]
        deg[:-1, :] += g_solid[1:,  :]
        deg[:, 1:]  += g_solid[:, :-1]
        deg[:, :-1] += g_solid[:, 1:]
        n_isolated = int(((g_solid == 1) & (deg == 0)).sum())

        n_2x2 = int((g_solid[:-1, :-1] * g_solid[1:, :-1] * g_solid[:-1, 1:] * g_solid[1:, 1:]).sum())
        
        ratio = n_walls / float(self.h * self.w)  # wall ratio (for backward compat)
        solid_ratio = n_solid / float(self.h * self.w)  # combined solid ratio
        movable_ratio = n_movable / float(self.h * self.w)

        return dict(
            n_walls=n_walls, n_movable=n_movable, n_solid=n_solid,
            ratio=ratio, solid_ratio=solid_ratio, movable_ratio=movable_ratio,
            n_isolated=n_isolated, n_adj_pairs=n_adj_pairs, n_2x2=n_2x2
        )

    def _free_space_components(self):
        H, W = self.h, self.w
        # Both WALL and MOVABLE block free space connectivity
        blocked = (self.grid == WALL) | (self.grid == MOVABLE)
        seen = np.zeros((H, W), dtype=bool)
        comps = 0
        for y in range(H):
            for x in range(W):
                if blocked[y, x] or seen[y, x]:
                    continue
                comps += 1
                q = [(y, x)]
                seen[y, x] = True
                while q:
                    cy, cx = q.pop()
                    for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < H and 0 <= nx < W and not blocked[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            q.append((ny, nx))
        return comps

    # ---------- terminal score ----------
    def _evaluate_grid(self):
        """
        Evaluate the final grid and compute terminal reward.
        
        Key logic:
        1. RELAXED solvability: Treat MOVABLE as empty (pushable). Both paths must exist.
        2. STRICT solvability: Treat MOVABLE as wall (solid). At least one path must be broken.
        3. MOVABLE-CRITICAL: Relaxed solvable AND strict unsolvable = ideal puzzle.
        
        Reward structure prioritizes:
        - Base solvability (relaxed) - essential
        - Movable-critical condition - dominant signal for good puzzles
        - Structural quality (walls, corridors) - important for aesthetics
        """
        wall_ratio_observed = float(np.mean(self.grid == WALL))
        metrics = {
            "valid": 0, "L1": 0, "L2": 0,
            "wall_ratio": wall_ratio_observed,
            "adj_per_wall": 0.0,
            "iso_frac": 0.0,
            "n_movable": 0,
            "movable_ratio": 0.0,
            "solid_ratio": 0.0,
            "n_movable_on_path": 0,
            "n_boxed": 0,
            # New metrics for movable-critical analysis
            "relaxed_solvable": False,
            "strict_solvable": False,
            "movable_critical": False,
            "L1_relaxed": 0,
            "L2_relaxed": 0,
            "L1_strict": None,
            "L2_strict": None,
        }

        if not self._valid_final():
            return -1.0, metrics

        ry, rx = self._pos(ROBOT)
        oy, ox = self._pos(OBJECT)
        gy, gx = self._pos(GOAL)

        # ========================================================================
        # CRITICAL: Explicit strict/relaxed solvability checks
        # ========================================================================
        
        # RELAXED: Treat MOVABLE as empty (pushable) - puzzle must be solvable
        L1_relaxed = shortest_path_len(self.grid, (ry, rx), (oy, ox), treat_movable_as_empty=True)
        L2_relaxed = shortest_path_len(self.grid, (oy, ox), (gy, gx), treat_movable_as_empty=True)
        relaxed_solvable = (L1_relaxed is not None) and (L2_relaxed is not None)
        
        # STRICT: Treat MOVABLE as wall (solid) - puzzle should be unsolvable
        L1_strict = shortest_path_len(self.grid, (ry, rx), (oy, ox), treat_movable_as_empty=False)
        L2_strict = shortest_path_len(self.grid, (oy, ox), (gy, gx), treat_movable_as_empty=False)
        strict_solvable = (L1_strict is not None) and (L2_strict is not None)
        
        # MOVABLE-CRITICAL condition: Relaxed solvable AND strict unsolvable
        # This means MOVABLE obstacles are truly critical blockers
        movable_critical = relaxed_solvable and (not strict_solvable)
        
        # ========================================================================
        # Movable-Critical Reward Logic (DOMINANT signal)
        # ========================================================================
        
        movable_critical_reward = 0.0
        
        if not relaxed_solvable:
            # Bad level: Even with pushable movables, puzzle is unsolvable
            # This could be due to walls blocking paths - heavy penalty
            movable_critical_reward = -3.0
        elif movable_critical:
            # IDEAL: Relaxed solvable, strict unsolvable
            # MOVABLE obstacles are critical blockers - strong bonus
            movable_critical_reward = +3.0
        elif strict_solvable:
            # Both relaxed and strict are solvable
            # MOVABLE obstacles are decorative, not functional - penalty
            movable_critical_reward = -2.0
        else:
            # Relaxed solvable, strict unsolvable (but we already checked movable_critical)
            # This shouldn't happen, but give small bonus for relaxed solvability
            movable_critical_reward = +2.0
        
        # Base reward for relaxed solvability (essential requirement)
        base_solvability_reward = +2.0 if relaxed_solvable else 0.0
        
        # ========================================================================
        # Structural Rewards (walls, corridors, etc.)
        # ========================================================================
        
        # trivial adjacency (touching)
        adj_trivial = 0.0
        if max(abs(ry - oy), abs(rx - ox)) <= 1:
            adj_trivial = 1.0
        if max(abs(oy - gy), abs(ox - gx)) <= 1:
            adj_trivial = 1.0

        # wall/obstacle stats (includes combined solid mask)
        ws = self._wall_stats()
        n_movable = ws["n_movable"]
        solid_ratio = ws["solid_ratio"]
        wr = ws["ratio"]
        
        # Wall ratio targeting (dynamic curriculum)
        wall_dev = abs(wr - self._cur_wall_target)
        band_lo = max(0.0, self._cur_wall_target - 0.05)
        band_hi = min(1.0, self._cur_wall_target + 0.05)
        band_bonus = 0.5 if (band_lo <= wr <= band_hi) else -0.5
        wall_term = band_bonus - 5.0 * wall_dev

        # Corridor quality (encourages structured walls, not blobs)
        if ws["n_solid"] > 0:
            adj_per_wall = ws["n_adj_pairs"] / float(ws["n_solid"])
            iso_frac = ws["n_isolated"] / float(ws["n_solid"])
        else:
            adj_per_wall, iso_frac = 0.0, 0.0

        corridor_term = (
                + self.lambda_corridor_term * adj_per_wall
                - self.lambda_isolated_term * iso_frac
                - self.lambda_block_term * (ws["n_2x2"] / max(1, ws["n_solid"]))
        )

        # ========================================================================
        # Movable Quantity Reward (secondary to movable-critical)
        # ========================================================================
        
        movable_count_term = 0.0
        
        # Early curriculum: penalize movables if target is 0
        if self._cur_movable_target < 0.1:
            if n_movable > 0:
                movable_count_term = -0.1 * n_movable
        elif n_movable > 0:
            # Target range: target +/- 2
            t_min = max(1, self._cur_movable_target - 2)
            t_max = self._cur_movable_target + 2
            
            if t_min <= n_movable <= t_max:
                movable_count_term = self.lambda_movable * 0.5  # Reduced weight (secondary to critical)
            else:
                diff = min(abs(n_movable - t_min), abs(n_movable - t_max))
                movable_count_term = -0.2 * diff
        
        # ========================================================================
        # Path Length Reward (use relaxed paths - encourages solvability)
        # ========================================================================
        
        if relaxed_solvable:
            path_len_reward = self.alpha * (L1_relaxed + L2_relaxed)
        else:
            path_len_reward = 0.0  # No reward if unsolvable
        
        # ========================================================================
        # Additional Penalties
        # ========================================================================
        
        border_pen = self._border_penalty(margin=1)
        free_comps = self._free_space_components()
        free_space_pen = 0.15 * max(0, free_comps - 1)

        # ========================================================================
        # Final Reward Assembly
        # ========================================================================
        
        R = (
            base_solvability_reward          # Essential: relaxed solvability
            + movable_critical_reward        # DOMINANT: movable-critical condition
            + path_len_reward                # Encourage non-trivial paths
            + wall_term                      # Structural: wall ratio
            + corridor_term                  # Structural: corridor quality
            + movable_count_term             # Secondary: movable quantity
            - self.gamma * adj_trivial       # Penalty: trivial adjacency
            - self.delta * border_pen        # Penalty: entities near borders
            - free_space_pen                 # Penalty: disconnected free space
        )
             
        if ws["n_solid"] == 0:
            R -= 0.5  # Penalty: no structure at all

        # ========================================================================
        # Update Metrics
        # ========================================================================
        
        metrics.update({
            "valid": 1, 
            "L1": int(L1_relaxed) if L1_relaxed is not None else 0,  # Use relaxed for backward compat
            "L2": int(L2_relaxed) if L2_relaxed is not None else 0,
            "wall_ratio": wr,
            "adj_per_wall": adj_per_wall,
            "iso_frac": iso_frac,
            "n_movable": n_movable,
            "movable_ratio": ws["movable_ratio"],
            "solid_ratio": solid_ratio,
            "n_movable_on_path": 0,  # Deprecated but kept for backward compat
            "n_boxed": 0,  # Deprecated but kept for backward compat
            # New metrics
            "relaxed_solvable": relaxed_solvable,
            "strict_solvable": strict_solvable,
            "movable_critical": movable_critical,
            "L1_relaxed": int(L1_relaxed) if L1_relaxed is not None else 0,
            "L2_relaxed": int(L2_relaxed) if L2_relaxed is not None else 0,
            "L1_strict": int(L1_strict) if L1_strict is not None else None,
            "L2_strict": int(L2_strict) if L2_strict is not None else None,
            "final_grid": self.grid.copy()
        })
        return float(R), metrics

    # ---------- Gymnasium API ----------
    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None:
            self.rng.seed(seed)
        self.steps = 0
        # 1. Fill with EMPTY
        self.grid.fill(EMPTY)
        
        # 2. Subtractive Curriculum: Seed with walls if target is high
        if self._cur_wall_target > 0.10:
            seed_density = self._cur_wall_target + 0.10
            mask = self.rng.rand(self.h, self.w) < seed_density
            self.grid[mask] = WALL
        else:
            # Low target: use old bootstrap (random segments)
            self._seed_walls_bootstrap()

        # 3. Place unique entities (curriculum)
        if self.use_curriculum:
            k = int(self.rng.randint(0, 3))  # 0, 1, or 2
            pool = [ROBOT, OBJECT, GOAL]
            self.rng.shuffle(pool)
            used = set()
            for t in pool[:k]:
                while True:
                    y = int(self.rng.randint(self.h))
                    x = int(self.rng.randint(self.w))
                    if (y, x) not in used and self.grid[y, x] == EMPTY:
                        used.add((y, x))
                        break
                self._place_unique(t, y, x)
        
        self._was_valid = self._valid_final()
        return self._obs(), {}

    def step(self, action: int):
        idx = int(action) // N_TILES
        t = int(action) % N_TILES
        y, x = divmod(idx, self.w)

        reward = 0.0

        def manhattan(a, b):
            (ay, ax), (by, bx) = a, b
            return abs(ay - by) + abs(ax - bx)

        def spread_total():
            r = self._pos(ROBOT); o = self._pos(OBJECT); g = self._pos(GOAL)
            if (r is None) or (o is None) or (g is None):
                return None
            return manhattan(r, o) + manhattan(o, g)

        def path_total():
            r = self._pos(ROBOT); o = self._pos(OBJECT); g = self._pos(GOAL)
            if (r is None) or (o is None) or (g is None):
                return None
            L1 = shortest_path_len(self.grid, r, o)
            L2 = shortest_path_len(self.grid, o, g)
            if L1 is None or L2 is None:
                return 0
            return L1 + L2

        prev_grid = self.grid.copy()
        had_R = self._count(ROBOT) > 0
        had_O = self._count(OBJECT) > 0
        had_G = self._count(GOAL)  > 0
        prev_valid = self._valid_final()
        prev_total = path_total()
        prev_spread = spread_total()

        prev_stats = self._wall_stats()
        prev_adj_per_wall = (prev_stats["n_adj_pairs"] / float(prev_stats["n_solid"])
                             if prev_stats["n_solid"] > 0 else 0.0)
        prev_iso_frac = (prev_stats["n_isolated"] / float(prev_stats["n_solid"])
                         if prev_stats["n_solid"] > 0 else 0.0)

        # Pre-calculate static path cells for shaping
        path_cells_static = set()
        if prev_valid:
            ry, rx = self._pos(ROBOT)
            oy, ox = self._pos(OBJECT)
            gy, gx = self._pos(GOAL)
            path_cells_static = get_shortest_path_cells(self.grid, (ry, rx), (oy, ox), treat_movable_as_empty=True)
            path_cells_static |= get_shortest_path_cells(self.grid, (oy, ox), (gy, gx), treat_movable_as_empty=True)

        # apply edit
        if t in (ROBOT, OBJECT, GOAL):
            cur = self._pos(t)
            if cur == (y, x):
                reward -= 0.05
            else:
                self._place_unique(t, y, x)
        elif t == WALL:
            if self.grid[y, x] in (ROBOT, OBJECT, GOAL):
                reward -= 0.02
            elif self.grid[y, x] == WALL:
                reward -= 0.01
            else:
                self.grid[y, x] = WALL
                reward += 0.05 # V4: Explicit bonus for placing a wall
        elif t == MOVABLE:
            # Movable obstacles: cannot overwrite entities, can replace walls or empty cells
            # Per-step shaping: small bonuses for placing movables (terminal reward handles criticality)
            if self.grid[y, x] in (ROBOT, OBJECT, GOAL):
                reward -= 0.03
            elif self.grid[y, x] == MOVABLE:
                reward -= 0.01  # redundant placement
            elif self.grid[y, x] == WALL:
                # Converting wall to movable
                self.grid[y, x] = MOVABLE
                reward += 0.01
                # Small bonus if on static path (shaping signal - terminal reward is dominant)
                if (y, x) in path_cells_static:
                    reward += 0.10
            else:  # EMPTY
                # Place movable on empty cell
                self.grid[y, x] = MOVABLE
                reward += 0.02
                # Small bonus if on static path (shaping signal)
                if (y, x) in path_cells_static:
                    reward += 0.10
                
        else:  # EMPTY
            if self.grid[y, x] == EMPTY:
                reward -= 0.01
            elif self.grid[y, x] == MOVABLE:
                # Removing a movable
                self.grid[y, x] = EMPTY
                reward -= 0.02
                # Small penalty if removing from path (shaping signal)
                if (y, x) in path_cells_static:
                    reward -= 0.03
            else:
                self.grid[y, x] = EMPTY

        changed = not np.array_equal(prev_grid, self.grid)
        if not changed:
            reward -= 0.02
        else:
            # Wall placement shaping (existing logic)
            if t == WALL and self.grid[y, x] == WALL:
                # neighbors
                H, W = self.h, self.w
                g = (self.grid == WALL).astype(np.int32)

                # count 4-neigh walls around (y,x)
                deg = 0
                for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
                    ny, nx = y+dy, x+dx
                    if 0 <= ny < H and 0 <= nx < W and g[ny, nx]:
                        deg += 1

                # favor corridor-like attachments (one or two neighbors, esp straight lines)
                if deg == 1:
                    reward += 0.03                 # start/extend a chain
                elif deg == 2:
                    # check if it's a straight segment (up-down XOR left-right)
                    straight = (0 < y < H-1 and g[y-1, x] and g[y+1, x]) or (0 < x < W-1 and g[y, x-1] and g[y, x+1])
                    reward += 0.035 if straight else 0.02

                # discourage isolated pixels and 2x2 blocks
                if deg == 0:
                    reward -= 0.03                 # isolated single wall is bad
                # 2x2 blob check (any of the four quarters)
                makes_2x2 = (
                        (y > 0 and x > 0 and g[y-1, x] and g[y, x-1] and g[y-1, x-1]) or
                        (y > 0 and x < W-1 and g[y-1, x] and g[y, x+1] and g[y-1, x+1]) or
                        (y < H-1 and x > 0 and g[y+1, x] and g[y, x-1] and g[y+1, x-1]) or
                        (y < H-1 and x < W-1 and g[y+1, x] and g[y, x+1] and g[y+1, x+1])
                )
                if makes_2x2:
                    reward -= 0.03
                reward += 0.03  # was 0.01
            
            # Movable placement shaping (similar to walls but using combined solid mask)
            if t == MOVABLE and self.grid[y, x] == MOVABLE:
                H, W = self.h, self.w
                # Use combined solid mask (walls + movables) for adjacency checks
                g_solid = ((self.grid == WALL) | (self.grid == MOVABLE)).astype(np.int32)
                
                deg = 0
                for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
                    ny, nx = y+dy, x+dx
                    if 0 <= ny < H and 0 <= nx < W and g_solid[ny, nx]:
                        deg += 1
                
                # Similar corridor shaping for movables
                if deg == 1:
                    reward += 0.03
                elif deg == 2:
                    straight = (0 < y < H-1 and g_solid[y-1, x] and g_solid[y+1, x]) or (0 < x < W-1 and g_solid[y, x-1] and g_solid[y, x+1])
                    reward += 0.035 if straight else 0.02
                if deg == 0:
                    reward -= 0.03
                makes_2x2 = (
                        (y > 0 and x > 0 and g_solid[y-1, x] and g_solid[y, x-1] and g_solid[y-1, x-1]) or
                        (y > 0 and x < W-1 and g_solid[y-1, x] and g_solid[y, x+1] and g_solid[y-1, x+1]) or
                        (y < H-1 and x > 0 and g_solid[y+1, x] and g_solid[y, x-1] and g_solid[y+1, x-1]) or
                        (y < H-1 and x < W-1 and g_solid[y+1, x] and g_solid[y, x+1] and g_solid[y+1, x+1])
                )
                if makes_2x2:
                    reward -= 0.03
                
                # Check if boxed (3+ neighbors solid)
                deg_solid = 0
                for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
                    ny, nx = y+dy, x+dx
                    if 0 <= ny < H and 0 <= nx < W:
                        if g_solid[ny, nx]: deg_solid += 1
                    else:
                        deg_solid += 1
                if deg_solid >= 3:
                    reward -= 0.05 # Penalty for placing a boxed movable
            
            # discourage deleting walls without reason (WALL -> EMPTY)
            if t == EMPTY and prev_grid[y, x] == WALL and self.grid[y, x] == EMPTY:
                reward -= 0.03

        # new-entity one-time bonuses (no farming)
        new_had_R = self._count(ROBOT) > 0
        new_had_O = self._count(OBJECT) > 0
        new_had_G = self._count(GOAL)  > 0
        if new_had_R and not had_R: reward += 0.25
        if new_had_O and not had_O: reward += 0.25
        if new_had_G and not had_G: reward += 0.25

        # coax toward some walls/obstacles once valid (using solid ratio)
        now_valid = self._valid_final()
        if now_valid:
            # Use dynamic wall target for coaxing
            # Only coax if we are below the current target
            wr = float(np.mean(self.grid == WALL))
            if wr < self._cur_wall_target:
                reward += self.wall_step_coax
            
            if t in (ROBOT, GOAL, OBJECT):
                reward -= 0.02

        # potential on L1+L2 and spread
        new_total = path_total()
        if prev_total is not None and new_total is not None:
            if new_total > prev_total:
                reward += 0.05
            elif new_total < prev_total:
                reward -= 0.02

        new_spread = spread_total()
        if prev_spread is not None and new_spread is not None:
            if new_spread > prev_spread:
                reward += 0.03
            elif new_spread < prev_spread:
                reward -= 0.01

        if (not prev_valid) and now_valid:
            reward += self.first_valid_bonus
        self._was_valid = now_valid

        # incremental corridor shaping (using combined solid mask)
        new_stats = self._wall_stats()
        if new_stats["n_solid"] > 0:
            new_adj_per_wall = new_stats["n_adj_pairs"] / float(new_stats["n_solid"])
            new_iso_frac = new_stats["n_isolated"] / float(new_stats["n_solid"])
            if new_adj_per_wall > prev_adj_per_wall:
                reward += self.lambda_corridor_step
            elif new_adj_per_wall < prev_adj_per_wall:
                reward -= 0.5 * self.lambda_corridor_step
            if new_iso_frac > prev_iso_frac:
                reward -= 0.5 * self.lambda_corridor_step

        reward -= self.step_cost

        # step / done
        self.steps += 1
        terminated = (self.steps >= self.max_steps)
        truncated = False
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
        chars = {EMPTY: ".", WALL: "#", ROBOT: "R", OBJECT: "O", GOAL: "G", MOVABLE: "M"}
        print("\n".join("".join(chars.get(int(self.grid[y, x]), "?") for x in range(self.w)) for y in range(self.h)))
