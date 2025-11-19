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

        self.steps = 0
        self.rng = np.random.RandomState(seed if seed is not None else 42)

        # -------- reward hyperparams (tuned for stability) --------
        self.alpha = 0.06       # weight on L1+L2 (path lengths)
        self.beta = 0.40       # wall-density penalty weight (terminal)
        self.gamma = 2.0        # adjacency (trivial) penalty
        self.delta = 0.5        # border penalty weight (terminal)

        # corridor / structure shaping
        self.lambda_corridor_term = 1.5    # terminal connectedness reward
        self.lambda_corridor_step = 0.25   # tiny incremental reward for more adjacency per wall
        self.lambda_isolated_term = 0.9    # terminal penalty for isolated walls
        self.lambda_block_term = 0.3    # terminal penalty for 2x2 wall blocks
        
        # movable obstacle reward shaping
        # movable obstacle reward shaping
        self.lambda_movable = 0.5  # bonus for having movable obstacles (terminal)
        self.movable_desired_count = 7.0  # target ~7 movables (range 4-10)
        
        # New movable-specific weights
        self.lambda_movable_on_path = 1.0  # Strong bonus for movable being on the static path
        self.lambda_movable_off_path = 0.1 # Penalty for movable NOT on path
        self.lambda_boxed = 0.2            # Penalty for boxed-in movables
        self.lambda_path_obstruction = 0.5 # Bonus if blocked path > static path

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
            "n_boxed": 0
        }

        if not self._valid_final():
            return -1.0, metrics

        ry, rx = self._pos(ROBOT)
        oy, ox = self._pos(OBJECT)
        gy, gx = self._pos(GOAL)

        # 1. Check STATIC solvability (ignoring movables)
        # This ensures no WALLS block the path.
        L1_static = shortest_path_len(self.grid, (ry, rx), (oy, ox), treat_movable_as_empty=True)
        L2_static = shortest_path_len(self.grid, (oy, ox), (gy, gx), treat_movable_as_empty=True)
        
        if L1_static is None or L2_static is None:
            # Unsolvable due to walls -> heavy penalty
            return -1.0, metrics

        # 2. Check BLOCKED solvability (respecting movables)
        # This tells us if movables are currently blocking the path (which is okay/good if pushable)
        L1_blocked = shortest_path_len(self.grid, (ry, rx), (oy, ox), treat_movable_as_empty=False)
        L2_blocked = shortest_path_len(self.grid, (oy, ox), (gy, gx), treat_movable_as_empty=False)
        
        # Note: L1_blocked might be None if movables block the path. That's allowed!
        # But we use L_static for the base path reward to encourage short *potential* paths.
        
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
        
        # STRICTLY target wall ratio (walls only) to force structure
        # Previous logic used solid_ratio which allowed movables to substitute for walls.
        wall_dev = abs(wr - self.wall_target)

        # small bonus if we land inside the [0.18, 0.32] band (using WALL ratio)
        band_bonus = 0.5 if (self._wall_band_lo <= wr <= self._wall_band_hi) else -0.5

        # LINEAR penalty away from target (V4: EXTREME wall enforcement)
        # If wall_ratio is 0.05 (dev 0.2), penalty is -1.0.
        wall_term = band_bonus - 5.0 * wall_dev

        # corridor quality
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

        # --- Movable Logic ---
        
        # A. Quantity Reward (Bell curve around target)
        # Target 7, sigma 2.5 gives good rewards for 4-10 range.
        movable_count_term = 0.0
        if n_movable > 0:
            if 4 <= n_movable <= 10:
                movable_count_term = self.lambda_movable # Max bonus in range
            else:
                # Linear penalty outside range
                diff = min(abs(n_movable - 4), abs(n_movable - 10))
                movable_count_term = -0.1 * diff
        
        # B. On-Path Reward
        # Identify cells on the STATIC shortest path
        path_cells = get_shortest_path_cells(self.grid, (ry, rx), (oy, ox), treat_movable_as_empty=True)
        path_cells |= get_shortest_path_cells(self.grid, (oy, ox), (gy, gx), treat_movable_as_empty=True)
        
        n_on_path = 0
        n_off_path = 0
        n_boxed = 0
        
        # Iterate movables to check path intersection and boxed status
        ys, xs = np.where(self.grid == MOVABLE)
        for my, mx in zip(ys, xs):
            if (my, mx) in path_cells:
                n_on_path += 1
            else:
                n_off_path += 1
            
            # Check if boxed (3+ neighbors are solid)
            deg = 0
            for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
                ny, nx = my+dy, mx+dx
                if 0 <= ny < self.h and 0 <= nx < self.w:
                    if self.grid[ny, nx] == WALL or self.grid[ny, nx] == MOVABLE:
                        deg += 1
                else:
                    deg += 1 # border is solid
            if deg >= 3:
                n_boxed += 1

        movable_quality_term = (
            + self.lambda_movable_on_path * n_on_path
            - self.lambda_movable_off_path * n_off_path
            - self.lambda_boxed * n_boxed
        )
        
        # C. Obstruction Bonus
        # If L_blocked > L_static (or None), it means movables are effectively increasing the path length
        # which implies they are "in the way" (good for a puzzle).
        obstruction_bonus = 0.0
        path_len_static = L1_static + L2_static
        path_len_blocked = (L1_blocked + L2_blocked) if (L1_blocked is not None and L2_blocked is not None) else float('inf')
        
        if path_len_blocked > path_len_static:
            obstruction_bonus = self.lambda_path_obstruction

        border_pen = self._border_penalty(margin=1)
        free_comps = self._free_space_components()
        free_space_pen = 0.15 * max(0, free_comps - 1)

        # Base reward uses STATIC path length (so we don't penalize the agent for blocking the path with movables)
        R = (0.5
             + self.alpha * path_len_static
             + wall_term
             + corridor_term
             + movable_count_term
             + movable_quality_term
             + obstruction_bonus
             - self.gamma * adj_trivial
             - self.delta * border_pen
             - free_space_pen)
             
        if ws["n_solid"] == 0:
            R -= 0.5

        metrics.update({
            "valid": 1, 
            "L1": int(L1_static), 
            "L2": int(L2_static),
            "wall_ratio": wr,
            "adj_per_wall": adj_per_wall,
            "iso_frac": iso_frac,
            "n_movable": n_movable,
            "movable_ratio": ws["movable_ratio"],
            "solid_ratio": solid_ratio,
            "n_movable_on_path": n_on_path,
            "n_boxed": n_boxed
        })
        return float(R), metrics

    # ---------- Gymnasium API ----------
    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None:
            self.rng.seed(seed)
        self.steps = 0
        self.grid.fill(EMPTY)

        # tiny curriculum: start with 0–2 entities already placed
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
        self._seed_walls_bootstrap()
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
            if self.grid[y, x] in (ROBOT, OBJECT, GOAL):
                reward -= 0.03
            elif self.grid[y, x] == MOVABLE:
                reward -= 0.01  # redundant
            elif self.grid[y, x] == WALL:
                # Converting wall to movable
                self.grid[y, x] = MOVABLE
                reward += 0.01
                # Bonus if this new movable is on the static path
                if (y, x) in path_cells_static:
                    reward += 0.08
                else:
                    reward -= 0.02 # Penalty for off-path
            else:  # EMPTY
                # Place movable on empty cell
                self.grid[y, x] = MOVABLE
                reward += 0.02
                # Strong bonus if placed on the static path
                if (y, x) in path_cells_static:
                    reward += 0.10
                else:
                    reward -= 0.03 # Penalty for off-path
                
        else:  # EMPTY
            if self.grid[y, x] == EMPTY:
                reward -= 0.01
            elif self.grid[y, x] == MOVABLE:
                # Removing a movable
                self.grid[y, x] = EMPTY
                reward -= 0.02
                # Penalty if removing from path (we want them there!)
                if (y, x) in path_cells_static:
                    reward -= 0.05
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
            # Use combined solid ratio (walls + movables) for coaxing
            solid_r = float(np.mean((self.grid == WALL) | (self.grid == MOVABLE)))
            reward += self.wall_step_coax * min(solid_r / self.wall_target, 1.0)
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
