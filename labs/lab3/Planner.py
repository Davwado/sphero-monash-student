import numpy as np
import heapq

class Planner:
    def __init__(self, map, dt=0.1, resolution=0.125):
        """
        map : 2D numpy array from build_occupancy_grid().
              1 = wall, 0 = free. Logical cells sit at ODD (row, col)
              indices; the cell halfway between two logical neighbours
              (even index) is the connector - free unless walled off.
              Cells where BOTH row and col are even are structural filler
              positions that are never explicitly opened - always 1,
              regardless of the actual maze layout.
        resolution : metres per occupancy cell (0.125 = half a 0.25m plate,
                     matching grid_resolution passed to SpheroEnv).
        """
        self.map = map
        self.dt = dt
        self.resolution = resolution
        self.occ_h, self.occ_w = map.shape

        # occ_dim = 2*MAZE_dim + 1  =>  MAZE_dim = (occ_dim - 1) // 2
        self.maze_w = (self.occ_w - 1) // 2
        self.maze_h = (self.occ_h - 1) // 2

    # ---------------------------------------------------------- conversion --
    # world_x = (col - maze_w) * resolution
    # world_y = (maze_h - row) * resolution

    def world_to_occ(self, xy):
        x, y = xy
        col = int(round(x / self.resolution)) + self.maze_w
        row = self.maze_h - int(round(y / self.resolution))
        row = int(np.clip(row, 0, self.occ_h - 1))
        col = int(np.clip(col, 0, self.occ_w - 1))
        return (row, col)

    def occ_to_world(self, cell):
        row, col = cell
        x = (col - self.maze_w) * self.resolution
        y = (self.maze_h - row) * self.resolution
        return np.array([x, y], dtype=np.float32)

    # -------------------------------------------------------- wall inflation --

    def inflate_walls(self, margin_cells=1):
        """
        Dilate real wall cells by margin_cells, axis-aligned only (matching
        4-connected movement - no point inflating diagonally when the
        robot can't move diagonally anyway).

        Two kinds of cells are excluded as inflation SOURCES:
        - outer border cells: represent the edge of the grid's bounding
          box, not a real obstacle to keep clearance from
        - (even, even) cells: structural filler positions that are always
          1 regardless of the actual maze layout - inflating from them
          produces false blocks on legitimate open cells nearby
        """
        if margin_cells <= 0:
            return self.map.copy()

        inflated = self.map.copy()
        wall_rows, wall_cols = np.where(self.map == 1)

        for r, c in zip(wall_rows, wall_cols):
            is_border = (r == 0 or r == self.occ_h - 1 or
                         c == 0 or c == self.occ_w - 1)
            is_corner_filler = (r % 2 == 0 and c % 2 == 0)
            if is_border or is_corner_filler:
                continue

            for dr in range(-margin_cells, margin_cells + 1):
                nr = r + dr
                if 0 <= nr < self.occ_h:
                    inflated[nr, c] = 1
            for dc in range(-margin_cells, margin_cells + 1):
                nc = c + dc
                if 0 <= nc < self.occ_w:
                    inflated[r, nc] = 1

        return inflated

    # ------------------------------------------------------- cell snapping --

    def find_nearest_free(self, cell, max_radius=5):
        """
        If cell lands on a wall (e.g. rounding put it on a structural
        filler position rather than a real logical/connector cell), search
        outward in expanding rings for the nearest actually-free cell.
        Returns None if nothing free is found within max_radius.
        """
        if self.map[cell] == 0:
            return cell

        r0, c0 = cell
        for radius in range(1, max_radius + 1):
            candidates = []
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    if max(abs(dr), abs(dc)) != radius:
                        continue
                    nr, nc = r0 + dr, c0 + dc
                    if 0 <= nr < self.occ_h and 0 <= nc < self.occ_w:
                        if self.map[nr, nc] == 0:
                            candidates.append((nr, nc))
            if candidates:
                candidates.sort(key=lambda c: (c[0]-r0)**2 + (c[1]-c0)**2)
                return candidates[0]

        return None

    # ------------------------------------------------------------- A* core --
    # 4-connected only: this grid has no connector cell for a diagonal move,
    # so diagonals aren't a valid transition here.

    def _heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _neighbors(self, map_to_use, cell):
        r, c = cell
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < self.occ_h and 0 <= nc < self.occ_w):
                continue
            if map_to_use[nr, nc] == 1:
                continue
            yield (nr, nc)

    def _astar(self, map_to_use, start_cell, goal_cell):
        open_set = [(0, start_cell)]
        came_from = {}
        g_score = {start_cell: 0}

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal_cell:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return path[::-1]

            for neighbor in self._neighbors(map_to_use, current):
                tentative_g = g_score[current] + 1
                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self._heuristic(neighbor, goal_cell)
                    heapq.heappush(open_set, (f, neighbor))

        return None

    def _thin(self, waypoints):
        """Keep only points where direction changes."""
        if len(waypoints) <= 2:
            return waypoints
        thinned = [waypoints[0]]
        for i in range(1, len(waypoints) - 1):
            prev_dir = waypoints[i] - waypoints[i - 1]
            next_dir = waypoints[i + 1] - waypoints[i]
            n1, n2 = np.linalg.norm(prev_dir), np.linalg.norm(next_dir)
            if n1 < 1e-9 or n2 < 1e-9:
                continue
            if not np.allclose(prev_dir / n1, next_dir / n2, atol=1e-6):
                thinned.append(waypoints[i])
        thinned.append(waypoints[-1])
        return thinned

    # ------------------------------------------------------------- public --

    def plan(self, state, goal, margin_cells=0):
        """
        state : [x, y, heading, speed]
        goal  : [x, y]
        Returns: list of np.array([x, y]) waypoints, start to goal.
        """
        search_map = self.inflate_walls(margin_cells=margin_cells)

        start_cell = self.world_to_occ((state[0], state[1]))
        goal_cell = self.world_to_occ((goal[0], goal[1]))

        if search_map[start_cell] == 1:
            snapped = self.find_nearest_free(start_cell)
            if snapped is None:
                raise ValueError(f"Start occ cell {start_cell} is a wall and no "
                                  f"free cell found nearby")
            start_cell = snapped

        if search_map[goal_cell] == 1:
            snapped = self.find_nearest_free(goal_cell)
            if snapped is None:
                raise ValueError(f"Goal occ cell {goal_cell} is a wall and no "
                                  f"free cell found nearby")
            goal_cell = snapped

        path_cells = self._astar(search_map, start_cell, goal_cell)
        if path_cells is None:
            raise RuntimeError(f"No path found from {start_cell} to {goal_cell} "
                                f"(margin_cells={margin_cells})")

        logical_cells = [c for c in path_cells if c[0] % 2 == 1 and c[1] % 2 == 1]
        waypoints_world = [self.occ_to_world(c) for c in logical_cells]
        waypoints = self._thin(waypoints_world)

        return waypoints
