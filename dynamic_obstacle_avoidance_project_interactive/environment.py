import math
import numpy as np

class Environment:
    """2-D occupancy grid containing editable static rectangular obstacles."""
    DEFAULT_STATIC_OBSTACLES = [
        (7, 7, 5, 15),
        (18, 0, 5, 13),
        (18, 19, 5, 12),
        (31, 7, 5, 14),
        (39, 0, 4, 11),
        (39, 19, 4, 14),
    ]

    def __init__(self, cfg, obstacles=None):
        self.cfg = cfg
        self.static_obstacles = [
            tuple(map(float, o)) for o in
            (obstacles if obstacles is not None else self.DEFAULT_STATIC_OBSTACLES)
        ]
        self._build_occupancy()

    def set_static_obstacles(self, obstacles):
        self.static_obstacles = [tuple(map(float, o)) for o in obstacles]
        self._build_occupancy()

    def reset_default_obstacles(self):
        self.set_static_obstacles(self.DEFAULT_STATIC_OBSTACLES)

    def _build_occupancy(self):
        self.occupancy = np.zeros((self.cfg.height, self.cfg.width), dtype=bool)
        for x0, y0, w, h in self.static_obstacles:
            x1 = min(self.cfg.width, int(math.ceil(x0 + w)))
            y1 = min(self.cfg.height, int(math.ceil(y0 + h)))
            xs = max(0, int(math.floor(x0)))
            ys = max(0, int(math.floor(y0)))
            if xs < x1 and ys < y1:
                self.occupancy[ys:y1, xs:x1] = True

    def in_bounds(self, x, y, margin=0.0):
        return margin <= x < self.cfg.width - margin and margin <= y < self.cfg.height - margin

    def is_static_collision(self, x, y, radius=0.0):
        if not self.in_bounds(x, y, radius):
            return True

        # Exact point-to-rectangle test first.
        for ox, oy, ow, oh in self.static_obstacles:
            cx = min(max(x, ox), ox + ow)
            cy = min(max(y, oy), oy + oh)
            if (cx - x) ** 2 + (cy - y) ** 2 <= radius ** 2:
                return True

        # Occupancy-cell fallback, preserving the behavior of the original project.
        r = int(math.ceil(radius / self.cfg.grid_resolution)) + 1
        gx, gy = int(x / self.cfg.grid_resolution), int(y / self.cfg.grid_resolution)
        for yy in range(max(0, gy-r), min(self.cfg.height, gy+r+1)):
            for xx in range(max(0, gx-r), min(self.cfg.width, gx+r+1)):
                if self.occupancy[yy, xx]:
                    cx, cy = xx + 0.5, yy + 0.5
                    if (cx-x)**2 + (cy-y)**2 <= (radius + 0.71)**2:
                        return True
        return False

    def point_free(self, x, y, radius):
        return not self.is_static_collision(x, y, radius)
