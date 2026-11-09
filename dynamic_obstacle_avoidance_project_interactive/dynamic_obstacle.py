import numpy as np

class DynamicObstacle:
    """Circular moving obstacle that bounces off world and static rectangular obstacles."""
    def __init__(self, start, velocity, radius=0.65, name="D1"):
        self.x, self.y = map(float, start)
        self.vx, self.vy = map(float, velocity)
        self.radius = float(radius)
        self.name = name
        self.initial = np.array([self.x, self.y], dtype=float)
        self.initial_velocity = np.array([self.vx, self.vy], dtype=float)
        self.t = 0.0
        self.bounces = 0

    @property
    def position(self):
        return np.array([self.x, self.y], dtype=float)

    @property
    def velocity(self):
        return np.array([self.vx, self.vy], dtype=float)

    def reset(self):
        self.x, self.y = self.initial
        self.vx, self.vy = self.initial_velocity
        self.t = 0.0
        self.bounces = 0

    @staticmethod
    def _circle_rect_collision(cx, cy, radius, rect):
        x0, y0, w, h = rect
        x1, y1 = x0 + w, y0 + h
        qx = min(max(cx, x0), x1)
        qy = min(max(cy, y0), y1)
        dx, dy = cx - qx, cy - qy
        return dx * dx + dy * dy <= radius * radius

    def _resolve_static_collision(self, old_pos, env):
        """Reflect velocity from the nearest intersected static-rectangle face."""
        collided = False
        reflect_x = False
        reflect_y = False
        eps = 1e-9

        for rect in env.static_obstacles:
            if not self._circle_rect_collision(self.x, self.y, self.radius, rect):
                continue

            collided = True
            x0, y0, w, h = rect
            x1, y1 = x0 + w, y0 + h

            # Compare distances to the four rectangle faces. This gives a stable
            # response for the common case of approaching a vertical or horizontal wall.
            dx_left = abs((self.x + self.radius) - x0)
            dx_right = abs(x1 - (self.x - self.radius))
            dy_bottom = abs((self.y + self.radius) - y0)
            dy_top = abs(y1 - (self.y - self.radius))

            # Prefer the axis on which the obstacle crossed the rectangle boundary.
            moved = self.position - np.asarray(old_pos)
            vertical_score = min(dx_left, dx_right)
            horizontal_score = min(dy_bottom, dy_top)

            if abs(moved[0]) > eps and vertical_score <= horizontal_score + 0.15:
                reflect_x = True
            elif abs(moved[1]) > eps:
                reflect_y = True
            else:
                # Corner/near-static fallback: reflect along the shallowest penetration.
                if vertical_score < horizontal_score:
                    reflect_x = True
                else:
                    reflect_y = True

            # Push the obstacle outside the rectangle so it does not remain embedded.
            qx = min(max(self.x, x0), x1)
            qy = min(max(self.y, y0), y1)
            nx, ny = self.x - qx, self.y - qy
            norm = float(np.hypot(nx, ny))
            if norm > eps:
                self.x += (nx / norm) * (self.radius - norm + 0.01)
                self.y += (ny / norm) * (self.radius - norm + 0.01)
            else:
                # Center ended up inside the rectangle: use the movement direction.
                self.x = min(max(self.x, x0 - self.radius - 0.01), x1 + self.radius + 0.01)
                self.y = min(max(self.y, y0 - self.radius - 0.01), y1 + self.radius + 0.01)

        if reflect_x:
            self.vx *= -1.0
        if reflect_y:
            self.vy *= -1.0
        if collided:
            self.bounces += int(reflect_x) + int(reflect_y)
        return collided

    def update(self, dt, env):
        self.t += dt
        old_pos = self.position.copy()
        self.x += self.vx * dt
        self.y += self.vy * dt

        # Reflect at world boundaries.
        if self.x - self.radius < 0.5 or self.x + self.radius > env.cfg.width - 0.5:
            self.vx *= -1.0
            self.x = min(env.cfg.width - 0.5 - self.radius, max(0.5 + self.radius, self.x))
            self.bounces += 1
        if self.y - self.radius < 0.5 or self.y + self.radius > env.cfg.height - 0.5:
            self.vy *= -1.0
            self.y = min(env.cfg.height - 0.5 - self.radius, max(0.5 + self.radius, self.y))
            self.bounces += 1

        self._resolve_static_collision(old_pos, env)
