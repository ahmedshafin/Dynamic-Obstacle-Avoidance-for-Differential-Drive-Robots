import math
import numpy as np


class DynamicObstacleDetector:
    def __init__(self, cfg):
        self.cfg = cfg

    def _ttc(self, robot_pos, robot_vel, obs_pos, obs_vel, safe):
        r = np.asarray(obs_pos, dtype=float) - np.asarray(robot_pos, dtype=float)
        v = np.asarray(obs_vel, dtype=float) - np.asarray(robot_vel, dtype=float)
        a = float(np.dot(v, v))
        b = 2.0 * float(np.dot(r, v))
        c = float(np.dot(r, r) - safe * safe)
        if c <= 0:
            return 0.0
        if a < 1e-9:
            return math.inf
        disc = b * b - 4.0 * a * c
        if disc < 0:
            return math.inf
        s = math.sqrt(disc)
        candidates = [t for t in ((-b - s) / (2 * a), (-b + s) / (2 * a)) if t >= 0]
        return min(candidates) if candidates else math.inf

    def detect(self, robot, obstacles, global_path, current_index, active_path=None):
        rp = np.array([robot.x, robot.y], dtype=float)
        rv = np.array([robot.v * math.cos(robot.theta), robot.v * math.sin(robot.theta)])
        ahead_global = global_path[current_index:min(len(global_path), current_index + 120)]
        ahead_active = active_path if active_path is not None else ahead_global
        threats = []

        for idx, obstacle in enumerate(obstacles):
            op = obstacle.position
            dist = float(np.linalg.norm(op - rp))
            safe = self.cfg.robot_collision_radius + obstacle.radius + self.cfg.safety_margin
            ttc = self._ttc(rp, rv, op, obstacle.velocity, safe)

            path_risk = False
            min_path_dist = math.inf
            # Scan both the global route and currently active local route.
            for route in (ahead_global, ahead_active):
                if len(route) == 0:
                    continue
                for k, p in enumerate(route[:130]):
                    t = min(self.cfg.prediction_horizon, 0.15 + k * self.cfg.dt)
                    pred = op + obstacle.velocity * t
                    dd = float(np.linalg.norm(np.asarray(p) - pred))
                    min_path_dist = min(min_path_dist, dd)
                    if dd < safe + 0.55:
                        path_risk = True
                        break
                if path_risk:
                    break

            trigger = dist <= self.cfg.sensor_radius and (
                ttc <= self.cfg.prediction_horizon or
                path_risk or
                dist <= self.cfg.robot_radius + obstacle.radius + self.cfg.safety_margin
            )
            if trigger:
                threats.append({
                    'index': idx,
                    'obstacle': obstacle,
                    'distance': dist,
                    'ttc': ttc,
                    'min_path_distance': min_path_dist,
                    'reason': 'predicted collision' if ttc < math.inf else 'path occupied',
                })

        threats.sort(key=lambda item: (item['ttc'], item['distance']))
        nearest = threats[0] if threats else None
        info = {
            'distance': nearest['distance'] if nearest else math.inf,
            'ttc': nearest['ttc'] if nearest else math.inf,
            'min_path_distance': nearest['min_path_distance'] if nearest else math.inf,
            'reason': nearest['reason'] if nearest else 'no collision predicted',
            'obstacle_index': nearest['index'] if nearest else -1,
            'threats': threats,
        }
        return bool(threats), info
