import math
import numpy as np


class LocalReplanner:
    """Bounded local A* detour planner with time-aware moving-obstacle validation.

    A normal grid A* is spatial, not space-time. Blocking every future position of
    every moving obstacle made the old planner overly conservative and caused WAIT
    deadlocks. This version plans around the obstacles' *current* inflated positions,
    then checks the resulting route against each obstacle's predicted motion.
    """

    def __init__(self, env, astar, smoother, cfg):
        self.env, self.astar, self.smoother, self.cfg = env, astar, smoother, cfg

    def _nearest_global(self, pos, global_path, start=0):
        arr = np.asarray(global_path[start:], dtype=float)
        if len(arr) == 0:
            return start
        d2 = np.sum((arr - np.asarray(pos, dtype=float)) ** 2, axis=1)
        return int(np.argmin(d2)) + start

    def _current_dynamic_blockers(self, obstacles, ignore_names=(), robot_pos=None):
        """Block nearby obstacles at their current positions; distant obstacles do not poison the local window."""
        blocked = []
        ignore = set(ignore_names)
        if robot_pos is None:
            robot_pos = np.zeros(2)
        robot_pos = np.asarray(robot_pos, dtype=float)
        max_dist = self.cfg.local_window * 0.78
        for o in obstacles:
            if o.name in ignore:
                continue
            if np.linalg.norm(o.position - robot_pos) > max_dist:
                continue
            r = self.cfg.robot_radius + o.radius + self.cfg.local_dynamic_inflation
            blocked.append((float(o.x), float(o.y), float(r)))
        return blocked

    def _candidate_targets(self, global_path, nearest, threat_obstacles, robot_pos):
        n = len(global_path)
        targets = []

        for offset in range(self.cfg.local_target_min_ahead,
                            min(self.cfg.local_target_max_ahead, n - nearest - 1) + 1,
                            self.cfg.local_target_step):
            idx = min(n - 1, nearest + offset)
            x, y = map(float, global_path[idx])
            if self.env.point_free(x, y, self.cfg.robot_radius + 0.18):
                targets.append((idx, np.array([x, y], dtype=float), 'global', ()))

        # First priority: move away from the currently threatening obstacle.
        if threat_obstacles:
            threat = threat_obstacles[0]['obstacle']
            away = np.asarray(robot_pos, dtype=float) - threat.position
            nrm = float(np.linalg.norm(away))
            if nrm < 1e-9:
                away = np.array([1.0, 0.0])
            else:
                away /= nrm
            # Try several escape distances and small left/right rotations.
            for dist in (2.4, 3.2, 4.0):
                for angle in (-0.45, 0.0, 0.45):
                    c, s = math.cos(angle), math.sin(angle)
                    vec = np.array([c*away[0]-s*away[1], s*away[0]+c*away[1]])
                    wp = np.asarray(robot_pos, dtype=float) + vec * dist
                    if self.env.point_free(float(wp[0]), float(wp[1]), self.cfg.robot_radius + 0.12):
                        ridx = min(n - 1, nearest + max(14, self.cfg.local_target_min_ahead))
                        targets.insert(0, (ridx, wp, 'escape', (threat.name,)))

            # Also try route-normal offsets around the threat.
            route_i = self._nearest_global(threat.position, global_path, nearest)
            p = np.asarray(global_path[route_i], dtype=float)
            p_prev = np.asarray(global_path[max(nearest, route_i - 3)], dtype=float)
            tangent = p - p_prev
            tn = float(np.linalg.norm(tangent))
            tangent = tangent / tn if tn > 1e-9 else np.array([1.0, 0.0])
            normal = np.array([-tangent[1], tangent[0]])
            for side in (1.0, -1.0):
                for lateral in (2.2, 3.0):
                    wp = p + normal * side * lateral + tangent * 1.0
                    if self.env.point_free(float(wp[0]), float(wp[1]), self.cfg.robot_radius + 0.12):
                        ridx = min(n - 1, max(route_i + 16, nearest + self.cfg.local_target_min_ahead))
                        targets.insert(0, (ridx, wp, 'escape', (threat.name,)))

        unique = []
        seen = set()
        for idx, p, kind, ignore in targets:
            key = (int(idx), round(float(p[0]), 1), round(float(p[1]), 1), tuple(ignore))
            if key not in seen:
                seen.add(key)
                unique.append((idx, p, kind, ignore))
        return unique

    @staticmethod
    def _path_length(path):
        if len(path) < 2:
            return 0.0
        return float(np.sum(np.linalg.norm(np.diff(np.asarray(path), axis=0), axis=1)))

    def _time_safe(self, path, obstacles):
        """Check dynamic collision risk using estimated arrival time along the route."""
        if path is None or len(path) < 2:
            return False
        pts = np.asarray(path, dtype=float)
        safe = self.cfg.robot_collision_radius + self.cfg.dynamic_obstacle_radius + self.cfg.safety_margin
        speed = max(0.55, min(0.95, self.cfg.max_linear_speed))

        cumulative = np.zeros(len(pts))
        cumulative[1:] = np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))
        # Account for the robot initially rotating/slowing near the first waypoint.
        eta = 0.45 + cumulative / speed

        for o in obstacles:
            for p, t in zip(pts[::2], eta[::2]):
                if t > self.cfg.local_prediction_horizon + 2.0:
                    break
                pred = o.position + o.velocity * min(t, self.cfg.local_prediction_horizon + 1.5)
                clearance = safe + 0.10 * min(t, 3.0)
                if float(np.linalg.norm(p - pred)) < clearance:
                    return False
        return True

    def _static_safe(self, path):
        pts = np.asarray(path, dtype=float)
        return all(self.env.point_free(float(p[0]), float(p[1]), self.cfg.robot_radius + 0.10) for p in pts)

    def _astar_segment(self, start, goal, obstacles, ignore_names=(), bounds=None):
        blocked = self._current_dynamic_blockers(obstacles, ignore_names=ignore_names, robot_pos=start)
        return self.astar.plan(
            tuple(start), tuple(goal), extra_blocked=blocked, bounds=bounds,
            robot_radius=self.cfg.robot_radius + 0.06
        )

    def _combine_to_rejoin(self, first_path, waypoint, rejoin):
        second = self.astar.plan(
            tuple(waypoint), tuple(rejoin),
            extra_blocked=[],
            bounds=None,
            robot_radius=self.cfg.robot_radius + 0.06,
        )
        if not second:
            return None
        return list(first_path) + list(second[1:])

    def plan(self, robot, obstacles, global_path, current_global_index, threat_info=None):
        nearest = self._nearest_global((robot.x, robot.y), global_path, current_global_index)
        half = int(self.cfg.local_window // 2)
        sx, sy = int(round(robot.x)), int(round(robot.y))
        xmin = max(0, sx - half)
        xmax = min(self.env.cfg.width - 1, sx + half)
        ymin = max(0, sy - half)
        ymax = min(self.env.cfg.height - 1, sy + half)
        bounds = (xmin, xmax, ymin, ymax)

        threats = (threat_info or {}).get('threats', [])
        candidates = self._candidate_targets(global_path, nearest, threats, (robot.x, robot.y))
        if not candidates:
            return None

        # Try escape waypoints first, then ordinary forward global rejoin points.
        for rejoin_index, target, kind, ignore_names in candidates:
            # Escape target can be off the global route: route to it in the bounded window,
            # then route from it to the global rejoin point in the same local window.
            if kind == 'escape':
                stage1 = self._astar_segment((robot.x, robot.y), target, obstacles, ignore_names=ignore_names, bounds=bounds)
                if not stage1:
                    continue
                stage2 = self._astar_segment(target, global_path[rejoin_index], obstacles, ignore_names=(), bounds=bounds)
                if not stage2:
                    continue
                raw = list(stage1) + list(stage2[1:])
            else:
                raw = self._astar_segment((robot.x, robot.y), target, obstacles, ignore_names=ignore_names, bounds=bounds)
                if not raw:
                    continue

            if len(raw) < 3:
                continue
            smooth = self.smoother.smooth(raw, samples_per_segment=6)
            if len(smooth) < 3:
                continue
            if not self._static_safe(smooth):
                continue
            if not self._time_safe(smooth, obstacles):
                continue

            # Require an actual lateral departure from the original global segment.
            if kind == 'escape':
                detour_measure = 2.0
            else:
                global_segment = np.asarray(global_path[nearest:rejoin_index + 1], dtype=float)
                detour_measure = 0.0
                if len(global_segment) >= 2:
                    for p in smooth:
                        detour_measure = max(
                            detour_measure,
                            float(np.min(np.linalg.norm(global_segment - p, axis=1)))
                        )

            if detour_measure < self.cfg.minimum_detour_offset and kind != 'escape':
                continue

            return {
                'path': smooth,
                'rejoin_index': int(rejoin_index),
                'global_nearest': int(nearest),
                'detour_measure': float(detour_measure),
            }

        return None
