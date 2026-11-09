import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button

from config import Config
from environment import Environment
from astar import AStarPlanner
from smoothing import KinodynamicSmoother
from kinematics import DifferentialDriveRobot
from dynamic_obstacle import DynamicObstacle
from detector import DynamicObstacleDetector
from local_replanner import LocalReplanner
from tracker import PurePursuitTracker

class Simulation:
    def __init__(self):
        self.cfg = Config()
        self.env = Environment(self.cfg)
        self.astar = AStarPlanner(self.env)
        self.smoother = KinodynamicSmoother(self.env, self.cfg)
        self.detector = DynamicObstacleDetector(self.cfg)
        self.local = LocalReplanner(self.env, self.astar, self.smoother, self.cfg)
        self.tracker = PurePursuitTracker(self.cfg)
        self.start = (2.0, 4.0)
        self.goal = (47.0, 45.0)
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.paused = False
        self.editing = False
        self.edit_mode = 'drag'
        self.drag_index = None
        self.drag_offset = (0.0, 0.0)
        self.pending_obstacles = [tuple(o) for o in self.env.static_obstacles]
        self.reset()
        self._setup_plot()


    # ------------------------------------------------------------------
    # Interactive static-obstacle editor
    # ------------------------------------------------------------------
    def _set_edit_mode(self, mode):
        self.edit_mode = mode
        self.editing = True
        self.paused = True
        self.drag_index = None
        label = {
            'drag': 'DRAG: left-click + drag a rectangle',
            'add': 'ADD: click an empty area',
            'remove': 'REMOVE: click a rectangle',
        }[mode]
        self.edit_status.set_text(
            f'EDIT MODE — {label} | Click Apply Layout when finished.'
        )
        self.fig.canvas.draw_idle()

    def _inside_world(self, x, y):
        return (
            x is not None and y is not None and
            0 <= x <= self.cfg.width and 0 <= y <= self.cfg.height
        )

    def _obstacle_at(self, x, y):
        for i in range(len(self.pending_obstacles) - 1, -1, -1):
            ox, oy, w, h = self.pending_obstacles[i]
            if ox <= x <= ox + w and oy <= y <= oy + h:
                return i
        return None

    def _layout_valid(self, obstacles):
        margin = self.cfg.robot_collision_radius + 0.05
        for ox, oy, w, h in obstacles:
            if w <= 0 or h <= 0:
                return False
            if ox < 0 or oy < 0 or ox + w > self.cfg.width or oy + h > self.cfg.height:
                return False
            for px, py in (self.start, self.goal):
                if (ox - margin <= px <= ox + w + margin and
                        oy - margin <= py <= oy + h + margin):
                    return False
        for i in range(len(obstacles)):
            a = obstacles[i]
            for j in range(i + 1, len(obstacles)):
                b = obstacles[j]
                overlap = (
                    a[0] < b[0] + b[2] and a[0] + a[2] > b[0] and
                    a[1] < b[1] + b[3] and a[1] + a[3] > b[1]
                )
                if overlap:
                    return False
        return True

    def _sync_static_patches(self):
        for patch in getattr(self, 'static_patches', []):
            patch.remove()
        self.static_patches = []
        for x, y, w, h in self.pending_obstacles:
            patch = Rectangle((x, y), w, h, fill=True, alpha=0.55)
            self.ax.add_patch(patch)
            self.static_patches.append(patch)

    def _mouse_press(self, event):
        if event.inaxes is not self.ax or not self.editing:
            return
        if not self._inside_world(event.xdata, event.ydata):
            return
        x, y = float(event.xdata), float(event.ydata)
        idx = self._obstacle_at(x, y)

        if self.edit_mode == 'add' and idx is None:
            # New static obstacle: 5m x 9m rectangle centered at the click.
            w, h = 5.0, 9.0
            candidate = (x - w / 2.0, y - h / 2.0, w, h)
            if self._layout_valid(self.pending_obstacles + [candidate]):
                self.pending_obstacles.append(candidate)
                self._sync_static_patches()
                self.edit_status.set_text(
                    'Added obstacle. Continue editing or click Apply Layout.'
                )
            else:
                self.edit_status.set_text(
                    'Cannot add here: out of bounds, overlaps another obstacle, or blocks A/B.'
                )
            self.fig.canvas.draw_idle()
            return

        if self.edit_mode == 'remove' and idx is not None:
            del self.pending_obstacles[idx]
            self._sync_static_patches()
            self.edit_status.set_text(
                'Removed obstacle. Continue editing or click Apply Layout.'
            )
            self.fig.canvas.draw_idle()
            return

        if self.edit_mode == 'drag' and idx is not None:
            ox, oy, w, h = self.pending_obstacles[idx]
            self.drag_index = idx
            self.drag_offset = (x - ox, y - oy)
            self.edit_status.set_text('Dragging obstacle — release mouse to place it.')

    def _mouse_move(self, event):
        if not self.editing or self.drag_index is None or event.inaxes is not self.ax:
            return
        if not self._inside_world(event.xdata, event.ydata):
            return
        x, y = float(event.xdata), float(event.ydata)
        idx = self.drag_index
        _, _, w, h = self.pending_obstacles[idx]
        ox = max(0.0, min(self.cfg.width - w, x - self.drag_offset[0]))
        oy = max(0.0, min(self.cfg.height - h, y - self.drag_offset[1]))
        candidate = list(self.pending_obstacles)
        candidate[idx] = (ox, oy, w, h)
        if self._layout_valid(candidate):
            self.pending_obstacles[idx] = candidate[idx]
            self.static_patches[idx].set_xy((ox, oy))
            self.fig.canvas.draw_idle()

    def _mouse_release(self, event):
        self.drag_index = None
        if self.editing:
            self.edit_status.set_text(
                'EDIT MODE — click Apply Layout to recalculate A* and restart the test.'
            )
            self.fig.canvas.draw_idle()

    def _apply_layout(self, event=None):
        if not self._layout_valid(self.pending_obstacles):
            self.edit_status.set_text(
                'Layout invalid. Move/remove obstacles so they do not overlap or block A/B.'
            )
            return
        # Rebuild the occupancy grid + all planners, then restart from A.
        self.env.set_static_obstacles(self.pending_obstacles)
        self.astar = AStarPlanner(self.env)
        self.smoother = KinodynamicSmoother(self.env, self.cfg)
        self.local = LocalReplanner(self.env, self.astar, self.smoother, self.cfg)
        self._sync_static_patches()
        try:
            self.reset()
        except RuntimeError as exc:
            self.edit_status.set_text(f'A* failed for this layout: {exc}')
            return
        self.editing = False
        self.paused = False
        self.edit_status.set_text(
            'Layout applied. Simulation restarted with the new static-obstacle map.'
        )
        self.fig.canvas.draw_idle()

    def _restore_default_layout(self, event=None):
        self.pending_obstacles = [tuple(o) for o in Environment.DEFAULT_STATIC_OBSTACLES]
        self._sync_static_patches()
        self._apply_layout()

    def _spawn_obstacles(self):
        """
        Spawn five dynamic obstacles at different points ahead of the robot.

        Each obstacle starts in free space to the side of a different future
        point on the smoothed global route and travels toward that route point.
        The candidate start position is checked against the static map so no
        obstacle starts embedded inside a wall.
        """
        path = np.asarray(self.global_path, dtype=float)

        target_indices = [80, 140, 200, 260, 320]
        preferred_sides = [1, -1, 1, -1, 1]
        candidate_offsets = [4.8, 4.2, 5.6, 3.8, 6.2]

        specs = []
        for i, (idx, preferred_side, base_offset) in enumerate(
            zip(target_indices, preferred_sides, candidate_offsets), start=1
        ):
            idx = min(idx, len(path) - 2)
            p = path[idx]
            p_prev = path[max(0, idx - 2)]

            direction = p - p_prev
            norm = float(np.linalg.norm(direction))
            if norm < 1e-9:
                direction = np.array([1.0, 0.0])
            else:
                direction = direction / norm

            normal = np.array([-direction[1], direction[0]])

            # Find the first free point on the preferred side; fall back to the
            # opposite side if that side is occupied by a static obstacle.
            candidates = []
            for mult in (1.0, 1.25, 1.5, 1.8):
                candidates.append(p + normal * preferred_side * base_offset * mult)
                candidates.append(p - normal * preferred_side * base_offset * mult)

            start = None
            for candidate in candidates:
                x, y = float(candidate[0]), float(candidate[1])
                if (
                    1.2 < x < self.cfg.width - 1.2
                    and 1.2 < y < self.cfg.height - 1.2
                    and self.env.point_free(x, y, self.cfg.dynamic_obstacle_radius + 0.10)
                ):
                    start = candidate
                    break

            if start is None:
                raise RuntimeError(f"Could not find a free spawn position for D{i}.")

            # Move toward the selected future global-path point with a small
            # along-path component. This makes the obstacle cross the main route.
            cross = p - start
            cross_norm = float(np.linalg.norm(cross))
            if cross_norm > 1e-9:
                cross = cross / cross_norm

            velocity = 0.90 * cross + 0.20 * direction
            specs.append((tuple(start), tuple(velocity), f"D{i}"))

        return [
            DynamicObstacle(
                start=s,
                velocity=v,
                radius=self.cfg.dynamic_obstacle_radius,
                name=n
            )
            for s, v, n in specs[:self.cfg.dynamic_obstacle_count]
        ]

    def reset(self):
        raw = self.astar.plan(self.start, self.goal, robot_radius=self.cfg.robot_radius + 0.18)
        if not raw:
            raise RuntimeError('Global A* failed. Change the static obstacle layout.')
        self.global_path = self.smoother.smooth(raw)
        self.global_speed = self.smoother.speed_profile(self.global_path)
        theta = math.atan2(self.global_path[1, 1] - self.global_path[0, 1],
                           self.global_path[1, 0] - self.global_path[0, 0])
        self.robot = DifferentialDriveRobot(self.cfg, (self.start[0], self.start[1], theta))
        self.obstacles = self._spawn_obstacles()
        self.mode = 'GLOBAL'
        self.local_path = None
        self.local_speed = None
        self.local_rejoin_index = self.global_index = 0
        self.last_local_replan_time = -math.inf
        self.time = 0.0
        self.replans = 0
        self.collisions = 0
        self.active_contacts = set()
        self.replan_timer = 0.0
        self.last_detection = {
            'distance': math.inf, 'ttc': math.inf, 'min_path_distance': math.inf,
            'reason': 'none', 'obstacle_index': -1, 'threats': []
        }
        self.history = []
        self.obs_history = [[] for _ in self.obstacles]
        self.event = 'Following global A* path'

    def _setup_plot(self):
        self.ax.set_xlim(0, self.cfg.width)
        self.ax.set_ylim(0, self.cfg.height)
        self.ax.set_aspect('equal')
        self.ax.set_title('Dynamic Obstacle Avoidance — 5 Dynamic Obstacles + Bouncing + Adaptive Local A*')
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.grid(alpha=0.15)
        self.ax.plot(*self.start, 'o', markersize=10, label='Start A')
        self.ax.plot(*self.goal, '*', markersize=13, label='Goal B')
        self.global_line, = self.ax.plot([], [], '--', lw=1.5, label='Global A* + smooth')
        self.active_line, = self.ax.plot([], [], lw=3, label='Active route')

        # Robot avatar: a small oriented differential-drive robot instead of a circle.
        self.robot_body = Polygon(np.zeros((5, 2)), closed=True, facecolor='none',
                                  edgecolor='black', lw=1.5, alpha=0.95, label='Robot')
        self.ax.add_patch(self.robot_body)
        self.robot_wheel_left, = self.ax.plot([], [], lw=4, solid_capstyle='round')
        self.robot_wheel_right, = self.ax.plot([], [], lw=4, solid_capstyle='round')
        self.robot_center, = self.ax.plot([], [], 'o', markersize=3)
        self.robot_eye, = self.ax.plot([], [], 'o', markersize=4)

        # Interactive static-obstacle editor.
        self.edit_status = self.ax.text(0.02, 0.02, '', transform=self.ax.transAxes,
                                        fontsize=8, va='bottom',
                                        bbox=dict(boxstyle='round', alpha=0.75))
        self.btn_drag = Button(self.fig.add_axes([0.64, 0.915, 0.09, 0.045]), 'Drag')
        self.btn_add = Button(self.fig.add_axes([0.735, 0.915, 0.09, 0.045]), 'Add')
        self.btn_remove = Button(self.fig.add_axes([0.83, 0.915, 0.09, 0.045]), 'Remove')
        self.btn_apply = Button(self.fig.add_axes([0.64, 0.86, 0.135, 0.045]), 'Apply Layout')
        self.btn_default = Button(self.fig.add_axes([0.785, 0.86, 0.135, 0.045]), 'Default Layout')
        self.btn_drag.on_clicked(lambda event: self._set_edit_mode('drag'))
        self.btn_add.on_clicked(lambda event: self._set_edit_mode('add'))
        self.btn_remove.on_clicked(lambda event: self._set_edit_mode('remove'))
        self.btn_apply.on_clicked(self._apply_layout)
        self.btn_default.on_clicked(self._restore_default_layout)
        self.fig.canvas.mpl_connect('button_press_event', self._mouse_press)
        self.fig.canvas.mpl_connect('motion_notify_event', self._mouse_move)
        self.fig.canvas.mpl_connect('button_release_event', self._mouse_release)

        self.static_patches = []
        self._sync_static_patches()
        self.dynamic_patches = []
        self.obstacle_heading = []
        for obstacle in self.obstacles:
            patch = Circle(tuple(obstacle.position), obstacle.radius, fill=True, alpha=0.7)
            self.ax.add_patch(patch)
            self.dynamic_patches.append(patch)
            line, = self.ax.plot([], [], lw=1.5)
            self.obstacle_heading.append(line)

        self.sensor_patch = Circle((0, 0), self.cfg.sensor_radius, fill=False, ls=':', lw=1.0, alpha=0.35)
        self.ax.add_patch(self.sensor_patch)
        self.status = self.ax.text(
            0.02, 0.98, '', transform=self.ax.transAxes, va='top', fontsize=9,
            bbox=dict(boxstyle='round', alpha=0.8)
        )
        self.ax.legend(loc='lower right', fontsize=8)

    def _global_index(self):
        pos = np.array([self.robot.x, self.robot.y])
        p = self.global_path
        begin = max(0, self.global_index - 10)
        end = min(len(p), self.global_index + 100)
        i = begin + int(np.argmin(np.sum((p[begin:end] - pos) ** 2, axis=1)))
        self.global_index = max(self.global_index, i)
        return self.global_index

    def _min_dynamic_distance(self):
        pos = np.array([self.robot.x, self.robot.y], dtype=float)
        return min(float(np.linalg.norm(pos - o.position)) for o in self.obstacles)

    def _global_route_clear(self, start_index):
        """Check whether the upcoming original route is actually collision-free."""
        route = self.global_path[start_index:min(len(self.global_path), start_index + 100)]
        if len(route) == 0:
            return True
        for obstacle in self.obstacles:
            for k, p in enumerate(route):
                t = min(self.cfg.prediction_horizon, 0.2 + k * self.cfg.dt)
                pred = obstacle.position + obstacle.velocity * t
                safe = self.cfg.robot_collision_radius + obstacle.radius + self.cfg.rejoin_clearance
                if np.linalg.norm(np.asarray(p) - pred) < safe:
                    return False
        return True

    def _do_local_replan(self, info, reason):
        """Perform an immediate local A* replanning request."""
        gi = self._global_index()
        result = self.local.plan(self.robot, self.obstacles, self.global_path, gi, threat_info=info)
        if result:
            self.local_path = result['path']
            self.local_speed = self.smoother.speed_profile(self.local_path)
            self.local_rejoin_index = result['rejoin_index']
            self.mode = 'LOCAL'
            self.replans += 1
            self.last_local_replan_time = self.time
            self.replan_timer = self.cfg.local_replan_cooldown
            names = ', '.join(t['obstacle'].name for t in info.get('threats', [])[:3]) or 'dynamic obstacle'
            self.event = f'LOCAL A* REPLAN → {reason} ({names})'
            return True

        # No safe local path yet: stop. The next frame retries immediately after the
        # moving obstacles have changed position instead of driving into the obstacle.
        self.robot.set_command(0.0, 0.0)
        self.mode = 'WAIT'
        self.event = f'NO SAFE LOCAL PATH → STOP / RETRY ({reason})'
        self.last_local_replan_time = self.time
        return False

    def _update_collision_contacts(self):
        """Record distinct robot/dynamic-obstacle collision encounters.

        This is called both before and after robot motion so a moving obstacle
        entering the robot footprint is counted even when the controller stops.
        """
        robot_pos = np.array([self.robot.x, self.robot.y], dtype=float)
        contacts = set()
        for obstacle in self.obstacles:
            d = float(np.linalg.norm(robot_pos - obstacle.position))
            if d <= self.cfg.robot_collision_radius + obstacle.radius:
                contacts.add(obstacle.name)

        new_contacts = contacts - self.active_contacts
        self.active_contacts = contacts

        if new_contacts:
            self.collisions += len(new_contacts)
            names = ", ".join(sorted(new_contacts))
            self.event = f'COLLISION #{self.collisions} ({names})'
        return new_contacts

    def step(self):
        if self.time >= self.cfg.max_time or self.mode == 'GOAL':
            return

        self.time += self.cfg.dt
        self.replan_timer = max(0.0, self.replan_timer - self.cfg.dt)
        for obstacle in self.obstacles:
            obstacle.update(self.cfg.dt, self.env)

        contacts_before_motion = self._update_collision_contacts()

        gi = self._global_index()
        active_for_detector = self.local_path if self.mode == 'LOCAL' and self.local_path is not None else self.global_path
        detected, info = self.detector.detect(
            self.robot, self.obstacles, self.global_path, gi, active_path=active_for_detector
        )
        self.last_detection = info

        min_dynamic_distance = self._min_dynamic_distance()
        contact_distance = self.cfg.robot_collision_radius + self.cfg.dynamic_obstacle_radius
        emergency_distance = contact_distance + 0.45
        emergency = min_dynamic_distance <= emergency_distance
        imminent = info['ttc'] <= self.cfg.collision_emergency_ttc

        # IMPORTANT: threat detection is processed BEFORE tracking. This prevents the
        # robot from taking one more forward step on a stale path when an obstacle
        # crosses its route.
        must_replan = detected and (
            self.mode in ('GLOBAL', 'LOCAL') or self.mode == 'WAIT'
        )
        if must_replan and (self.replan_timer <= 0.0 or imminent or emergency):
            self._do_local_replan(info, 'OBSTACLE THREAT')

        # If no plan exists, remain stopped until the obstacle moves enough for A* to
        # find a genuinely free detour.
        if self.mode == 'WAIT':
            self.robot.set_command(0.0, 0.0)
            self._record()
            # A moving obstacle can overlap the robot while it is waiting; the
            # collision was already recorded immediately after obstacle motion.
            return

        if self.mode == 'LOCAL':
            # While on the detour, continually re-plan if another dynamic obstacle
            # blocks the detour itself. This supports five moving obstacles rather
            # than assuming only one obstacle will ever matter.
            if detected and self.replan_timer <= 0.0 and (imminent or emergency or info['min_path_distance'] < 1.9):
                result = self.local.plan(self.robot, self.obstacles, self.global_path, gi, threat_info=info)
                if result:
                    self.local_path = result['path']
                    self.local_speed = self.smoother.speed_profile(self.local_path)
                    self.local_rejoin_index = result['rejoin_index']
                    self.replans += 1
                    self.last_local_replan_time = self.time
                    self.replan_timer = self.cfg.local_replan_cooldown
                    names = ', '.join(t['obstacle'].name for t in info['threats'][:3])
                    self.event = f'LOCAL A* REPLAN → NEW DETOUR ({names})'
                else:
                    # Keep following the existing local detour unless it is an
                    # immediate collision threat. A failed replacement plan should
                    # not throw away a route that is still safe and strand the robot.
                    self.replan_timer = 0.35
                    if imminent or emergency:
                        self.robot.set_command(0.0, 0.0)
                        self.mode = 'WAIT'
                        self.event = 'LOCAL DETOUR BLOCKED → SAFE STOP / RETRY'
                        self._record()
                        return

            v, w, _ = self.tracker.command(
                self.robot, self.local_path, self.local_speed,
                speed_scale=0.42 if emergency else (0.62 if imminent else 0.82)
            )

            # Rejoin only after the dynamic obstacles have cleared the original path,
            # and only when the robot is physically close to the planned rejoin point.
            rejoin_idx = min(self.local_rejoin_index, len(self.global_path) - 1)
            rejoin_point = np.asarray(self.global_path[rejoin_idx])
            dist_to_rejoin = float(np.linalg.norm(np.array([self.robot.x, self.robot.y]) - rejoin_point))
            global_clear = self._global_route_clear(max(0, rejoin_idx - 3))
            local_end_dist = float(np.linalg.norm(np.array([self.robot.x, self.robot.y]) - self.local_path[-1]))

            if global_clear and (dist_to_rejoin < self.cfg.rejoin_distance or local_end_dist < 0.75):
                self.global_index = max(self.global_index, rejoin_idx)
                self.local_path = None
                self.local_speed = None
                self.mode = 'GLOBAL'
                self.event = 'PATH CLEAR → SMOOTHLY REJOINING ORIGINAL GLOBAL A* PATH'
            active = self.local_path if self.mode == 'LOCAL' and self.local_path is not None else self.global_path

        elif self.mode == 'GLOBAL':
            # If an obstacle is close but A* cannot yet find a detour, stop rather than
            # continue toward the collision. If a detour exists, _do_local_replan above
            # will have switched to LOCAL.
            speed_scale = 0.35 if emergency else (0.55 if imminent else 1.0)
            v, w, gi2 = self.tracker.command(self.robot, self.global_path, self.global_speed, speed_scale=speed_scale)
            self.global_index = max(self.global_index, gi2)
            active = self.global_path
        else:
            active = self.global_path
            v, w = 0.0, 0.0

        # A final forward-collision guard. If the commanded motion would place the
        # robot inside a moving obstacle on this time step, stop and request a replan
        # before advancing the robot.
        if min_dynamic_distance <= emergency_distance:
            blocked_info = self.last_detection
            if self.mode != 'LOCAL' or (imminent and self.replan_timer <= 0.0):
                if self._do_local_replan(blocked_info, 'EMERGENCY SAFETY STOP'):
                    v, w, _ = self.tracker.command(
                        self.robot, self.local_path, self.local_speed,
                        speed_scale=0.45
                    )
                else:
                    v, w = 0.0, 0.0

        self.robot.set_command(v, w)
        self.robot.update(self.cfg.dt)

        new_contacts = self._update_collision_contacts()
        if new_contacts:
            contact_info = dict(self.last_detection)
            contact_info['threats'] = [
                {
                    'index': i,
                    'obstacle': o,
                    'ttc': 0.0,
                    'distance': float(np.linalg.norm(np.array([self.robot.x, self.robot.y]) - o.position)),
                    'min_path_distance': 0.0,
                }
                for i, o in enumerate(self.obstacles) if o.name in new_contacts
            ]
            # Collision is already recorded. Immediately attempt a recovery route.
            self._do_local_replan(contact_info, 'COLLISION RECOVERY')
            self.replan_timer = 0.0

        self._record()
        goal_dist = math.hypot(self.robot.x - self.goal[0], self.robot.y - self.goal[1])
        if goal_dist < 0.7:
            self.mode = 'GOAL'
            self.robot.set_command(0, 0)
            self.event = 'GOAL REACHED'

    def _record(self):
        self.history.append((self.robot.x, self.robot.y))
        for i, obstacle in enumerate(self.obstacles):
            self.obs_history[i].append(tuple(obstacle.position))

    @staticmethod
    def _robot_vertices(x, y, theta):
        # Nose-forward chassis polygon in robot local coordinates.
        local = np.array([
            [0.68, 0.00], [0.35, 0.43], [-0.58, 0.43],
            [-0.58, -0.43], [0.35, -0.43]
        ])
        c, s = math.cos(theta), math.sin(theta)
        R = np.array([[c, -s], [s, c]])
        return local @ R.T + np.array([x, y])

    def update_plot(self, frame):
        if not self.paused:
            self.step()

        gp = np.asarray(self.global_path)
        self.global_line.set_data(gp[:, 0], gp[:, 1])
        if self.mode == 'LOCAL' and self.local_path is not None:
            lp = np.asarray(self.local_path)
            self.active_line.set_data(lp[:, 0], lp[:, 1])
        else:
            self.active_line.set_data(gp[:, 0], gp[:, 1])

        self.robot_body.set_xy(self._robot_vertices(self.robot.x, self.robot.y, self.robot.theta))
        c, s = math.cos(self.robot.theta), math.sin(self.robot.theta)
        lateral = np.array([-s, c])
        forward = np.array([c, s])
        wheel_center = np.array([self.robot.x, self.robot.y]) - 0.23 * forward
        for line, sign in ((self.robot_wheel_left, 1), (self.robot_wheel_right, -1)):
            center = wheel_center + sign * 0.44 * lateral
            p1 = center - 0.20 * forward
            p2 = center + 0.20 * forward
            line.set_data([p1[0], p2[0]], [p1[1], p2[1]])
        eye = np.array([self.robot.x, self.robot.y]) + 0.34 * forward
        self.robot_center.set_data([self.robot.x], [self.robot.y])
        self.robot_eye.set_data([eye[0]], [eye[1]])

        for obstacle, patch, line in zip(self.obstacles, self.dynamic_patches, self.obstacle_heading):
            patch.center = tuple(obstacle.position)
            end = obstacle.position + 0.9 * obstacle.velocity
            line.set_data([obstacle.x, end[0]], [obstacle.y, end[1]])

        self.sensor_patch.center = (self.robot.x, self.robot.y)
        goal_dist = math.hypot(self.robot.x - self.goal[0], self.robot.y - self.goal[1])
        threat_text = ', '.join(t['obstacle'].name for t in self.last_detection['threats']) or 'none'
        bounce_text = ', '.join(f'{o.name}:{o.bounces}' for o in self.obstacles)
        self.status.set_text(
            f'Mode: {self.mode}\nEvent: {self.event}\nTime: {self.time:5.1f}s\n'
            f'Replans: {self.replans}   Collisions encountered: {self.collisions}\n'
            f'Rejoin index: {self.local_rejoin_index}   Local target: ' + ('active' if self.mode == 'LOCAL' else 'none') + '\n'
            f'Threats: {threat_text}\nNearest TTC: {self.last_detection["ttc"]:4.2f}s\n'
            f'Bounces: {bounce_text}\nGoal distance: {goal_dist:4.2f}m\n'
            f'Press E to edit static obstacles'
        )
        return (self.global_line, self.active_line, self.robot_body,
                self.robot_wheel_left, self.robot_wheel_right, self.robot_center,
                self.robot_eye, self.sensor_patch, self.status, *self.dynamic_patches,
                *self.obstacle_heading)

    def run(self):
        self.anim = FuncAnimation(
            self.fig, self.update_plot, interval=self.cfg.dt * 1000,
            blit=False, cache_frame_data=False
        )
        self.fig.canvas.mpl_connect('key_press_event', self._key)
        plt.show()

    def _key(self, event):
        if event.key == ' ':
            self.paused = not self.paused
        elif event.key in ('r', 'R'):
            self.reset()
        elif event.key in ('e', 'E'):
            self._set_edit_mode('drag')
        elif event.key == 'a' and self.editing:
            self._set_edit_mode('add')
        elif event.key in ('delete', 'backspace') and self.editing:
            self._set_edit_mode('remove')
        elif event.key == 'enter' and self.editing:
            self._apply_layout()
        elif event.key == 'escape':
            plt.close(self.fig)
