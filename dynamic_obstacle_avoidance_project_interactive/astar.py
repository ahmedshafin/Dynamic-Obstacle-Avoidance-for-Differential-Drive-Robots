import heapq
import math

class AStarPlanner:
    """8-connected A* planner over a boolean occupancy grid."""
    def __init__(self, env):
        self.env = env

    def _neighbors(self, node):
        x, y = node
        for dx, dy, cost in [
            (-1,0,1), (1,0,1), (0,-1,1), (0,1,1),
            (-1,-1,math.sqrt(2)), (-1,1,math.sqrt(2)),
            (1,-1,math.sqrt(2)), (1,1,math.sqrt(2))]:
            yield (x+dx, y+dy, cost)

    @staticmethod
    def _h(a,b):
        return math.hypot(a[0]-b[0], a[1]-b[1])

    def plan(self, start, goal, extra_blocked=None, bounds=None, robot_radius=0.45):
        sx, sy = int(round(start[0])), int(round(start[1]))
        gx, gy = int(round(goal[0])), int(round(goal[1]))
        if bounds is None:
            xmin, xmax, ymin, ymax = 0, self.env.cfg.width-1, 0, self.env.cfg.height-1
        else:
            xmin, xmax, ymin, ymax = bounds

        def blocked(x,y):
            if x < xmin or x > xmax or y < ymin or y > ymax:
                return True
            if self.env.is_static_collision(x+0.5, y+0.5, robot_radius):
                return True
            if extra_blocked:
                for bx, by, br in extra_blocked:
                    if (x+0.5-bx)**2 + (y+0.5-by)**2 <= (br+robot_radius)**2:
                        return True
            return False

        start_n, goal_n = (sx,sy), (gx,gy)
        if blocked(*start_n) or blocked(*goal_n):
            return []
        open_heap = [(self._h(start_n, goal_n), 0.0, start_n)]
        came = {}
        g = {start_n: 0.0}
        closed = set()
        while open_heap:
            _, cost, cur = heapq.heappop(open_heap)
            if cur in closed:
                continue
            closed.add(cur)
            if cur == goal_n:
                path = [cur]
                while cur in came:
                    cur = came[cur]
                    path.append(cur)
                path.reverse()
                return [(x+0.5, y+0.5) for x,y in path]
            for nx, ny, step in self._neighbors(cur):
                if blocked(nx,ny) or (nx,ny) in closed:
                    continue
                ng = cost + step
                if ng < g.get((nx,ny), float('inf')):
                    g[(nx,ny)] = ng
                    came[(nx,ny)] = cur
                    heapq.heappush(open_heap, (ng+self._h((nx,ny),goal_n), ng, (nx,ny)))
        return []
