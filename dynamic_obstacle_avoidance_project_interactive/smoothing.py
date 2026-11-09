import math
import numpy as np

class KinodynamicSmoother:
    """Creates a smooth path and a curvature/acceleration-limited speed profile."""
    def __init__(self, env, cfg):
        self.env, self.cfg = env, cfg

    def _catmull_rom(self, p0,p1,p2,p3,t):
        t2, t3 = t*t, t*t*t
        return 0.5*((2*p1) + (-p0+p2)*t + (2*p0-5*p1+4*p2-p3)*t2 + (-p0+3*p1-3*p2+p3)*t3)

    def smooth(self, path, samples_per_segment=8):
        if len(path) < 3:
            return np.asarray(path, dtype=float)
        pts = np.asarray(path, dtype=float)
        dense = []
        for i in range(len(pts)-1):
            p0 = pts[max(0,i-1)]
            p1 = pts[i]
            p2 = pts[i+1]
            p3 = pts[min(len(pts)-1,i+2)]
            for k in range(samples_per_segment):
                dense.append(self._catmull_rom(p0,p1,p2,p3,k/samples_per_segment))
        dense.append(pts[-1])
        dense = np.asarray(dense)
        # Reject excursions into obstacles; this keeps smoothing physically useful.
        valid = []
        for p in dense:
            if self.env.point_free(float(p[0]), float(p[1]), self.cfg.robot_radius + 0.12):
                valid.append(p)
            elif valid:
                valid.append(valid[-1].copy())
        if len(valid) < 2:
            return pts
        return np.asarray(valid)

    def speed_profile(self, path):
        p = np.asarray(path, dtype=float)
        if len(p) < 2:
            return np.zeros(len(p))
        d = np.linalg.norm(np.diff(p,axis=0),axis=1)
        heading = np.unwrap(np.arctan2(np.diff(p[:,1]), np.diff(p[:,0])))
        curvature = np.zeros(len(p))
        if len(heading) > 1:
            ds = np.maximum((d[:-1]+d[1:])*0.5, 1e-3)
            curvature[1:-1] = np.abs(np.diff(heading))/ds
        vmax = self.cfg.max_linear_speed / (1.0 + 1.8*curvature)
        vmax = np.clip(vmax, 0.28, self.cfg.max_linear_speed)
        # Forward/backward acceleration limits.
        for i in range(1,len(vmax)):
            vmax[i] = min(vmax[i], math.sqrt(max(0, vmax[i-1]**2 + 2*self.cfg.max_accel*max(d[i-1],1e-3))))
        for i in range(len(vmax)-2,-1,-1):
            vmax[i] = min(vmax[i], math.sqrt(max(0, vmax[i+1]**2 + 2*self.cfg.max_decel*max(d[i],1e-3))))
        return vmax
