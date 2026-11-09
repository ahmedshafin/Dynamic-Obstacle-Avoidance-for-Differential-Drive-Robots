import math
import numpy as np

class PurePursuitTracker:
    def __init__(self,cfg): self.cfg=cfg

    @staticmethod
    def _angle(a): return (a+math.pi)%(2*math.pi)-math.pi

    def find_target(self, robot, path, speed_scale=1.0):
        if len(path)==0: return None,0
        pos=np.array([robot.x,robot.y])
        d=np.linalg.norm(path-pos,axis=1)
        nearest=int(np.argmin(d))
        look=self.cfg.lookahead_min+(self.cfg.lookahead_max-self.cfg.lookahead_min)*min(1.0,abs(robot.v)/self.cfg.max_linear_speed)
        look*=speed_scale
        target=nearest
        for i in range(nearest,len(path)):
            if np.linalg.norm(path[i]-pos)>=look:
                target=i;break
        return path[target],target

    def command(self,robot,path,speed_profile=None,speed_scale=1.0):
        target,idx=self.find_target(robot,path,speed_scale)
        if target is None: return 0.0,0.0,0
        dx,dy=target[0]-robot.x,target[1]-robot.y
        distance=math.hypot(dx,dy)
        desired=math.atan2(dy,dx)
        err=self._angle(desired-robot.theta)
        omega=self.cfg.heading_gain*err
        # Position-based steering helps when local path starts close to robot.
        omega += self.cfg.position_gain*math.sin(err)*min(distance,2.0)
        omega=float(np.clip(omega,-self.cfg.max_angular_speed,self.cfg.max_angular_speed))
        v=self.cfg.max_linear_speed*speed_scale
        if speed_profile is not None and len(speed_profile):
            v=min(v,float(speed_profile[min(idx,len(speed_profile)-1)]))
        v*=max(0.12,1.0-0.62*min(abs(err)/math.pi,1.0))
        if distance<self.cfg.waypoint_tolerance:
            v*=0.6
        return v,omega,idx
