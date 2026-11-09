import math
import numpy as np

class DifferentialDriveRobot:
    def __init__(self, cfg, start):
        self.cfg = cfg
        self.x, self.y, self.theta = float(start[0]), float(start[1]), float(start[2])
        self.v = 0.0
        self.omega = 0.0
        self.wl = self.wr = 0.0

    @property
    def pose(self):
        return np.array([self.x,self.y,self.theta])

    def set_command(self, v_cmd, omega_cmd):
        v_cmd = float(np.clip(v_cmd, -self.cfg.max_linear_speed, self.cfg.max_linear_speed))
        omega_cmd = float(np.clip(omega_cmd, -self.cfg.max_angular_speed, self.cfg.max_angular_speed))
        self.v = v_cmd
        self.omega = omega_cmd
        self.wr = (2*v_cmd + omega_cmd*self.cfg.wheel_base)/(2*self.cfg.wheel_radius)
        self.wl = (2*v_cmd - omega_cmd*self.cfg.wheel_base)/(2*self.cfg.wheel_radius)
        self.wr = float(np.clip(self.wr, -self.cfg.max_wheel_speed/self.cfg.wheel_radius, self.cfg.max_wheel_speed/self.cfg.wheel_radius))
        self.wl = float(np.clip(self.wl, -self.cfg.max_wheel_speed/self.cfg.wheel_radius, self.cfg.max_wheel_speed/self.cfg.wheel_radius))

    def update(self, dt):
        self.x += self.v*math.cos(self.theta)*dt
        self.y += self.v*math.sin(self.theta)*dt
        self.theta = (self.theta + self.omega*dt + math.pi)%(2*math.pi)-math.pi
