from dataclasses import dataclass

@dataclass
class Config:
    # World/grid
    width: int = 50
    height: int = 50
    grid_resolution: float = 1.0
    robot_radius: float = 0.45
    # Approximate robot chassis footprint used for dynamic collision checks.
    robot_collision_radius: float = 0.72
    safety_margin: float = 0.35

    # Differential drive
    wheel_radius: float = 0.16
    wheel_base: float = 0.62
    max_wheel_speed: float = 1.35
    max_linear_speed: float = 1.0
    max_angular_speed: float = 2.2
    max_accel: float = 0.85
    max_decel: float = 1.25

    # Tracking
    lookahead_min: float = 0.9
    lookahead_max: float = 1.8
    heading_gain: float = 2.0
    position_gain: float = 1.1
    waypoint_tolerance: float = 0.55

    # Dynamic obstacle
    sensor_radius: float = 8.0
    dynamic_obstacle_radius: float = 0.65
    collision_distance: float = 1.25
    prediction_horizon: float = 5.0
    dynamic_obstacle_count: int = 5

    # Local replanning
    local_window: int = 41
    local_resolution: float = 0.5
    local_replan_cooldown: float = 1.0
    rejoin_distance: float = 2.0
    local_prediction_horizon: float = 5.5
    local_prediction_samples: int = 16
    local_dynamic_inflation: float = 0.20

    # Local replanning / rejoin
    local_target_min_ahead: int = 10
    local_target_max_ahead: int = 42
    local_target_step: int = 2
    local_start_clearance: float = 0.35
    minimum_detour_offset: float = 0.75
    rejoin_clearance: float = 1.70
    collision_emergency_ttc: float = 1.80

    # Simulation
    dt: float = 0.06
    max_time: float = 180.0
