# Dynamic Obstacle Avoidance for Differential-Drive Robots

Python/Matplotlib simulation implementing:

1. Environment & grid mapping with editable static rectangular obstacles
2. A* global path planning
3. Kinodynamic smoothing with curvature-aware speed profile
4. Differential-drive robot kinematics
5. Pure-Pursuit trajectory tracking
6. Dynamic obstacle detection and predicted collision checking
7. Adaptive bounded local A* re-planning and safe rejoining of the original path

## Current version

This version uses **5 dynamic obstacles**. Dynamic obstacles bounce from both static rectangular obstacles and the world boundaries.

## Interactive static-obstacle editor

The simulation window now lets you experiment with different static maps without editing the source code.

### Buttons

- **Drag** — click and drag any static rectangular obstacle to a new position.
- **Add** — click an empty area to add a new 5 m × 9 m static obstacle.
- **Remove** — click a static obstacle to remove it.
- **Apply Layout** — rebuilds the occupancy grid, recalculates the A* global path and restarts the complete simulation from A.
- **Default Layout** — restores the original static-obstacle arrangement.

Editing automatically pauses the simulation. The layout editor prevents obstacles from being moved outside the map, overlapping each other, or covering the start/goal area.

### Keyboard shortcuts

- **Space** — pause/resume simulation
- **E** — enter drag/edit mode
- **A** — enter add mode while editing
- **Delete / Backspace** — enter remove mode while editing
- **Enter** — apply the current edited layout
- **R** — restart the simulation with the current static layout
- **Esc** — close

## Re-planning behavior

When a dynamic obstacle is predicted to intersect the robot's active route:

- The robot checks the threat **before** executing the next tracking command.
- A bounded local A* search creates a **different detour** around the moving obstacles.
- The local planner predicts the moving obstacles forward in time and inflates their swept region.
- If a safe local route cannot be found, the robot **stops instead of driving into the obstacle** and retries as the obstacle moves.
- While on the local route, another obstacle can trigger another local re-plan.
- The robot does **not** return to the global route until the original route is predicted to be collision-free and the robot reaches the selected rejoin point.
- After clearance, it switches back to the original smoothed A* path and continues to the goal.

## Run in VS Code

```bash
pip install -r requirements.txt
python main.py
```

## Headless demo image

```bash
python save_demo.py
```
