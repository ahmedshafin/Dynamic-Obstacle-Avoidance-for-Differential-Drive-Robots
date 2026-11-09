import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon
import numpy as np
from simulation import Simulation

sim = Simulation()
# Simulation() creates a figure for the live viewer; this script uses a separate figure.
plt.close(sim.fig)
for _ in range(2600):
    sim.step()
    if sim.mode == 'GOAL':
        break

fig, ax = plt.subplots(figsize=(10, 8))
ax.set_xlim(0, sim.cfg.width)
ax.set_ylim(0, sim.cfg.height)
ax.set_aspect('equal')
ax.set_title('Dynamic Obstacle Avoidance — 5 Dynamic Obstacles')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.grid(alpha=0.15)

for x, y, w, h in sim.env.static_obstacles:
    ax.add_patch(Rectangle((x, y), w, h, alpha=0.55))

gp = np.asarray(sim.global_path)
hist = np.asarray(sim.history)
ax.plot(gp[:, 0], gp[:, 1], '--', lw=1.5, label='Global A* + smooth')
if len(hist):
    ax.plot(hist[:, 0], hist[:, 1], lw=2.5, label='Robot trajectory')

for i, obstacle in enumerate(sim.obstacles):
    data = np.asarray(sim.obs_history[i])
    if len(data):
        ax.plot(data[:, 0], data[:, 1], ':', lw=1.2, alpha=0.8, label=f'{obstacle.name} path')
    ax.add_patch(Circle(tuple(obstacle.position), obstacle.radius, alpha=0.7))

ax.plot(sim.start[0], sim.start[1], 'o', markersize=10, label='Start A')
ax.plot(sim.goal[0], sim.goal[1], '*', markersize=13, label='Goal B')

# Final robot avatar.
verts = sim._robot_vertices(sim.robot.x, sim.robot.y, sim.robot.theta)
ax.add_patch(Polygon(verts, closed=True, fill=False, lw=2.0, label='Robot avatar'))

ax.text(
    0.02, 0.98,
    f'Result: {sim.mode}\nReplans: {sim.replans}\nRobot contacts: {sim.collisions}\n'
    f'Obstacle bounces: {sum(o.bounces for o in sim.obstacles)}\nTime: {sim.time:.1f}s',
    transform=ax.transAxes, va='top',
    bbox=dict(boxstyle='round', alpha=0.8)
)
ax.legend(loc='lower right', fontsize=7)
fig.tight_layout()
fig.savefig('demo_result.png', dpi=160)
print(f'Saved demo_result.png | mode={sim.mode} | replans={sim.replans} | '
      f'robot_contacts={sim.collisions} | bounces={sum(o.bounces for o in sim.obstacles)} | time={sim.time:.1f}s')
