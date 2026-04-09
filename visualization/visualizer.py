"""
Real-time Matplotlib visualization for the drone delivery simulation.
Mesa 3.0 compatible — uses model.agents_by_type[AgentClass] (AgentSet).

Shows:
  - Grid with drones (coloured by state), servers, charging stations
  - Active delivery routes as annotated arrows
  - Live delivery statistics line chart
  - Per-drone battery level bar chart
"""

import matplotlib
matplotlib.use("TkAgg")   # change to "Qt5Agg" or "Agg" if TkAgg unavailable

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np

from agents.drone_agent import DroneState, DroneAgent
from agents.server_agent import ServerAgent
from agents.charging_station import ChargingStation


STATE_COLORS = {
    DroneState.IDLE:                "#8800ff",
    DroneState.MANAGER:             "#ff00d0",
    DroneState.CONTRACTOR_WAITING:  "#0055ff",
    DroneState.DELIVERING:          "#00ff26",
    DroneState.CHARGING:            "#f50000",
}


class SimulationVisualizer:

    def __init__(self, model, figsize=(16, 9)):
        self.model   = model
        self.fig     = plt.figure(figsize=figsize, facecolor="#0f172a")
        self.fig.canvas.manager.set_window_title("Drone CNP Delivery Simulation")

        gs = gridspec.GridSpec(
            2, 3, figure=self.fig,
            left=0.04, right=0.98, top=0.94, bottom=0.06,
            wspace=0.35, hspace=0.45,
        )
        self.ax_grid  = self.fig.add_subplot(gs[:, :2])
        self.ax_stats = self.fig.add_subplot(gs[0, 2])
        self.ax_bat   = self.fig.add_subplot(gs[1, 2])

        for ax in [self.ax_grid, self.ax_stats, self.ax_bat]:
            ax.set_facecolor("#1e293b")
            for spine in ax.spines.values():
                spine.set_color("#334155")

        self._history = {"completed": [], "failed": [], "pending": [], "step": []}

        plt.ion()
        plt.show()

    def update(self):
        m = self.model
        self.ax_grid.cla()
        self.ax_stats.cla()
        self.ax_bat.cla()

        self._draw_grid(m)
        self._draw_stats(m)
        self._draw_batteries(m)

        self.fig.suptitle(
            f"Drone CNP Delivery — Step {m.step_count}  |  "
            f"Completed: {m.completed_deliveries}  Failed: {m.failed_deliveries} Total: {m.total_requests}",
            color="#e2e8f0", fontsize=13, fontweight="bold",
        )
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _draw_grid(self, m):
        ax = self.ax_grid
        ax.set_facecolor("#1e293b")
        ax.set_xlim(-0.5, m.width  - 0.5)
        ax.set_ylim(-0.5, m.height - 0.5)
        ax.set_aspect("equal")
        ax.set_title("Simulation Grid", color="#94a3b8", fontsize=10)
        ax.tick_params(colors="#475569", labelsize=7)

        for x in range(m.width):
            ax.axvline(x - 0.5, color="#1e3a5f", linewidth=0.3, zorder=0)
        for y in range(m.height):
            ax.axhline(y - 0.5, color="#1e3a5f", linewidth=0.3, zorder=0)

        # Mesa 3.0: agents_by_type returns an AgentSet — iterate directly
        for cs in m.agents_by_type[ChargingStation]:
            ax.plot(*cs.pos, marker="s", markersize=11, color="#fbbf24",
                    zorder=3, alpha=0.85)

        for sv in m.agents_by_type[ServerAgent]:
            ax.plot(*sv.pos, marker="^", markersize=13, color="#38bdf8",
                    zorder=3, alpha=0.9)
            ax.annotate(
                str(sv.queue_depth), xy=sv.pos,
                xytext=(sv.pos[0] + 0.3, sv.pos[1] + 0.3),
                color="#e2e8f0", fontsize=7, zorder=5,
            )

        for req in m.active_requests.values():
            px, py = req.pickup_pos
            dx, dy = req.delivery_pos
            color = "#00ff26"  if req.assigned_drone_id else "#a78bfa"
            ax.annotate(
                "", xy=(dx, dy), xytext=(px, py),
                arrowprops=dict(
                    arrowstyle="->", color=color, lw=1.2,
                    connectionstyle="arc3,rad=0.2",
                ),
                zorder=2,
            )
        
        for req in m.pending_requests:
            px, py = req.pickup_pos
            dx, dy = req.delivery_pos
            ax.annotate(
                "", xy=(dx, dy), xytext=(px, py),
                arrowprops=dict(
                    arrowstyle="->", color="#fbbf24", lw=0.8, linestyle="--",
                    connectionstyle="arc3,rad=0.2",
                ),
                zorder=1,
            )

        for drone in m.agents_by_type[DroneAgent]:
            color = STATE_COLORS.get(drone.state, "#ffffff")
            ax.plot(*drone.pos, marker="D", markersize=9, color=color,
                    zorder=6, alpha=0.95,
                    markeredgecolor="#0f172a", markeredgewidth=0.8)
            ax.annotate(
                str(drone.unique_id), xy=drone.pos,
                xytext=(drone.pos[0] + 0.2, drone.pos[1] + 0.25),
                color="#e2e8f0", fontsize=6, zorder=7,
            )

        legend_elements = (
            [mpatches.Patch(color=c, label=s.name) for s, c in STATE_COLORS.items()]
            + [
                mpatches.Patch(color="#fbbf24", label="Charging Station"),
                mpatches.Patch(color="#38bdf8", label="Server"),
            ]
        )
        ax.legend(
            handles=legend_elements, loc="upper right", fontsize=7, ncol=2,
            facecolor="#0f172a", edgecolor="#334155", labelcolor="#cbd5e1",
        )

    def _draw_stats(self, m):
        ax = self.ax_stats
        ax.set_facecolor("#1e293b")
        ax.set_title("Delivery Statistics", color="#94a3b8", fontsize=10)
        ax.tick_params(colors="#475569", labelsize=8)

        self._history["step"].append(m.step_count)
        self._history["completed"].append(m.completed_deliveries)
        self._history["failed"].append(m.failed_deliveries)
        self._history["pending"].append(len(m.pending_requests))

        steps = self._history["step"]
        ax.plot(steps, self._history["completed"],
                color="#4ade80", label="Completed", linewidth=1.5)
        ax.plot(steps, self._history["failed"],
                color="#f87171", label="Failed",    linewidth=1.5)
        ax.plot(steps, self._history["pending"],
                color="#fbbf24", label="Pending",   linewidth=1.2, linestyle="--")

        ax.set_xlabel("Step", color="#64748b", fontsize=8)
        ax.legend(fontsize=7, facecolor="#0f172a",
                  edgecolor="#334155", labelcolor="#cbd5e1")

    def _draw_batteries(self, m):
        ax = self.ax_bat
        ax.set_facecolor("#1e293b")
        ax.set_title("Drone Battery Levels", color="#94a3b8", fontsize=10)
        ax.tick_params(colors="#475569", labelsize=7)

        # Mesa 3.0: iterate AgentSet
        drones    = list(m.agents_by_type[DroneAgent])
        ids       = [str(d.unique_id) for d in drones]
        batteries = [d.battery for d in drones]
        colors    = [STATE_COLORS.get(d.state, "#94a3b8") for d in drones]

        ax.barh(ids, batteries, color=colors, alpha=0.85, height=0.6)
        ax.set_xlim(0, drones[0].battery_max if drones else 100)

        avg_reserve = (
            np.mean([d.safety_reserve for d in drones]) * drones[0].battery_max
            if drones else 15
        )
        ax.axvline(x=avg_reserve, color="#ef4444", linewidth=1,
                   linestyle=":", label="Avg safety reserve")
        ax.set_xlabel("Battery", color="#64748b", fontsize=8)
        ax.legend(fontsize=7, facecolor="#0f172a",
                  edgecolor="#334155", labelcolor="#cbd5e1")