"""
Real-time Matplotlib visualization for the drone delivery simulation.

Shows:
  - Grid with drones (coloured by state), servers, charging stations
  - Active delivery routes as annotated arrows
  - Live delivery statistics line chart
  - Per-drone battery level bar chart
  - Manual request injection panel (pause → enter coords → resume)
"""

import matplotlib
matplotlib.use("TkAgg")   # "Qt5Agg" or "Agg" if TkAgg unavailable

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.widgets as widgets
import numpy as np

from agents.drone_agent import DroneState, DroneAgent
from agents.server_agent import ServerAgent
from agents.charging_station import ChargingStation
from protocols.cnp_protocol import DeliveryRequest
from config.settings import SimConfig


STATE_COLORS = {
    DroneState.IDLE:                "#8800ff",
    DroneState.MANAGER:             "#ff00d0",
    DroneState.CONTRACTOR_WAITING:  "#0055ff",
    DroneState.DELIVERING:          "#00ff26",
    DroneState.CHARGING:            "#f50000",
}

# Dark-theme colours reused in the inject panel
_BG       = "#0f172a"
_PANEL_BG = "#1e293b"
_BORDER   = "#334155"
_TEXT     = "#e2e8f0"
_MUTED    = "#64748b"
_ACCENT   = "#38bdf8"
_SUCCESS  = "#4ade80"
_WARN     = "#fbbf24"
_ERR      = "#f87171"


class SimulationVisualizer:

    def __init__(self, model, figsize=(16, 10)):
        self.model   = model
        self.paused  = False          # simulation pause flag
        self._status_msg   = ""       # feedback shown below the inject button
        self._status_color = _TEXT

        self.fig = plt.figure(figsize=figsize, facecolor=_BG)
        self.fig.canvas.manager.set_window_title("Drone CNP Delivery Simulation")

        # ── Layout ──────────────────────────────────────────────────────
        # Three rows, three columns.
        # Row 0-1 col 0-1 : simulation grid  (large, spans both rows)
        # Row 0   col 2   : delivery stats chart
        # Row 1   col 2   : battery chart
        # Row 2   col 0-2 : inject / control panel (thin strip)
        gs = gridspec.GridSpec(
            3, 3, figure=self.fig,
            left=0.04, right=0.98, top=0.94, bottom=0.04,
            wspace=0.55, hspace=0.50,
            height_ratios=[0.45, 0.45, 0.10],
        )
        self.ax_grid  = self.fig.add_subplot(gs[:2, :2])
        self.ax_stats = self.fig.add_subplot(gs[0, 2])
        self.ax_bat   = self.fig.add_subplot(gs[1, 2])
        self.ax_panel = self.fig.add_subplot(gs[2, :])

        for ax in [self.ax_grid, self.ax_stats, self.ax_bat]:
            ax.set_facecolor(_PANEL_BG)
            for spine in ax.spines.values():
                spine.set_color(_BORDER)

        # Panel ax is purely decorative — widgets sit on top of it
        self.ax_panel.set_facecolor(_PANEL_BG)
        for spine in self.ax_panel.spines.values():
            spine.set_color(_BORDER)
        self.ax_panel.set_xticks([])
        self.ax_panel.set_yticks([])

        self._injected_request_ids: set = set()   # IDs of user-injected requests
        self._history = {"completed": [], "failed": [], "pending": [], "step": []}

        # ── Build control widgets ────────────────────────────────────────
        self._build_controls()

        plt.ion()
        plt.show()

    # ------------------------------------------------------------------ #
    #  Widget construction                                                 #
    # ------------------------------------------------------------------ #

    def _build_controls(self):
        """
        Lay out the inject panel inside the bottom strip (ax_panel bbox).
        Uses fig-fraction coordinates so it stays aligned regardless of
        the figure size chosen by the caller.
        """
        fig = self.fig

        # Helper: convert axes-relative [0,1] to figure-fraction
        def ax_to_fig(ax, rel_x, rel_y, rel_w, rel_h):
            bb = ax.get_position()
            return [
                bb.x0 + rel_x * bb.width,
                bb.y0 + rel_y * bb.height,
                rel_w * bb.width,
                rel_h * bb.height,
            ]

        panel_bb = self.ax_panel.get_position()
        px0  = panel_bb.x0
        py0  = panel_bb.y0
        pw   = panel_bb.width
        ph   = panel_bb.height

        btn_h  = ph * 0.55
        fld_h  = ph * 0.55   # match btn_h so all widgets are the same height
        gap    = pw * 0.008
        lbl_w  = pw * 0.045
        fld_w  = pw * 0.065
        btn_w  = pw * 0.075
        inj_w  = pw * 0.085

        # Single vertical centre for every widget in the strip
        vy_center = py0 + (ph - btn_h) * 0.5
        vy_btn = vy_center
        vy_fld = vy_center

        cur_x = px0 + gap

        # ── Pause / Resume button ────────────────────────────────────────
        ax_pause = fig.add_axes([cur_x, vy_btn, btn_w, btn_h])
        self._btn_pause = widgets.Button(
            ax_pause, "⏸  Pause",
            color=_PANEL_BG, hovercolor="#2d3f55",
        )
        self._btn_pause.label.set_color(_WARN)
        self._btn_pause.label.set_fontsize(8)
        self._btn_pause.on_clicked(self._on_pause_toggle)
        cur_x += btn_w + gap * 3

        # ── Separator label ──────────────────────────────────────────────
        ax_sep = fig.add_axes([cur_x, vy_btn, lbl_w * 1.8, btn_h])
        ax_sep.set_facecolor(_PANEL_BG)
        ax_sep.set_xticks([]); ax_sep.set_yticks([])
        for sp in ax_sep.spines.values():
            sp.set_visible(False)
        ax_sep.text(0.5, 0.5, "Inject Request →",
                    ha="center", va="center",
                    color=_MUTED, fontsize=7.5,
                    transform=ax_sep.transAxes)
        cur_x += lbl_w * 1.8 + gap * 2

        # ── Pickup X ────────────────────────────────────────────────────
        cur_x = self._add_labelled_field(
            fig, cur_x, vy_fld, vy_btn,
            lbl_w, fld_w, btn_h, fld_h, gap,
            label="PU X", attr="_tb_px",
            hint=f"0–{SimConfig.GRID_WIDTH - 1}",
        )

        # ── Pickup Y ────────────────────────────────────────────────────
        cur_x = self._add_labelled_field(
            fig, cur_x, vy_fld, vy_btn,
            lbl_w, fld_w, btn_h, fld_h, gap,
            label="PU Y", attr="_tb_py",
            hint=f"0–{SimConfig.GRID_HEIGHT - 1}",
        )

        cur_x += gap * 2   # visual gap between pickup / delivery

        # ── Delivery X ──────────────────────────────────────────────────
        cur_x = self._add_labelled_field(
            fig, cur_x, vy_fld, vy_btn,
            lbl_w, fld_w, btn_h, fld_h, gap,
            label="DL X", attr="_tb_dx",
            hint=f"0–{SimConfig.GRID_WIDTH - 1}",
        )

        # ── Delivery Y ──────────────────────────────────────────────────
        cur_x = self._add_labelled_field(
            fig, cur_x, vy_fld, vy_btn,
            lbl_w, fld_w, btn_h, fld_h, gap,
            label="DL Y", attr="_tb_dy",
            hint=f"0–{SimConfig.GRID_HEIGHT - 1}",
        )

        cur_x += gap * 3

        # ── Inject button ────────────────────────────────────────────────
        ax_inject = fig.add_axes([cur_x, vy_btn, inj_w, btn_h])
        self._btn_inject = widgets.Button(
            ax_inject, "➕  Inject",
            color=_PANEL_BG, hovercolor="#1a3a2a",
        )
        self._btn_inject.label.set_color(_SUCCESS)
        self._btn_inject.label.set_fontsize(8)
        self._btn_inject.on_clicked(self._on_inject)
        cur_x += inj_w + gap * 3

        # ── Status text area ─────────────────────────────────────────────
        ax_status = fig.add_axes([cur_x, vy_btn, pw * 0.25, btn_h])
        ax_status.set_facecolor(_PANEL_BG)
        ax_status.set_xticks([]); ax_status.set_yticks([])
        for sp in ax_status.spines.values():
            sp.set_visible(False)
        self._status_text = ax_status.text(
            0.01, 0.5, "",
            ha="left", va="center",
            color=_TEXT, fontsize=7.5,
            transform=ax_status.transAxes,
        )

    def _add_labelled_field(
        self, fig, cur_x, vy_fld, vy_btn,
        lbl_w, fld_w, btn_h, fld_h, gap,
        label, attr, hint,
    ):
        """Add a small label + TextBox pair; store the TextBox as self.<attr>."""
        # Label
        ax_lbl = fig.add_axes([cur_x, vy_btn, lbl_w, btn_h])
        ax_lbl.set_facecolor(_PANEL_BG)
        ax_lbl.set_xticks([]); ax_lbl.set_yticks([])
        for sp in ax_lbl.spines.values():
            sp.set_visible(False)
        ax_lbl.text(0.5, 0.5, label,
                    ha="center", va="center",
                    color=_MUTED, fontsize=7,
                    transform=ax_lbl.transAxes)
        cur_x += lbl_w

        # Text box
        ax_fld = fig.add_axes([cur_x, vy_fld, fld_w, fld_h])
        tb = widgets.TextBox(ax_fld, "", initial=hint,
                             color=_PANEL_BG, hovercolor="#253348")
        tb.text_disp.set_color(_TEXT)
        tb.text_disp.set_fontsize(7.5)
        # clear placeholder on first text change
        tb._placeholder = hint
        tb._cleared = False

        def _clear_placeholder(text, _tb=tb):
            if not _tb._cleared and text == hint:
                _tb.set_val("")
                _tb._cleared = True

        tb.on_text_change(_clear_placeholder)
        setattr(self, attr, tb)
        cur_x += fld_w + gap
        return cur_x

    # ------------------------------------------------------------------ #
    #  Widget callbacks                                                    #
    # ------------------------------------------------------------------ #

    def _on_pause_toggle(self, _event):
        self.paused = not self.paused
        if self.paused:
            self._btn_pause.label.set_text("▶  Resume")
            self._btn_pause.label.set_color(_SUCCESS)
            self._set_status("⏸ Paused — enter coordinates and click Inject.", _WARN)
        else:
            self._btn_pause.label.set_text("⏸  Pause")
            self._btn_pause.label.set_color(_WARN)
            self._set_status("▶ Resumed.", _SUCCESS)
        self.fig.canvas.draw_idle()

    def _on_inject(self, _event):
        """Parse fields, create a DeliveryRequest, push it to the model."""
        try:
            px = int(self._tb_px.text)
            py = int(self._tb_py.text)
            dx = int(self._tb_dx.text)
            dy = int(self._tb_dy.text)
        except ValueError:
            self._set_status("⚠ All coordinates must be integers.", _ERR)
            return

        W, H = SimConfig.GRID_WIDTH, SimConfig.GRID_HEIGHT

        if not (0 <= px < W and 0 <= py < H):
            self._set_status(f"⚠ Pickup ({px},{py}) out of grid bounds.", _ERR)
            return
        if not (0 <= dx < W and 0 <= dy < H):
            self._set_status(f"⚠ Delivery ({dx},{dy}) out of grid bounds.", _ERR)
            return
        if (px, py) == (dx, dy):
            self._set_status("⚠ Pickup and delivery must differ.", _ERR)
            return

        m   = self.model
        req = DeliveryRequest(
            request_id   = m.total_requests,
            pickup_pos   = (px, py),
            delivery_pos = (dx, dy),
            created_step = m.step_count,
            priority     = 1.0,           # manual requests get standard priority
        )
        m.pending_requests.append(req)
        m.total_requests += 1
        self._injected_request_ids.add(req.request_id)

        self._set_status(
            f"✓ Request #{req.request_id}  ({px},{py})→({dx},{dy})  queued.",
            _SUCCESS,
        )
        # Auto-resume after inject so the viewer can watch it unfold
        if self.paused:
            self.paused = False
            self._btn_pause.label.set_text("⏸  Pause")
            self._btn_pause.label.set_color(_WARN)

        self.fig.canvas.draw_idle()

    def _set_status(self, msg, color=_TEXT):
        self._status_text.set_text(msg)
        self._status_text.set_color(color)

    # ------------------------------------------------------------------ #
    #  Main update — called every simulation step                          #
    # ------------------------------------------------------------------ #

    def update(self):
        m = self.model
        self.ax_grid.cla()
        self.ax_stats.cla()
        self.ax_bat.cla()

        self._draw_grid(m)
        self._draw_stats(m)
        self._draw_batteries(m)

        pause_label = "  [PAUSED]" if self.paused else ""
        self.fig.suptitle(
            f"Drone CNP Delivery — Step {m.step_count}  |  "
            f"Completed: {m.completed_deliveries}  "
            f"Failed: {m.failed_deliveries}  "
            f"Total: {m.total_requests}"
            f"{pause_label}",
            color="#e2e8f0", fontsize=13, fontweight="bold",
        )
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    # ------------------------------------------------------------------ #
    #  Grid drawing                                                        #
    # ------------------------------------------------------------------ #

    def _draw_grid(self, m):
        ax = self.ax_grid
        ax.set_facecolor(_PANEL_BG)
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
            if req.request_id in self._injected_request_ids:
                color = "#ff0000"   # red for user-injected (assigned or not)
            else:
                color = _SUCCESS if req.assigned_drone_id else "#a78bfa"
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
            arrow_color = "#fb923c" if req.request_id in self._injected_request_ids else "#fbbf24"
            ax.annotate(
                "", xy=(dx, dy), xytext=(px, py),
                arrowprops=dict(
                    arrowstyle="->", color=arrow_color, lw=0.8, linestyle="--",
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
                str(drone.unique_id - 8), xy=drone.pos,
                xytext=(drone.pos[0] + 0.2, drone.pos[1] + 0.25),
                color="#e2e8f0", fontsize=6, zorder=7,
            )

        legend_elements = (
            [mpatches.Patch(color=c, label=s.name) for s, c in STATE_COLORS.items()]
            + [
                mpatches.Patch(color="#fbbf24", label="Charging Station"),
                mpatches.Patch(color="#38bdf8", label="Server"),
                mpatches.Patch(color="#ff0000", label="Injected Request"),
            ]
        )
        ax.legend(
            handles=legend_elements,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            borderaxespad=0,
            fontsize=7, ncol=1,
            facecolor="#0f172a", edgecolor="#334155", labelcolor="#cbd5e1",
        )

    # ------------------------------------------------------------------ #
    #  Stats / battery charts (unchanged)                                  #
    # ------------------------------------------------------------------ #

    def _draw_stats(self, m):
        ax = self.ax_stats
        ax.set_facecolor(_PANEL_BG)
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
        ax.set_facecolor(_PANEL_BG)
        ax.set_title("Drone Battery Levels", color="#94a3b8", fontsize=10)
        ax.tick_params(colors="#475569", labelsize=7)

        drones    = list(m.agents_by_type[DroneAgent])
        ids       = [str(d.unique_id - 8) for d in drones]
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