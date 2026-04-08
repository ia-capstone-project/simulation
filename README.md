# Drone CNP Delivery Simulation

A Mesa-based multi-agent simulation of drone package delivery using the **Contract Net Protocol (CNP)**. Drones dynamically negotiate task ownership through decentralised bidding, with adaptive utility weight learning and battery-aware safety constraints.

---

## Directory Structure

```
drone_cnp/
├── run_simulation.py          # Entry point (headless or visualized)
├── model.py                   # DroneDeliveryModel — CNP manager assignment
├── mesa_lite.py               # Drop-in Mesa shim (used if mesa not installed)
├── requirements.txt
│
├── agents/
│   ├── drone_agent.py         # DroneAgent — full CNP + adaptive learning
│   ├── server_agent.py        # Fixed server nodes (request origin)
│   └── charging_station.py   # Passive charging infrastructure
│
├── protocols/
│   └── cnp_protocol.py        # CNP message types, DeliveryRequest, CNPRound
│
├── config/
│   └── settings.py            # All tunable parameters in one place
│
├── utils/
│   └── helpers.py             # Manhattan distance, summary stats
│
└── visualization/
    └── visualizer.py          # Live Matplotlib grid + stats dashboard
```

---

## Quick Start

```bash
# Install dependencies (mesa optional — mesa_lite is the built-in fallback)
pip install mesa numpy matplotlib

# Headless run (200 steps, prints summary)
python run_simulation.py

# With live Matplotlib visualization
python run_simulation.py --vis

# Custom parameters
python run_simulation.py --vis --steps 500 --drones 15 --seed 42

# All options
python run_simulation.py --help
```

---

## Architecture

### Agent Types

| Agent | Role |
|-------|------|
| `DroneAgent` | Mobile delivery agent. Assumes IDLE / MANAGER / CONTRACTOR / DELIVERING / CHARGING roles dynamically. |
| `ServerAgent` | Fixed grid node. Generates delivery requests stochastically each step. |
| `ChargingStation` | Passive infrastructure. Drones navigate here to recharge. |

### CNP Flow Per Request

```
Server generates request
       │
       ▼
model.assign_managers()          ← closest idle drone in comm_range
       │
       ▼
Manager.become_manager(req)      ← sets state = MANAGER
Manager.issue_cfp(req)           ← broadcasts CFP to all neighbours
       │
       ▼  (within CFP_DEADLINE_STEPS)
Contractors._handle_cfp()        ← each computes utility + battery check
Contractors → ProposalMessage    ← sent directly to manager's inbox
       │
       ▼
Manager._evaluate_and_award()    ← picks highest utility_score
Manager → AwardMessage (winner)
Manager → RejectMessage (losers)
       │
       ▼
Winner._handle_award()           ← state = DELIVERING
       → flies to pickup → flies to delivery → mark_completed()
```

### Why the Model Assigns the Manager

The model (not the drones) selects the manager to avoid race conditions.
If two drones self-nominated simultaneously, both could become managers for the same request.
The model applies a deterministic rule — closest idle drone, tie-break by `unique_id` — then calls `drone.become_manager(req)` exactly once. All subsequent negotiation (CFP, bidding, evaluation, award) is fully decentralised.

---

## Utility Function

Each contractor computes:

```
U = α × priority  −  β × d_norm  −  γ × (1 − battery_norm)
```

- `d_norm = (dist_to_pickup + dist_to_delivery) / grid_diagonal`
- `battery_norm = battery / battery_max`
- `α, β, γ` are per-drone adaptive weights (initialised from `SimConfig`)

### Adaptive Weight Learning (Parameter Adaptation)

This is **not reinforcement learning**. Weights shift based on rolling success rate over a sliding window (`ADAPTATION_WINDOW = 20` steps):

| Condition | Adjustment |
|-----------|-----------|
| success_rate < 0.5 | Increase β, γ (be more conservative) · Decrease α |
| success_rate > 0.8 | Decrease β, γ (slightly more aggressive) · Increase α |

---

## Battery Management

Before bidding, each drone runs a feasibility check:

```python
required = (task_distance + dist_to_nearest_charger) * drain_per_step
         + safety_reserve * battery_max
feasible = battery >= required
```

The `safety_reserve` threshold adapts:
- **Increases** by `SAFETY_RESERVE_INCREMENT` after any battery depletion event
- **Decreases** by `SAFETY_RESERVE_DECREMENT` after successful runs (via weight adaptation)
- Clamped to `[SAFETY_RESERVE_MIN, SAFETY_RESERVE_MAX]`

> Incase of an agent losing bids over long time due to unable to handle the battery requirements, it triggers a **preemptive** charging.


---

## Configuration (`config/settings.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `GRID_WIDTH / HEIGHT` | 30 | Grid dimensions |
| `NUM_DRONES` | 12 | Number of drone agents |
| `NUM_SERVERS` | 4 | Fixed server stations |
| `NUM_CHARGING_STATIONS` | 6 | Charging infrastructure |
| `REQUEST_RATE` | 0.08 | Probability of request per server per step |
| `COMM_RANGE` | 10 | Manhattan radius for CFP broadcast |
| `BATTERY_MAX` | 100 | Max battery level |
| `BATTERY_DRAIN_MOVE` | 0.4 | Battery drained per step |
| `CHARGE_RATE` | 5.0 | Battery gained per step at station |
| `CFP_DEADLINE_STEPS` | 3 | Steps manager waits for proposals |
| `ALPHA_INIT` | 1.0 | Initial reward weight |
| `BETA_INIT` | 0.6 | Initial distance cost weight |
| `GAMMA_INIT` | 0.4 | Initial battery cost weight |
| `ADAPTATION_WINDOW` | 20 | Rolling window for success rate |
| `REQUEST_TIMEOUT` | 80 | Steps before unserved request expires |

---

## Using Real Mesa

The simulation uses `mesa_lite.py` (built-in) if Mesa is not installed.
To use the real Mesa library:

```bash
pip install mesa
```

All imports automatically prefer real Mesa over the shim — no code changes needed.

---

## Extending the Simulation

- **New utility factors**: Edit `DroneAgent._compute_utility()` — add weather, load, urgency.
- **Different manager selection**: Edit `DroneDeliveryModel._select_manager()` — e.g. highest battery instead of closest.
- **Multi-hop delivery**: Add relay waypoints to `DeliveryRequest`.
- **Priority queues**: Replace the flat `pending_requests` list with a heap keyed on `req.priority`.
- **Mesa Solara UI**: The `datacollector` is already set up — wire it to `mesa.visualization.solara_viz` for a web dashboard.