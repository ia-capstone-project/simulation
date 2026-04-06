"""
Simulation configuration constants.
Centralised here so tuning is easy.
"""


class SimConfig:
    # ---- Grid ----
    GRID_WIDTH: int = 30
    GRID_HEIGHT: int = 30

    # ---- Agents ----
    NUM_DRONES: int = 12
    NUM_SERVERS: int = 4
    NUM_CHARGING_STATIONS: int = 6

    # ---- Requests ----
    REQUEST_RATE: float = 0.08      # probability per server per step
    REQUEST_TIMEOUT: int = 80       # steps before a request expires

    # ---- Communication ----
    COMM_RANGE: int = 10            # Manhattan distance for CFP broadcast

    # ---- Battery ----
    BATTERY_MAX: float = 100.0
    BATTERY_DRAIN_MOVE: float = 0.4    # drained per step while active
    CHARGE_RATE: float = 5.0           # gained per step at station
    CHARGE_FULL_THRESHOLD: float = 0.95

    # Safety reserve (adaptive)
    INITIAL_SAFETY_RESERVE: float = 0.15   # fraction of battery_max
    SAFETY_RESERVE_MIN: float = 0.10
    SAFETY_RESERVE_MAX: float = 0.40
    SAFETY_RESERVE_INCREMENT: float = 0.02
    SAFETY_RESERVE_DECREMENT: float = 0.005

    # ---- CNP ----
    CFP_DEADLINE_STEPS: int = 3     # Manager waits this many steps for bids

    # ---- Utility Weights (adaptive) ----
    ALPHA_INIT: float = 1.0     # reward / priority weight
    BETA_INIT: float = 0.6      # distance cost weight
    GAMMA_INIT: float = 0.4     # battery cost weight

    ALPHA_MIN: float = 0.5
    ALPHA_MAX: float = 2.0
    BETA_MIN: float = 0.2
    BETA_MAX: float = 1.5
    GAMMA_MIN: float = 0.1
    GAMMA_MAX: float = 1.0

    WEIGHT_LEARNING_RATE: float = 0.05
    ADAPTATION_WINDOW: int = 20   # rolling window for success rate

    # ---- Visualization ----
    CELL_SIZE: int = 18
    FPS: int = 10