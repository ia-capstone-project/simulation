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
    NUM_CHARGING_STATIONS: int = 4

    # ---- Requests ----
    REQUEST_RATE: float = 0.2     # probability per server per step
    REQUEST_TIMEOUT: int = 1000       # TODO: steps before a request expires - now set to huge value

    # ---- Communication ----
    COMM_RANGE: int = 10            # Manhattan distance for CFP broadcast

    # ---- Battery ----
    BATTERY_MAX: float = 100.0
    BATTERY_DRAIN_MOVE: float = 0.4    # drained per step while active
    CHARGE_RATE: float = 5.0           # gained per step at station
    CHARGE_FULL_THRESHOLD: float = 0.95

    # ---- Preemptive Charging ----
    STARVATION_IDLE_THRESHOLD = 70
    STARVATION_DELIVERY_THRESHOLD = 40
    BID_FAILURE_THRESHOLD = 3
    REJECTION_THRESHOLD = 4
    PREEMPTIVE_CHARGE_BATTERY_THRESHOLD = 0.75

    # ---- Safety reserve (adaptive) ----
    INITIAL_SAFETY_RESERVE: float = 0.15   # fraction of battery_max
    SAFETY_RESERVE_MIN: float = 0.10
    SAFETY_RESERVE_MAX: float = 0.40
    SAFETY_RESERVE_INCREMENT: float = 0.02
    SAFETY_RESERVE_DECREMENT: float = 0.2

    # ---- Idle patrol / exploration ----
    PATROL_COMM_FRACTION = 2              # neighborhood size used to avoid crowding
    PATROL_EXPLORATION_BUDGET_STEPS = 1     # assumed future patrol cost in feasibility check
    PATROL_TARGET_RADIUS = 5                # how far a patrol target may be from current position
    PATROL_RESELECT_STEPS = 3               # how long to keep a patrol target before reselecting
    PATROL_MIN_BATTERY_FRAC = 0.35          # don't patrol below this fraction of max battery

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