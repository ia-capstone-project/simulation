"""
Drone Delivery Simulation - Contract Net Protocol (CNP)
Mesa-based multi-agent simulation
"""

try:
    import mesa
    _Agent = mesa.Agent
    _Model = mesa.Model
    _MultiGrid = mesa.space.MultiGrid
    _RandomActivation = mesa.time.RandomActivation
    _DataCollector = mesa.DataCollector
except ImportError:
    import mesa_lite as mesa
    _Agent = mesa.Agent
    _Model = mesa.Model
    _MultiGrid = mesa.MultiGrid
    _RandomActivation = mesa.RandomActivation
    _DataCollector = mesa.DataCollector

import numpy as np
from agents.drone_agent import DroneAgent
from agents.server_agent import ServerAgent
from agents.charging_station import ChargingStation
from protocols.cnp_protocol import DeliveryRequest
from config.settings import SimConfig


class DroneDeliveryModel(_Model):
    """
    Main simulation model for drone delivery using Contract Net Protocol.

    CNP Manager Assignment (key design):
    ─────────────────────────────────────
    Each step, BEFORE drones act, assign_managers() is called.
    For each pending request, the model finds the closest IDLE drone
    within comm_range of the request pickup. That drone is assigned
    the Manager role and immediately issues a CFP to all neighbours.

    Why the MODEL does this (not the drones themselves):
    - Prevents race conditions: two drones can't both claim manager
      for the same request in the same step.
    - Matches CNP spec: a 3rd-party (server/environment) designates
      the initiator. Drones remain decentralised for bidding/execution.
    """

    def __init__(
        self,
        width=SimConfig.GRID_WIDTH,
        height=SimConfig.GRID_HEIGHT,
        num_drones=SimConfig.NUM_DRONES,
        num_servers=SimConfig.NUM_SERVERS,
        num_charging_stations=SimConfig.NUM_CHARGING_STATIONS,
        request_rate=SimConfig.REQUEST_RATE,
        comm_range=SimConfig.COMM_RANGE,
        seed=None,
    ):
        super().__init__(seed=seed)

        self.width = width
        self.height = height
        self.num_drones = num_drones
        self.num_servers = num_servers
        self.num_charging_stations = num_charging_stations
        self.request_rate = request_rate
        self.comm_range = comm_range

        self.grid = _MultiGrid(width, height, torus=False)
        self.schedule = _RandomActivation(self)

        self.pending_requests = []
        self.active_requests = {}
        self.completed_deliveries = 0
        self.failed_deliveries = 0
        self.total_requests = 0
        self.step_count = 0
        self.delivery_times = []
        self.battery_depletions = 0

        self.datacollector = _DataCollector(
            model_reporters={
                "Completed":         lambda m: m.completed_deliveries,
                "Failed":            lambda m: m.failed_deliveries,
                "Pending":           lambda m: len(m.pending_requests),
                "Active":            lambda m: len(m.active_requests),
                "BatteryDepletions": lambda m: m.battery_depletions,
                "AvgBattery":        self._avg_battery,
            },
        )

        self._place_charging_stations()
        self._place_servers()
        self._place_drones()

        self.charging_station_positions = [
            cs.pos for cs in self.agents_by_type[ChargingStation]
        ]
        self.server_agents = list(self.agents_by_type[ServerAgent])
        self.drone_agents  = list(self.agents_by_type[DroneAgent])

    # ---- Placement ---------------------------------------------------- #

    def _place_charging_stations(self):
        for pos in self._spread_positions(self.num_charging_stations, margin=2):
            cs = ChargingStation(self, pos)
            self._register_agent(cs)
            self.grid.place_agent(cs, pos)
            self.schedule.add(cs)

    def _place_servers(self):
        cols = int(np.ceil(np.sqrt(self.num_servers)))
        rows = int(np.ceil(self.num_servers / cols))
        step_x = max(1, self.width  // (cols + 1))
        step_y = max(1, self.height // (rows + 1))
        placed = 0
        for r in range(1, rows + 1):
            for c in range(1, cols + 1):
                if placed >= self.num_servers:
                    break
                pos = (c * step_x, r * step_y)
                sv = ServerAgent(self, pos)
                self._register_agent(sv)
                self.grid.place_agent(sv, pos)
                self.schedule.add(sv)
                placed += 1

    def _place_drones(self):
        cs_list = list(self.agents_by_type[ChargingStation])
        for i in range(self.num_drones):
            cs = cs_list[i % len(cs_list)]
            drone = DroneAgent(self, cs.pos)
            self._register_agent(drone)
            self.grid.place_agent(drone, cs.pos)
            self.schedule.add(drone)

    def _spread_positions(self, n, margin=1):
        positions = set()
        attempts = 0
        while len(positions) < n and attempts < n * 200:
            x = self.random.randint(margin, self.width  - 1 - margin)
            y = self.random.randint(margin, self.height - 1 - margin)
            positions.add((x, y))
            attempts += 1
        return list(positions)

    # ---- Request Generation ------------------------------------------- #

    def _generate_requests(self):
        for server in self.server_agents:
            if self.random.random() < self.request_rate:
                dest = (
                    self.random.randint(0, self.width  - 1),
                    self.random.randint(0, self.height - 1),
                )
                req = DeliveryRequest(
                    request_id=self.total_requests,
                    pickup_pos=server.pos,
                    delivery_pos=dest,
                    created_step=self.step_count,
                    priority=self.random.uniform(0.5, 1.5),
                )
                self.pending_requests.append(req)
                self.total_requests += 1
                server.add_request(req)

    # ---- CNP Manager Assignment --------------------------------------- #

    def assign_managers(self):
        """
        For each pending request, designate the closest eligible drone
        as Manager. The manager then issues a CFP to its neighbours.

        Called once per step, before the agent schedule runs, ensuring
        no two drones can race to claim the same request.
        """
        still_pending = []

        for req in self.pending_requests:
            if req.request_id in self.active_requests:
                continue

            manager = self._select_manager(req)

            if manager is not None:
                req.manager_id = manager.unique_id
                self.active_requests[req.request_id] = req
                manager.become_manager(req)   # sets state = MANAGER
                manager.issue_cfp(req)        # broadcasts CFP to neighbours
            else:
                still_pending.append(req)

        self.pending_requests = still_pending

    def _select_manager(self, req):
        """
        Select the best idle drone for the manager role.

        Rules:
          1. DroneState.IDLE only
          2. Battery above safety threshold
          3. Within comm_range (Manhattan) of the pickup
          4. Closest first; ties broken by unique_id (lowest wins)
        """
        best = None
        best_dist = float("inf")

        for drone in self.drone_agents:
            if not drone.is_available_as_manager():
                continue
            dist = self.manhattan(drone.pos, req.pickup_pos)
            if dist > self.comm_range:
                continue
            if dist < best_dist or (
                dist == best_dist and best and drone.unique_id < best.unique_id
            ):
                best_dist = dist
                best = drone

        return best

    # ---- Utilities ---------------------------------------------------- #

    def manhattan(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def nearest_charging_station(self, pos):
        if not self.charging_station_positions:
            return (0, 0)
        return min(self.charging_station_positions,
                   key=lambda cs: self.manhattan(pos, cs))

    def get_drones_in_range(self, pos, radius):
        return [d for d in self.drone_agents
                if self.manhattan(d.pos, pos) <= radius]

    def _avg_battery(self):
        if not self.drone_agents:
            return 0.0
        return float(np.mean([d.battery for d in self.drone_agents]))

    # ---- Step --------------------------------------------------------- #

    def step(self):
        self.step_count += 1
        self._generate_requests()
        self.assign_managers()       # ← manager assignment BEFORE agents act
        self.schedule.step()
        self.datacollector.collect(self)
        self._expire_old_requests()

    def _expire_old_requests(self):
        expired = [
            rid for rid, req in list(self.active_requests.items())
            if (self.step_count - req.created_step) > SimConfig.REQUEST_TIMEOUT
            and not req.is_completed
        ]
        for rid in expired:
            req = self.active_requests.pop(rid)
            req.is_failed = True
            self.failed_deliveries += 1

    def mark_completed(self, request_id):
        req = self.active_requests.pop(request_id, None)
        if req:
            req.is_completed = True
            self.delivery_times.append(self.step_count - req.created_step)
            self.completed_deliveries += 1

    def report_battery_depletion(self):
        self.battery_depletions += 1