import random
from models import Position, Request, RequestStatus, Drone, DroneState
from agent import DroneAgent
from cnp import allocate_request


class Simulation:
    def __init__(self):
        self.time = 0
        self.width = 20
        self.height = 20

        self.charging_stations = [Position(0, 0), Position(19, 19)]
        self.requests = {}
        self.request_counter = 0

        self.drone_agents = [
            DroneAgent(Drone("D1", Position(2, 2))),
            DroneAgent(Drone("D2", Position(10, 5))),
            DroneAgent(Drone("D3", Position(15, 15))),
        ]

        self.success_count = 0
        self.failed_count = 0
        self.total_energy_used = 0

    def generate_request(self):
        pickup = Position(random.randint(0, 19), random.randint(0, 19))
        dropoff = Position(random.randint(0, 19), random.randint(0, 19))
        req = Request(
            request_id=f"R{self.request_counter}",
            pickup=pickup,
            dropoff=dropoff,
            created_at=self.time
        )
        self.requests[req.request_id] = req
        self.request_counter += 1
        return req

    def step(self):
        self.time += 1

        if random.random() < 0.15:
            req = self.generate_request()
            winner = allocate_request(req, self.drone_agents, self.charging_stations)
            if winner:
                print(f"[t={self.time}] {req.request_id} assigned to {winner.drone.drone_id}")
            else:
                print(f"[t={self.time}] {req.request_id} failed: no valid bids")

        for agent in self.drone_agents:
            drone = agent.drone

            if drone.current_request is None:
                if drone.battery < 30:
                    drone.state = DroneState.CHARGING
                    nearest = min(
                        self.charging_stations,
                        key=lambda c: abs(c.x - drone.position.x) + abs(c.y - drone.position.y)
                    )
                    if drone.position.x != nearest.x or drone.position.y != nearest.y:
                        old_battery = drone.battery
                        agent.move_one_step_towards(nearest)
                        self.total_energy_used += old_battery - drone.battery
                    else:
                        drone.battery = min(100, drone.battery + 5)
                        if drone.battery >= 90:
                            drone.state = DroneState.IDLE
                continue

            req = self.requests[drone.current_request]

            if drone.state == DroneState.TO_PICKUP:
                if drone.position.x == req.pickup.x and drone.position.y == req.pickup.y:
                    drone.carrying = True
                    drone.state = DroneState.TO_DROPOFF
                    req.status = RequestStatus.PICKED
                else:
                    old_battery = drone.battery
                    agent.move_one_step_towards(req.pickup)
                    self.total_energy_used += old_battery - drone.battery

            elif drone.state == DroneState.TO_DROPOFF:
                if drone.position.x == req.dropoff.x and drone.position.y == req.dropoff.y:
                    drone.carrying = False
                    drone.current_request = None
                    drone.state = DroneState.IDLE if drone.battery >= 30 else DroneState.CHARGING
                    req.status = RequestStatus.DELIVERED
                    self.success_count += 1
                else:
                    old_battery = drone.battery
                    agent.move_one_step_towards(req.dropoff)
                    self.total_energy_used += old_battery - drone.battery

            if drone.battery <= 0:
                req.status = RequestStatus.FAILED
                self.failed_count += 1
                drone.current_request = None
                drone.state = DroneState.IDLE
                print(f"[t={self.time}] {drone.drone_id} failed due to battery depletion")

    def get_state(self):
        return {
            "time": self.time,
            "drones": [
                {
                    "id": a.drone.drone_id,
                    "x": a.drone.position.x,
                    "y": a.drone.position.y,
                    "battery": a.drone.battery,
                    "state": a.drone.state,
                    "request": a.drone.current_request,
                }
                for a in self.drone_agents
            ],
            "requests": [
                {
                    "id": r.request_id,
                    "pickup": (r.pickup.x, r.pickup.y),
                    "dropoff": (r.dropoff.x, r.dropoff.y),
                    "status": r.status,
                    "assigned_to": r.assigned_to,
                }
                for r in self.requests.values()
            ],
            "metrics": {
                "success": self.success_count,
                "failed": self.failed_count,
                "energy_used": self.total_energy_used,
            }
        }