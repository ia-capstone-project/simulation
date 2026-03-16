from models import Drone, Request, DroneState, Position
from utils import manhattan


class DroneAgent:
    def __init__(self, drone: Drone):
        self.drone = drone

    def compute_bid(self, request: Request, charging_stations: list[Position]) -> float:
        if self.drone.state not in [DroneState.IDLE, DroneState.BIDDING]:
            return float("inf")

        dist_to_pickup = manhattan(self.drone.position, request.pickup)
        delivery_dist = manhattan(request.pickup, request.dropoff)

        nearest_charger_dist = min(
            manhattan(request.dropoff, charger) for charger in charging_stations
        ) if charging_stations else 0

        required_energy = dist_to_pickup + delivery_dist + nearest_charger_dist
        if self.drone.battery < required_energy:
            return float("inf")

        battery_penalty = max(0, 30 - self.drone.battery)
        cost = dist_to_pickup + 0.5 * delivery_dist + 2.0 * battery_penalty
        return cost

    def move_one_step_towards(self, target: Position):
        if self.drone.position.x < target.x:
            self.drone.position.x += 1
        elif self.drone.position.x > target.x:
            self.drone.position.x -= 1
        elif self.drone.position.y < target.y:
            self.drone.position.y += 1
        elif self.drone.position.y > target.y:
            self.drone.position.y -= 1

        self.drone.battery -= 1