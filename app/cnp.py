from models import Request, RequestStatus, DroneState
from agent import DroneAgent


def allocate_request(request: Request, drone_agents: list[DroneAgent], charging_stations):
    bids = []

    for agent in drone_agents:
        bid = agent.compute_bid(request, charging_stations)
        if bid != float("inf"):
            bids.append((agent, bid))

    if not bids:
        request.status = RequestStatus.FAILED
        return None

    winner, best_cost = min(bids, key=lambda x: x[1])
    request.assigned_to = winner.drone.drone_id
    request.status = RequestStatus.ASSIGNED
    winner.drone.current_request = request.request_id
    winner.drone.state = DroneState.TO_PICKUP
    return winner