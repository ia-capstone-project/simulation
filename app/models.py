from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class DroneState(str, Enum):
    IDLE = "IDLE"
    BIDDING = "BIDDING"
    TO_PICKUP = "TO_PICKUP"
    TO_DROPOFF = "TO_DROPOFF"
    CHARGING = "CHARGING"


class RequestStatus(str, Enum):
    NEW = "NEW"
    ANNOUNCED = "ANNOUNCED"
    ASSIGNED = "ASSIGNED"
    PICKED = "PICKED"
    DELIVERED = "DELIVERED"
    FAILED = "FAILED"


@dataclass
class Position:
    x: int
    y: int


@dataclass
class Request:
    request_id: int
    pickup: Position
    dropoff: Position
    created_at: int
    status: RequestStatus = RequestStatus.NEW
    assigned_to: Optional[int] = None


@dataclass
class Drone:
    drone_id: int
    position: Position
    battery: float = 100.0
    state: DroneState = DroneState.IDLE
    current_request: Optional[int] = None
    carrying: bool = False