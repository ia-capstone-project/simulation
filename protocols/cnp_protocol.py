"""
Contract Net Protocol (CNP) data structures and message types.

Flow per request:
  Server detects request → Model assigns Manager (closest drone in range)
  → Manager issues CFP to neighbours → Contractors compute & submit bids
  → Manager evaluates bids → Awards contract to winner
  → Winner executes delivery
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum, auto


class CNPMessageType(Enum):
    CFP = auto()           # Call For Proposals
    PROPOSE = auto()       # Contractor submits bid
    ACCEPT = auto()        # Manager awards contract
    REJECT = auto()        # Manager rejects bid
    INFORM_DONE = auto()   # Winner reports completion
    REFUSE = auto()        # Contractor declines to bid


@dataclass
class DeliveryRequest:
    request_id: int
    pickup_pos: tuple
    delivery_pos: tuple
    created_step: int
    priority: float = 1.0

    # Set when a manager is assigned
    manager_id: Optional[int] = None

    # Lifecycle flags
    is_completed: bool = False
    is_failed: bool = False

    # Assigned contractor
    assigned_drone_id: Optional[int] = None


@dataclass
class CFPMessage:
    """Call For Proposals broadcast by the Manager."""
    message_type: CNPMessageType = CNPMessageType.CFP
    manager_id: int = -1
    request: Optional[DeliveryRequest] = None
    deadline_step: int = 0          # Step by which bids must arrive


@dataclass
class ProposalMessage:
    """Bid submitted by a Contractor."""
    message_type: CNPMessageType = CNPMessageType.PROPOSE
    contractor_id: int = -1
    request_id: int = -1
    utility_score: float = 0.0
    # Extra info manager may inspect
    estimated_distance: int = 0
    battery_level: float = 0.0


@dataclass
class AwardMessage:
    """Contract award sent by Manager to the winning Contractor."""
    message_type: CNPMessageType = CNPMessageType.ACCEPT
    manager_id: int = -1
    winner_id: int = -1
    request_id: int = -1
    request: Optional[DeliveryRequest] = None


@dataclass
class RejectMessage:
    message_type: CNPMessageType = CNPMessageType.REJECT
    manager_id: int = -1
    contractor_id: int = -1
    request_id: int = -1


@dataclass
class CNPRound:
    """
    Tracks a single CNP negotiation round for one delivery request.
    Held by the Manager drone.
    """
    request: DeliveryRequest
    cfp_step: int                              # Step CFP was issued
    proposals: list = field(default_factory=list)   # List[ProposalMessage]
    awarded: bool = False
    winner_id: Optional[int] = None
    evaluation_step: Optional[int] = None

    # How many steps to wait for proposals before evaluating
    waiting_window: int = 3

    def is_ready_to_evaluate(self, current_step: int) -> bool:
        return (current_step - self.cfp_step) >= self.waiting_window

    def add_proposal(self, proposal: ProposalMessage):
        self.proposals.append(proposal)

    def best_proposal(self) -> Optional[ProposalMessage]:
        if not self.proposals:
            return None
        return max(self.proposals, key=lambda p: p.utility_score)