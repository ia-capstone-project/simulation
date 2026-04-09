"""
DroneAgent — Mesa 3.0 compatible.

Mesa 3.0 changes applied:
  - super().__init__(model)  only — no unique_id argument
  - Agent auto-registers with model.agents on construction
  - No schedule.add() anywhere

Drone roles (dynamic, per CNP task):
  IDLE               → drifts toward servers; available as manager
  MANAGER            → running a CFP round for one request
  CONTRACTOR_WAITING → bid submitted; awaiting award/reject
  DELIVERING         → flying pickup → dropoff
  CHARGING           → flying to / sitting at charging station

Adaptive learning (parameter adaptation, NOT RL):
  Weights α, β, γ adjust based on rolling delivery success rate.
  Safety reserve tightens after battery depletions, relaxes after
  clean runs.
"""

try:
    import mesa
    _Agent = mesa.Agent
except ImportError:
    import mesa_lite as mesa
    _Agent = mesa.Agent

import numpy as np
from enum import Enum, auto

from protocols.cnp_protocol import (
    CFPMessage, ProposalMessage, AwardMessage,
    RejectMessage, CNPRound, CNPMessageType,
)
from config.settings import SimConfig


class DroneState(Enum):
    IDLE               = auto()
    MANAGER            = auto()
    CONTRACTOR_WAITING = auto()
    DELIVERING         = auto()
    CHARGING           = auto()


class DroneAgent(_Agent):

    def __init__(self, model, start_pos):
        # Mesa 3.0: only model argument; unique_id auto-assigned
        super().__init__(model)
        self.pos = start_pos

        # ---- Battery -------------------------------------------------- #
        self.battery        = float(SimConfig.BATTERY_MAX)
        self.battery_max    = float(SimConfig.BATTERY_MAX)
        self.battery_drain  = SimConfig.BATTERY_DRAIN_MOVE
        self.safety_reserve = SimConfig.INITIAL_SAFETY_RESERVE   # adaptive

        # ---- State ---------------------------------------------------- #
        self.state             = DroneState.IDLE
        self.current_request   = None
        self.current_cnp_round = None
        self.target_pos        = None
        self.delivery_phase    = "pickup"   # "pickup" | "dropoff"
        self.charging_target   = None

        # ---- Adaptive utility weights --------------------------------- #
        self.alpha = SimConfig.ALPHA_INIT
        self.beta  = SimConfig.BETA_INIT
        self.gamma = SimConfig.GAMMA_INIT

        # ---- Performance tracking ------------------------------------- #
        self.completed_tasks  = 0
        self.failed_tasks     = 0
        self.last_utility     = None
        self.recent_outcomes  = []   # 1=success, 0=fail (sliding window)

        # ---- Activity / starvation tracking -------------------------------- #
        self.steps_since_delivery = 0
        self.steps_since_bid = 0
        self.consecutive_bid_failures = 0   # CFP seen but could not bid
        self.consecutive_rejections = 0     # bid sent but not awarded
        self.idle_starvation_steps = 0

        # ---- Idle patrol / exploration -------------------------------------- #
        self.patrol_target = None
        self.patrol_steps_remaining = 0

        # ---- Message inbox -------------------------------------------- #
        self._inbox = []

    # ================================================================== #
    #  Public interface called by the Model                               #
    # ================================================================== #

    def is_available_as_manager(self) -> bool:
        """True only when idle and battery comfortably above safety reserve."""
        return (
            self.state == DroneState.IDLE
            and self.battery > self.safety_reserve * self.battery_max
        )

    def become_manager(self, req):
        """Called by model.assign_managers() to designate this drone as Manager."""
        self.state             = DroneState.MANAGER
        self.current_request   = req
        self.current_cnp_round = CNPRound(
            request=req,
            cfp_step=self.model.step_count,
        )
        self.patrol_target = None
        self.patrol_steps_remaining = 0

    def issue_cfp(self, req):
        """Broadcast a CFP to every drone within comm_range (excluding self)."""
        cfp = CFPMessage(
            manager_id=self.unique_id,
            request=req,
            deadline_step=self.model.step_count + SimConfig.CFP_DEADLINE_STEPS,
        )
        for drone in self.model.get_drones_in_range(self.pos, self.model.comm_range):
            if drone.unique_id != self.unique_id:
                drone.receive_message(cfp)

    def receive_message(self, msg):
        """External agents push messages here; processed at start of step()."""
        self._inbox.append(msg)

    # ================================================================== #
    #  Mesa 3.0 step — called by model.agents.shuffle_do("step")         #
    # ================================================================== #

    def step(self):
        self._update_starvation_counters()
        self._process_inbox()

        if self._needs_preemptive_charging():
            self._initiate_charging()
            return

        if self._needs_charging():
            self._initiate_charging()
            return

        if   self.state == DroneState.IDLE:
            self._step_idle()
        elif self.state == DroneState.MANAGER:
            self._step_manager()
        elif self.state == DroneState.CONTRACTOR_WAITING:
            pass   # waiting for award/reject — nothing to do
        elif self.state == DroneState.DELIVERING:
            self._step_delivering()
        elif self.state == DroneState.CHARGING:
            self._step_charging()

    # ================================================================== #
    #  Inbox processing                                                    #
    # ================================================================== #

    def _process_inbox(self):
        messages, self._inbox = list(self._inbox), []
        for msg in messages:
            if   msg.message_type == CNPMessageType.CFP:
                self._handle_cfp(msg)
            elif msg.message_type == CNPMessageType.PROPOSE:
                self._handle_proposal(msg)
            elif msg.message_type == CNPMessageType.ACCEPT:
                self._handle_award(msg)
            elif msg.message_type == CNPMessageType.REJECT:
                self._handle_reject(msg)

    def _handle_cfp(self, cfp):
        """Contractor: evaluate and bid (or silently refuse if busy/low battery)."""
        if self.state != DroneState.IDLE:
            return
        req = cfp.request
        if not self._battery_feasible(req):
            self.consecutive_bid_failures += 1
            return

        utility = self._compute_utility(req)
        self.last_utility = utility

        proposal = ProposalMessage(
            contractor_id=self.unique_id,
            request_id=req.request_id,
            utility_score=utility,
            estimated_distance=self._task_distance(req),
            battery_level=self.battery,
        )
        manager = self._find_agent(cfp.manager_id)
        if manager:
            manager.receive_message(proposal)
            self.state = DroneState.CONTRACTOR_WAITING
            self.steps_since_bid = 0
            self.consecutive_bid_failures = 0
            self.patrol_target = None
            self.patrol_steps_remaining = 0

    def _handle_proposal(self, proposal):
        """Manager: collect incoming bid."""
        if self.state != DroneState.MANAGER or self.current_cnp_round is None:
            bidder = self._find_agent(proposal.contractor_id)
            if bidder:
                bidder.receive_message(
                    RejectMessage(
                        manager_id=self.unique_id,
                        contractor_id=proposal.contractor_id,
                        request_id=proposal.request_id,
                    )
                )
            return
        if proposal.request_id == self.current_request.request_id:
            self.current_cnp_round.add_proposal(proposal)

    def _handle_award(self, award):
        """Winner starts delivering; loser returns to idle."""
        if award.winner_id == self.unique_id:
            self.state           = DroneState.DELIVERING
            self.current_request = award.request
            self.target_pos      = award.request.pickup_pos
            self.delivery_phase  = "pickup"
            self.consecutive_rejections = 0
            self.consecutive_bid_failures = 0
            self.steps_since_bid = 0
            self.patrol_target = None
            self.patrol_steps_remaining = 0
        else:
            self.state = DroneState.IDLE

    def _handle_reject(self, _msg):
        self.consecutive_rejections += 1
        self.state = DroneState.IDLE

    def _manager_proposal(self):
        """Manager's own bid for its request (called at CFP awarding)."""
        req = self.current_request
        if not self._battery_feasible(req):
            self.consecutive_bid_failures += 1
            return None
        
        utility = self._compute_utility(req)
        self.last_utility = utility
        message = ProposalMessage(
            contractor_id=self.unique_id,
            request_id=req.request_id,
            utility_score=utility,
            estimated_distance=self._task_distance(req),
            battery_level=self.battery,
        )
        self.current_cnp_round.add_proposal(message)
        self.consecutive_bid_failures = 0


    # ================================================================== #
    #  Manager FSM                                                         #
    # ================================================================== #

    def _step_manager(self):
        if self.current_cnp_round is None:
            self.state = DroneState.IDLE
            return
        if self.current_cnp_round.is_ready_to_evaluate(self.model.step_count):
            self._evaluate_and_award()

    def _evaluate_and_award(self):
        current_cnp_round = self.current_cnp_round
        self._manager_proposal()  # manager adds its own bid before evaluating
        req  = self.current_request
        best = current_cnp_round.best_proposal()

        # Manager returns to idle after awarding
        self.state             = DroneState.IDLE
        self.current_request   = None
        self.current_cnp_round = None

        if best is None:
            # No bids - return request to pending pool and go idle
            self.model.pending_requests.append(req)
            self.model.active_requests.pop(req.request_id, None)
            req.manager_id = None
            return

        winner_id             = best.contractor_id
        req.assigned_drone_id = winner_id

        award = AwardMessage(
            manager_id=self.unique_id,
            winner_id=winner_id,
            request_id=req.request_id,
            request=req,
        )
        for proposal in current_cnp_round.proposals:
            agent = self._find_agent(proposal.contractor_id)
            if agent:
                if proposal.contractor_id == winner_id:
                    agent.receive_message(award)
                else:
                    agent.receive_message(
                        RejectMessage(
                            manager_id=self.unique_id,
                            contractor_id=proposal.contractor_id,
                            request_id=req.request_id,
                        )
                    )


    # ================================================================== #
    #  Delivery FSM                                                        #
    # ================================================================== #

    def _step_delivering(self):
        if self.target_pos is None or self.current_request is None:
            self.state = DroneState.IDLE
            return

        self._move_toward(self.target_pos)

        if self.pos == self.target_pos:
            if self.delivery_phase == "pickup":
                self.delivery_phase = "dropoff"
                self.target_pos     = self.current_request.delivery_pos
            else:
                self._complete_delivery()

    def _complete_delivery(self):
        self.model.mark_completed(self.current_request.request_id)
        self.completed_tasks += 1
        self.recent_outcomes.append(1)
        self._adapt_weights(success=True)

        self.steps_since_delivery = 0
        self.idle_starvation_steps = 0
        self.consecutive_bid_failures = 0
        self.consecutive_rejections = 0

        self.current_request = None
        self.target_pos      = None
        self.state           = DroneState.IDLE

    # ================================================================== #
    #  Idle behaviour                                                      #
    # ================================================================== #

    def _step_idle(self):
        """
        Patrol to improve spatial coverage for communication-range-based
        request discovery, but only if battery safely allows it.
        """
        if not self._battery_feasible_for_patrol():
            return

        # Reuse an existing patrol target for a few steps, otherwise pick a new one
        if (
            self.patrol_target is None
            or self.pos == self.patrol_target
            or self.patrol_steps_remaining <= 0
        ):
            self.patrol_target = self._choose_patrol_target()
            self.patrol_steps_remaining = SimConfig.PATROL_RESELECT_STEPS

        if self.patrol_target is None:
            return

        self._move_toward(self.patrol_target)
        self.patrol_steps_remaining -= 1

    # ================================================================== #
    # Patrolling                                                         #       
    # ================================================================== #
    def _battery_feasible_for_patrol(self) -> bool:
        """
        Patrol only if the drone can spend a small exploration budget,
        then still reach the nearest charging station while preserving reserve.
        """
        if self.battery < self.battery_max * SimConfig.PATROL_MIN_BATTERY_FRAC:
            return False

        nearest_cs = self.model.nearest_charging_station(self.pos)
        dist_to_cs = self.model.manhattan(self.pos, nearest_cs)

        required = (
            (dist_to_cs + SimConfig.PATROL_EXPLORATION_BUDGET_STEPS) * self.battery_drain
            + self.safety_reserve * self.battery_max
        )
        return self.battery >= required

    def _choose_patrol_target(self):
        x, y = self.pos
        radius = SimConfig.PATROL_TARGET_RADIUS

        candidates = []
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx == 0 and dy == 0:
                    continue

                cx, cy = x + dx, y + dy
                if not (0 <= cx < self.model.width and 0 <= cy < self.model.height):
                    continue

                candidates.append((cx, cy))

        if not candidates:
            return None

        # Use full comm range for connectivity reasoning
        neighbors = [
            d for d in self.model.get_drones_in_range(self.pos, self.model.comm_range)
            if d.unique_id != self.unique_id and d.state == DroneState.IDLE
        ]

        if not neighbors:
            return None

        desired_dist = max(1, int(0.8 * self.model.comm_range))
        min_safe_dist = max(1, int(0.55 * self.model.comm_range))
        max_safe_dist = self.model.comm_range

        def score(cell):
            dists = [self.model.manhattan(cell, d.pos) for d in neighbors]

            # Must remain connected to at least one neighbor
            if min(dists) > max_safe_dist:
                return -10**9

            # Prefer being near the outer communication band, not too close, not too far
            edge_score = -min(abs(d - desired_dist) for d in dists)

            # Mild penalty if too many drones are very close
            close_count = sum(1 for d in dists if d < min_safe_dist)

            # Small reward for having 1-2 reachable neighbors
            connected_count = sum(1 for d in dists if d <= max_safe_dist)
            connectivity_bonus = min(connected_count, 2)

            jitter = self.random.random() * 0.1

            return 3.0 * edge_score - 2.0 * close_count + connectivity_bonus + jitter

        best_cell = max(candidates, key=score)
        return best_cell
    
    # ================================================================== #
    #  Charging                                                          #
    # ================================================================== #

    def _update_starvation_counters(self):
        if self.state != DroneState.DELIVERING:
            self.steps_since_delivery += 1
        else:
            self.steps_since_delivery = 0

        if self.state == DroneState.IDLE:
            self.idle_starvation_steps += 1
        else:
            self.idle_starvation_steps = 0

        if self.state != DroneState.CONTRACTOR_WAITING:
            self.steps_since_bid += 1

    def _needs_preemptive_charging(self) -> bool:
        if self.state != DroneState.IDLE:
            return False

        # Only do this when there is active demand in the system
        active_request_count = (
            len(self.model.pending_requests)
        )
        if active_request_count == 0:
            return False

        # Request pressure: stricter when many requests are coming in
        pressure = min(active_request_count, 5)

        long_idle = self.idle_starvation_steps >= (
            SimConfig.STARVATION_IDLE_THRESHOLD - pressure * 2
        )

        repeated_bid_failure = (
            self.consecutive_bid_failures >= SimConfig.BID_FAILURE_THRESHOLD
        )

        # If demand exists and this drone keeps being ineffective, top it up
        # so it can compete better in future rounds.
        return (
            self.battery < self.battery_max * SimConfig.PREEMPTIVE_CHARGE_BATTERY_THRESHOLD
            and (
                long_idle
                or repeated_bid_failure
            )
        )


    def _needs_charging(self) -> bool:
        if self.state == DroneState.CHARGING:
            return False
        nearest_cs = self.model.nearest_charging_station(self.pos)
        dist_to_cs = self.model.manhattan(self.pos, nearest_cs)
        usable     = self.battery - self.safety_reserve * self.battery_max
        return usable <= dist_to_cs * self.battery_drain

    def _initiate_charging(self):
        self.patrol_target = None
        self.patrol_steps_remaining = 0
        if self.state == DroneState.DELIVERING and self.current_request:
            req = self.current_request
            self.model.active_requests.pop(req.request_id, None)
            self.model.failed_deliveries += 1
            self.model.report_battery_depletion()
            self.failed_tasks += 1
            self.recent_outcomes.append(0)
            self._adapt_weights(success=False)
            self.current_request = None

        self.charging_target = self.model.nearest_charging_station(self.pos)
        self.state           = DroneState.CHARGING
        # Tighten safety reserve after depletion
        self.safety_reserve = min(
            self.safety_reserve + SimConfig.SAFETY_RESERVE_INCREMENT,
            SimConfig.SAFETY_RESERVE_MAX,
        )

    def _step_charging(self):
        if self.charging_target and self.pos != self.charging_target:
            self._move_toward(self.charging_target)
        else:
            self.battery = min(
                self.battery + SimConfig.CHARGE_RATE, self.battery_max
            )
            if self.battery >= self.battery_max * SimConfig.CHARGE_FULL_THRESHOLD:
                self.state           = DroneState.IDLE
                self.charging_target = None

    # ================================================================== #
    #  Utility function                                                    #
    # ================================================================== #

    def _compute_utility(self, req) -> float:
        """
        U = α * priority  −  β * d_norm  −  γ * (1 − battery_norm)

        d_norm       = (dist_to_pickup + dist_to_delivery) / grid_diagonal
        battery_norm = battery / battery_max

        α, β, γ are per-drone adaptive weights.
        """
        d_pickup   = self.model.manhattan(self.pos, req.pickup_pos)
        d_delivery = self.model.manhattan(req.pickup_pos, req.delivery_pos)
        diagonal   = self.model.width + self.model.height
        d_norm     = (d_pickup + d_delivery) / diagonal
        bat_norm   = self.battery / self.battery_max

        return (
            self.alpha * req.priority
            - self.beta  * d_norm
            - self.gamma * (1.0 - bat_norm)
        )

    def _task_distance(self, req) -> int:
        return (
            self.model.manhattan(self.pos, req.pickup_pos)
            + self.model.manhattan(req.pickup_pos, req.delivery_pos)
        )

    def _battery_feasible(self, req) -> bool:
        """
        Can this drone complete the task AND reach a charger afterwards
        while keeping at least safety_reserve * battery_max?
        """
        task_dist  = self._task_distance(req)
        nearest_cs = self.model.nearest_charging_station(req.delivery_pos)
        cs_dist    = self.model.manhattan(req.delivery_pos, nearest_cs)
        drain      = (task_dist + cs_dist) * self.battery_drain
        required   = drain + self.safety_reserve * self.battery_max
        return self.battery >= required

    # ================================================================== #
    #  Adaptive weight adjustment (parameter adaptation, not RL)          #
    # ================================================================== #

    def _adapt_weights(self, success: bool):
        """
        Adjust α, β, γ and safety_reserve based on rolling success rate.

        Low success  → more conservative (higher β/γ, lower α, tighter reserve)
        High success → slightly more aggressive (lower β/γ, higher α, looser reserve)
        """
        window = SimConfig.ADAPTATION_WINDOW
        if len(self.recent_outcomes) > window:
            self.recent_outcomes = self.recent_outcomes[-window:]

        if len(self.recent_outcomes) < 5:
            return  # need enough samples first

        sr = float(np.mean(self.recent_outcomes))
        lr = SimConfig.WEIGHT_LEARNING_RATE

        if sr < 0.5:
            self.beta  = min(self.beta  + lr,       SimConfig.BETA_MAX)
            self.gamma = min(self.gamma + lr,       SimConfig.GAMMA_MAX)
            self.alpha = max(self.alpha - lr * 0.5, SimConfig.ALPHA_MIN)
            self.safety_reserve = min(
                self.safety_reserve + SimConfig.SAFETY_RESERVE_INCREMENT,
                SimConfig.SAFETY_RESERVE_MAX,
            )
        elif sr > 0.8:
            self.beta  = max(self.beta  - lr * 0.3, SimConfig.BETA_MIN)
            self.gamma = max(self.gamma - lr * 0.3, SimConfig.GAMMA_MIN)
            self.alpha = min(self.alpha + lr * 0.2, SimConfig.ALPHA_MAX)
            self.safety_reserve = max(
                self.safety_reserve - SimConfig.SAFETY_RESERVE_DECREMENT,
                SimConfig.SAFETY_RESERVE_MIN,
            )

    # ================================================================== #
    #  Movement                                                            #
    # ================================================================== #

    def _move_toward(self, target):
        if self.pos == target:
            return
        # drain battery only if moving
        self._drain_battery()
        x, y   = self.pos
        tx, ty = target
        dx, dy = tx - x, ty - y

        if abs(dx) >= abs(dy):
            new_pos = (x + int(np.sign(dx)), y)
        else:
            new_pos = (x, y + int(np.sign(dy)))

        new_pos = (
            max(0, min(new_pos[0], self.model.width  - 1)),
            max(0, min(new_pos[1], self.model.height - 1)),
        )
        self.model.grid.move_agent(self, new_pos)

    def _drain_battery(self):
        self.battery = max(0.0, self.battery - self.battery_drain)
        if self.battery == 0.0:
            self.model.report_battery_depletion()

    # ================================================================== #
    #  Helpers                                                             #
    # ================================================================== #

    def _find_agent(self, agent_id):
        """Look up any agent by unique_id via model.agents (Mesa 3.0 AgentSet)."""
        for agent in self.model.agents:
            if agent.unique_id == agent_id:
                return agent
        return None

    def __repr__(self):
        return (
            f"Drone({self.unique_id}, {self.state.name}, "
            f"bat={self.battery:.1f}, α={self.alpha:.2f})"
        )