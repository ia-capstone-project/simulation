"""
ChargingStation

Drones navigate to these positions to recharge; the station itself
is passive (charging logic lives in DroneAgent._step_charging).
"""

try:
    import mesa
    _Agent = mesa.Agent
except ImportError:
    import mesa_lite as mesa
    _Agent = mesa.Agent


class ChargingStation(_Agent):

    def __init__(self, model, pos: tuple):
        super().__init__(model)   # Mesa 3.0: model only
        self.pos = pos

    def step(self):
        pass  # passive — no behaviour needed

    def __repr__(self):
        return f"ChargingStation({self.unique_id}, pos={self.pos})"