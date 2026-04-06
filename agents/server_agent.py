"""
ServerAgent — Mesa 3.0 compatible fixed server node.

Mesa 3.0 changes:
  - super().__init__(model) only — no unique_id arg
  - auto-registers with model.agents on construction
"""

try:
    import mesa
    _Agent = mesa.Agent
except ImportError:
    import mesa_lite as mesa
    _Agent = mesa.Agent


class ServerAgent(_Agent):

    def __init__(self, model, pos: tuple):
        super().__init__(model)   # Mesa 3.0: model only, no unique_id
        self.pos            = pos
        self.request_queue  = []
        self.total_received = 0

    def add_request(self, req):
        self.request_queue.append(req)
        self.total_received += 1

    def step(self):
        # Prune completed / failed requests from local queue
        self.request_queue = [
            r for r in self.request_queue
            if not r.is_completed and not r.is_failed
        ]

    @property
    def queue_depth(self) -> int:
        return len(self.request_queue)

    def __repr__(self):
        return f"Server({self.unique_id}, pos={self.pos}, queue={self.queue_depth})"