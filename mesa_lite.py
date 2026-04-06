"""
mesa_lite.py — Minimal Mesa-compatible base classes.

Drop-in replacements for mesa.Agent, mesa.Model,
mesa.space.MultiGrid, and mesa.time.RandomActivation.

This lets the simulation run without mesa installed,
while keeping the same API so you can swap in real Mesa
simply by removing this file and adjusting imports.
"""

import random
import numpy as np
from collections import defaultdict


# ------------------------------------------------------------------ #
#  Agent                                                               #
# ------------------------------------------------------------------ #

class Agent:
    _id_counter = 0

    def __init__(self, model):
        Agent._id_counter += 1
        self.unique_id: int = Agent._id_counter
        self.model = model
        self.pos: tuple = None

    def step(self):
        pass


# ------------------------------------------------------------------ #
#  Schedulers                                                          #
# ------------------------------------------------------------------ #

class RandomActivation:
    def __init__(self, model):
        self.model = model
        self._agents: dict[int, Agent] = {}

    def add(self, agent: Agent):
        self._agents[agent.unique_id] = agent

    def remove(self, agent: Agent):
        self._agents.pop(agent.unique_id, None)

    @property
    def agents(self):
        return list(self._agents.values())

    def step(self):
        agents = list(self._agents.values())
        random.shuffle(agents)
        for agent in agents:
            agent.step()


# ------------------------------------------------------------------ #
#  MultiGrid                                                           #
# ------------------------------------------------------------------ #

class MultiGrid:
    def __init__(self, width: int, height: int, torus: bool = False):
        self.width = width
        self.height = height
        self.torus = torus
        # cell_contents[(x,y)] -> list of agents
        self._grid: dict[tuple, list] = defaultdict(list)

    def place_agent(self, agent: Agent, pos: tuple):
        agent.pos = pos
        self._grid[pos].append(agent)

    def move_agent(self, agent: Agent, new_pos: tuple):
        if agent.pos and agent in self._grid[agent.pos]:
            self._grid[agent.pos].remove(agent)
        agent.pos = new_pos
        self._grid[new_pos].append(agent)

    def remove_agent(self, agent: Agent):
        if agent.pos:
            self._grid[agent.pos].discard(agent)

    def get_cell_list_contents(self, positions: list) -> list:
        result = []
        for pos in positions:
            result.extend(self._grid.get(pos, []))
        return result

    def is_in_bounds(self, pos: tuple) -> bool:
        return 0 <= pos[0] < self.width and 0 <= pos[1] < self.height


# ------------------------------------------------------------------ #
#  DataCollector (simplified)                                          #
# ------------------------------------------------------------------ #

class DataCollector:
    def __init__(self, model_reporters=None, agent_reporters=None):
        self.model_reporters = model_reporters or {}
        self.agent_reporters = agent_reporters or {}
        self._model_data: dict[str, list] = {k: [] for k in self.model_reporters}

    def collect(self, model):
        for key, fn in self.model_reporters.items():
            try:
                self._model_data[key].append(fn(model))
            except Exception:
                self._model_data[key].append(None)

    def get_model_vars_dataframe(self):
        """Returns dict of lists (not pandas; avoids dependency)."""
        return self._model_data


# ------------------------------------------------------------------ #
#  Model                                                               #
# ------------------------------------------------------------------ #

class Model:
    def __init__(self, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.random = random.Random(seed)
        self._agent_type_cache: dict[type, list] = defaultdict(list)

    def _register_agent(self, agent: Agent):
        self._agent_type_cache[type(agent)].append(agent)

    @property
    def agents_by_type(self):
        return self._agent_type_cache

    def step(self):
        pass


# ------------------------------------------------------------------ #
#  Namespace shims so code can do `import mesa` style                 #
# ------------------------------------------------------------------ #

class _Space:
    MultiGrid = MultiGrid

class _Time:
    RandomActivation = RandomActivation

space = _Space()
time = _Time()