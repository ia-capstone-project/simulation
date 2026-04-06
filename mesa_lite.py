"""
mesa_lite.py — Mesa 3.0-compatible drop-in shim.

Mirrors the Mesa 3.0 API exactly:
  - Agent(model)           → unique_id auto-assigned; auto-registers with model
  - Model.agents           → AgentSet (iterable, .shuffle_do, .select)
  - Model.agents_by_type   → dict[type, AgentSet]
  - MultiGrid              → unchanged from 2.x
  - DataCollector          → unchanged from 2.x
  - NO schedule / RandomActivation — activation via agents.shuffle_do("step")

To swap in real Mesa 3.0, install it and the import guards handle the rest.
"""

import random
import numpy as np
from collections import defaultdict


# ------------------------------------------------------------------ #
#  AgentSet                                                            #
# ------------------------------------------------------------------ #

class AgentSet:
    """Mirrors mesa.AgentSet from Mesa 3.0."""

    def __init__(self, agents=None, random_instance=None):
        self._agents: list = list(agents or [])
        self._random = random_instance or random

    def add(self, agent):
        self._agents.append(agent)

    def remove(self, agent):
        try:
            self._agents.remove(agent)
        except ValueError:
            pass

    def shuffle_do(self, method_name: str, *args, **kwargs):
        """Mesa 3.0 primary activation pattern."""
        agents = list(self._agents)
        self._random.shuffle(agents)
        for agent in agents:
            getattr(agent, method_name)(*args, **kwargs)

    def do(self, method_name: str, *args, **kwargs):
        for agent in list(self._agents):
            getattr(agent, method_name)(*args, **kwargs)

    def select(self, filter_func=None, agent_type=None):
        result = list(self._agents)
        if agent_type is not None:
            result = [a for a in result if isinstance(a, agent_type)]
        if filter_func is not None:
            result = [a for a in result if filter_func(a)]
        return AgentSet(result, self._random)

    def __iter__(self):
        return iter(list(self._agents))

    def __len__(self):
        return len(self._agents)

    def __contains__(self, agent):
        return agent in self._agents


# ------------------------------------------------------------------ #
#  Agent                                                               #
# ------------------------------------------------------------------ #

class Agent:
    """
    Mesa 3.0 Agent.
    unique_id is auto-assigned. Agent auto-registers with model on init.
    No unique_id argument — Mesa 3.0 removed it from the constructor.
    """
    _id_counter = 0

    def __init__(self, model: "Model"):
        Agent._id_counter += 1
        self.unique_id: int = Agent._id_counter
        self.model = model
        self.pos: tuple = None
        model._register_agent(self)   # auto-register, Mesa 3.0 style

    def step(self):
        pass

    def remove(self):
        self.model._deregister_agent(self)


# ------------------------------------------------------------------ #
#  MultiGrid                                                           #
# ------------------------------------------------------------------ #

class MultiGrid:
    def __init__(self, width: int, height: int, torus: bool = False):
        self.width = width
        self.height = height
        self.torus = torus
        self._grid: dict[tuple, list] = defaultdict(list)

    def place_agent(self, agent, pos: tuple):
        agent.pos = pos
        if agent not in self._grid[pos]:
            self._grid[pos].append(agent)

    def move_agent(self, agent, new_pos: tuple):
        if agent.pos is not None and agent in self._grid.get(agent.pos, []):
            self._grid[agent.pos].remove(agent)
        agent.pos = new_pos
        if agent not in self._grid[new_pos]:
            self._grid[new_pos].append(agent)

    def remove_agent(self, agent):
        if agent.pos is not None:
            cell = self._grid.get(agent.pos, [])
            if agent in cell:
                cell.remove(agent)

    def get_cell_list_contents(self, positions) -> list:
        result = []
        for pos in positions:
            result.extend(self._grid.get(pos, []))
        return result

    def is_in_bounds(self, pos: tuple) -> bool:
        return 0 <= pos[0] < self.width and 0 <= pos[1] < self.height


# ------------------------------------------------------------------ #
#  DataCollector                                                       #
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
        return self._model_data


# ------------------------------------------------------------------ #
#  Model                                                               #
# ------------------------------------------------------------------ #

class Model:
    """
    Mesa 3.0 Model.

    Mesa 3.0 changes vs 2.x:
    - self.agents           → AgentSet of ALL agents (no self.schedule)
    - self.agents_by_type   → {AgentClass: AgentSet}
    - Activation:           self.agents.shuffle_do("step")
    - Agent registration:   automatic on Agent.__init__
    - self.running removed
    """

    def __init__(self, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.random = random.Random(seed)
        self._all_agents = AgentSet(random_instance=self.random)
        self._agents_by_type: dict = defaultdict(
            lambda: AgentSet(random_instance=self.random)
        )

    @property
    def agents(self) -> AgentSet:
        return self._all_agents

    @property
    def agents_by_type(self) -> dict:
        return self._agents_by_type

    def _register_agent(self, agent: Agent):
        """Called by Agent.__init__ automatically."""
        self._all_agents.add(agent)
        self._agents_by_type[type(agent)].add(agent)

    def _deregister_agent(self, agent: Agent):
        self._all_agents.remove(agent)
        self._agents_by_type[type(agent)].remove(agent)

    def step(self):
        pass