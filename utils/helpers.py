"""
Utility functions shared across the simulation.
"""

import numpy as np


def manhattan(a: tuple, b: tuple) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def moving_average(data: list, window: int) -> list:
    if len(data) < window:
        return data
    kernel = np.ones(window) / window
    return list(np.convolve(data, kernel, mode="valid"))


def summarise_run(model) -> dict:
    """Return a dict of end-of-run statistics."""
    drones = model.drone_agents
    return {
        "steps": model.step_count,
        "completed": model.completed_deliveries,
        "failed": model.failed_deliveries,
        "total_requests": model.total_requests,
        "completion_rate": (
            model.completed_deliveries / max(model.total_requests, 1)
        ),
        "battery_depletions": model.battery_depletions,
        "avg_battery": np.mean([d.battery for d in drones]),
        "avg_alpha": np.mean([d.alpha for d in drones]),
        "avg_beta": np.mean([d.beta for d in drones]),
        "avg_gamma": np.mean([d.gamma for d in drones]),
        "avg_safety_reserve": np.mean([d.safety_reserve for d in drones]),
        "avg_delivery_time": (
            np.mean(model.delivery_times) if model.delivery_times else None
        ),
    }