"""
run_simulation.py — Entry point for the Drone CNP Delivery Simulation.

Usage:
    python run_simulation.py                   # headless, 200 steps, prints summary
    python run_simulation.py --vis             # with live Matplotlib visualization
    python run_simulation.py --steps 500       # custom step count
    python run_simulation.py --vis --steps 300 --drones 15 --seed 42

Options:
    --vis            Enable live visualization
    --steps N        Number of simulation steps (default: 200)
    --drones N       Number of drone agents (default: 12)
    --servers N      Number of server stations (default: 4)
    --chargers N     Number of charging stations (default: 6)
    --rate F         Request generation rate (default: 0.08)
    --range N        Communication range (default: 10)
    --seed N         Random seed for reproducibility
"""

import argparse
import sys
import time

from model import DroneDeliveryModel
from utils.helpers import summarise_run


def parse_args():
    p = argparse.ArgumentParser(description="Drone CNP Delivery Simulation")
    p.add_argument("--vis",     action="store_true", help="Live visualization")
    p.add_argument("--steps",   type=int,   default=200)
    p.add_argument("--drones",  type=int,   default=12)
    p.add_argument("--servers", type=int,   default=4)
    p.add_argument("--chargers",type=int,   default=6)
    p.add_argument("--rate",    type=float, default=0.08)
    p.add_argument("--range",   type=int,   default=10)
    p.add_argument("--seed",    type=int,   default=None)
    return p.parse_args()


def run_headless(model, steps: int):
    print(f"\n{'='*60}")
    print(f"  Drone CNP Delivery Simulation — {steps} steps")
    print(f"{'='*60}")
    print(f"  Drones: {model.num_drones}  |  Servers: {model.num_servers}  "
          f"|  Chargers: {model.num_charging_stations}")
    print(f"  Grid: {model.width}×{model.height}  |  Comm range: {model.comm_range}")
    print(f"{'='*60}\n")

    start = time.time()
    for step in range(steps):
        model.step()
        if step % 50 == 0 or step == steps - 1:
            print(
                f"  Step {step:4d} | "
                f"Completed: {model.completed_deliveries:4d} | "
                f"Failed: {model.failed_deliveries:3d} | "
                f"Active: {len(model.active_requests):3d} | "
                f"Pending: {len(model.pending_requests):3d} | "
                f"Depletions: {model.battery_depletions}"
            )

    elapsed = time.time() - start
    summary = summarise_run(model)

    print(f"\n{'='*60}")
    print("  Final Summary")
    print(f"{'='*60}")
    for k, v in summary.items():
        if v is None:
            continue
        if isinstance(v, float):
            print(f"  {k:<28} {v:.4f}")
        else:
            print(f"  {k:<28} {v}")
    print(f"\n  Wall-clock time: {elapsed:.2f}s")
    print(f"{'='*60}\n")

    return model


def run_visualized(model, steps: int):
    """Run with live Matplotlib window."""
    try:
        from visualization.visualizer import SimulationVisualizer
    except ImportError as e:
        print(f"Visualization unavailable: {e}\nFalling back to headless.")
        return run_headless(model, steps)

    viz = SimulationVisualizer(model)
    print(f"\nRunning {steps} steps with visualization...")
    print("Close the window or press Ctrl+C to stop early.\n")

    try:
        for step in range(steps):
            model.step()
            if step % 2 == 0:     # Update display every 2 steps for speed
                viz.update()
            time.sleep(0.05)      # ~20 fps
    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    # Keep window open
    print("\nSimulation complete. Close the window to exit.")
    import matplotlib.pyplot as plt
    plt.ioff()
    plt.show()

    return model


def main():
    args = parse_args()

    model = DroneDeliveryModel(
        num_drones=args.drones,
        num_servers=args.servers,
        num_charging_stations=args.chargers,
        request_rate=args.rate,
        comm_range=args.range,
        seed=args.seed,
    )

    if args.vis:
        run_visualized(model, args.steps)
    else:
        run_headless(model, args.steps)


if __name__ == "__main__":
    main()