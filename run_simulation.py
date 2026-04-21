"""
run_simulation.py — Entry point for the Drone CNP Delivery Simulation.

Usage:
    python run_simulation.py                        # headless, 200 steps
    python run_simulation.py --vis                  # live Matplotlib visualization
    python run_simulation.py --steps 500 --seed 42  # custom params

Options:
    --vis            Enable live Matplotlib visualization
    --steps  N       Number of simulation steps          (default: 200)
    --drones N       Number of drone agents              (default: 12)
    --servers N      Number of server stations           (default: 4)
    --chargers N     Number of charging stations         (default: 4)
    --rate   F       Request generation rate per server  (default: 0.02)
    --range  N       Drone communication range           (default: 10)
    --seed   N       Random seed for reproducibility
"""

import argparse
import time

from model import DroneDeliveryModel
from utils.helpers import summarise_run


def parse_args():
    p = argparse.ArgumentParser(description="Drone CNP Delivery Simulation (Mesa 3.0)")
    p.add_argument("--vis",      action="store_true")
    p.add_argument("--steps",    type=int,   default=500)
    p.add_argument("--drones",   type=int,   default=12)
    p.add_argument("--servers",  type=int,   default=4)
    p.add_argument("--chargers", type=int,   default=4)
    p.add_argument("--rate",     type=float, default=0.2)
    p.add_argument("--range",    type=int,   default=10)
    p.add_argument("--seed",     type=int,   default=None)
    return p.parse_args()


def run_headless(model, steps: int):
    print(f"\n{'='*62}")
    print(f"  Drone CNP Delivery Simulation (Mesa 3.0) — {steps} steps")
    print(f"{'='*62}")
    print(f"  Drones: {model.num_drones}  |  Servers: {model.num_servers}  "
          f"|  Chargers: {model.num_charging_stations}")
    print(f"  Grid: {model.width}×{model.height}  |  Comm range: {model.comm_range}")
    print(f"{'='*62}\n")

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

    print(f"\n{'='*62}")
    print("  Final Summary")
    print(f"{'='*62}")
    for k, v in summary.items():
        if v is None:
            continue
        print(f"  {k:<30} {round(v, 4) if isinstance(v, float) else v}")
    print(f"\n  Wall-clock time: {elapsed:.2f}s")
    print(f"{'='*62}\n")


def run_visualized(model, steps: int):
    try:
        from visualization.visualizer import SimulationVisualizer
    except ImportError as e:
        print(f"Visualization unavailable ({e}). Running headless.")
        run_headless(model, steps)
        return

    viz = SimulationVisualizer(model)
    print(f"\nRunning {steps} steps with visualization. Close window or Ctrl+C to stop.\n")

    try:
        completed_steps = 0
        while completed_steps < steps:
            if not viz.paused:
                model.step()
                completed_steps += 1
            viz.update()
            time.sleep(0.01)   # controlling the speed
    except KeyboardInterrupt:
        print("\nInterrupted.")

    import matplotlib.pyplot as plt
    plt.ioff()
    plt.show()


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