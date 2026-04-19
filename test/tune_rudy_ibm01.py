"""
Hyperparameter sweep: RUDY congestion loss weight on ibm01.

Runs multiple configurations of (cong_weight, cong_temperature, lr, num_steps)
and records final proxy cost + individual cost components.

Usage:
    uv run python test/tune_rudy_ibm01.py
"""

import sys
import os
import itertools
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from macro_place.loader import load_benchmark
from macro_place.objective import compute_proxy_cost
from submissions.gradient_placer import _gradient_overlap_finetune, RudyPrecompute


def run_experiment(bench, plc, cong_weight, cong_temperature, lr, num_steps):
    """Run one gradient optimization and return final costs."""
    placement = bench.macro_positions.clone().float()
    sizes = bench.macro_sizes.float()
    movable = bench.get_movable_mask() & bench.get_hard_macro_mask()

    result = _gradient_overlap_finetune(
        placement, bench, sizes, movable,
        lr=lr,
        num_steps=num_steps,
        cong_weight=cong_weight,
        cong_temperature=cong_temperature,
        log_interval=num_steps,  # only log start and end
    )

    costs = compute_proxy_cost(result, bench, plc)
    return costs


def main():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    netlist = os.path.join(base, "external/MacroPlacement/Testcases/ICCAD04/ibm01/netlist.pb.txt")
    plc_file = os.path.join(base, "external/MacroPlacement/Testcases/ICCAD04/ibm01/initial.plc")

    print("Loading ibm01...")
    bench, plc = load_benchmark(netlist, plc_file, name="ibm01")

    # Baseline: overlap-only
    print("\n" + "="*70)
    print("BASELINE: overlap-only (cong_weight=0)")
    print("="*70)
    base_costs = run_experiment(bench, plc, 0.0, 20.0, 0.01, 3000)
    print(f"\nBaseline result:")
    print(f"  proxy={base_costs['proxy_cost']:.6f}  wl={base_costs['wirelength_cost']:.6f}  "
          f"den={base_costs['density_cost']:.6f}  cong={base_costs['congestion_cost']:.6f}  "
          f"overlaps={base_costs['overlap_count']}")

    # Sweep: vary cong_weight with fixed temperature
    print("\n" + "="*70)
    print("SWEEP: cong_weight x temperature (lr=0.01, steps=3000)")
    print("="*70)
    print(f"{'cong_w':>8} {'temp':>6} {'proxy':>8} {'wl':>8} {'den':>8} {'cong':>8} {'overlaps':>8}")
    print("-"*70)

    configs = [
        # (cong_weight, temperature)
        (0.005, 20.0),
        (0.01,  20.0),
        (0.05,  20.0),
        (0.1,   20.0),
        (0.5,   20.0),
        (1.0,   20.0),
        (0.05,  5.0),
        (0.05,  50.0),
        (0.1,   5.0),
        (0.1,   50.0),
    ]

    results = []
    for cong_weight, temp in configs:
        print(f"\n  Running cong_weight={cong_weight}, temp={temp}...")
        costs = run_experiment(bench, plc, cong_weight, temp, 0.01, 3000)
        results.append((cong_weight, temp, costs))
        print(f"  {cong_weight:>8.3f} {temp:>6.1f} "
              f"{costs['proxy_cost']:>8.5f} {costs['wirelength_cost']:>8.5f} "
              f"{costs['density_cost']:>8.5f} {costs['congestion_cost']:>8.5f} "
              f"{costs['overlap_count']:>8}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY (sorted by congestion_cost)")
    print("="*70)
    print(f"{'cong_w':>8} {'temp':>6} {'proxy':>8} {'wl':>8} {'den':>8} {'cong':>8} {'overlaps':>8}")
    print("-"*70)

    # Include baseline
    results.insert(0, (0.0, 20.0, base_costs))
    results_sorted = sorted(results, key=lambda x: x[2]['congestion_cost'])

    for cong_weight, temp, costs in results_sorted:
        marker = " ← best cong" if costs == results_sorted[0][2] else ""
        print(f"  {cong_weight:>8.3f} {temp:>6.1f} "
              f"{costs['proxy_cost']:>8.5f} {costs['wirelength_cost']:>8.5f} "
              f"{costs['density_cost']:>8.5f} {costs['congestion_cost']:>8.5f} "
              f"{costs['overlap_count']:>8}{marker}")

    # Fine sweep around best
    best_cw, best_temp, best_c = results_sorted[0]
    print(f"\nBest config: cong_weight={best_cw}, temp={best_temp}, "
          f"proxy={best_c['proxy_cost']:.6f}")

    print(f"\nBaseline: proxy={base_costs['proxy_cost']:.6f}, "
          f"cong={base_costs['congestion_cost']:.6f}")
    if base_costs['congestion_cost'] > 0:
        delta_cong = (best_c['congestion_cost'] - base_costs['congestion_cost']) / base_costs['congestion_cost'] * 100
        print(f"Congestion change: {delta_cong:+.1f}%")


if __name__ == "__main__":
    main()
