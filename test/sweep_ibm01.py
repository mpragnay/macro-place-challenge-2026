"""
Hyperparameter sweep for GradientPlacer on ibm01 — Round 3.
Anchored on lr=0.01 finding from Round 2.

Usage:
    uv run python test/sweep_ibm01.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost
from submissions.gradient_placer import GradientPlacer

TESTCASE = "external/MacroPlacement/Testcases/ICCAD04/ibm01"

configs = [
    # (lr,    w_overlap, w_density, num_dense, neighbor_ratio, refresh_every, num_steps, label)
    (0.010, 10.0, 10.0, 3, 0.3, 200, 4000, "lr=0.01 steps=4k [ref]"),
    (0.005, 10.0, 10.0, 3, 0.3, 200, 8000, "lr=0.005 steps=8k"),
    (0.001, 10.0, 10.0, 3, 0.3, 200,10000, "lr=0.001 steps=10k"),
    (0.010, 10.0,  0.0, 3, 0.3, 200, 4000, "lr=0.01 wd=0 (overlap only)"),
    (0.010, 10.0, 50.0, 3, 0.3, 200, 4000, "lr=0.01 wd=50"),
    (0.010, 10.0, 10.0, 1, 0.3, 200, 4000, "lr=0.01 nd=1"),
    (0.010, 10.0, 10.0, 3, 0.2, 200, 4000, "lr=0.01 nr=0.2 (stricter)"),
    (0.010,  5.0, 10.0, 3, 0.3, 200, 4000, "lr=0.01 wo=5"),
]

print(f"{'label':<30}  {'proxy':>7}  {'wl':>7}  {'den':>7}  {'cong':>7}  {'overlaps':>8}")
print("-" * 80)

for lr, wo, wd, nd, nr, re, steps, label in configs:
    benchmark, plc = load_benchmark_from_dir(TESTCASE)

    placer = GradientPlacer(
        lr=lr, num_steps=steps,
        w_overlap=wo, w_density=wd,
        num_dense=nd, neighbor_ratio=nr,
        refresh_every=re, log_every=steps,
        verbose=True,
    )
    placement = placer.place(benchmark)
    c = compute_proxy_cost(placement, benchmark, plc)

    print(
        f"{label:<30}  {c['proxy_cost']:7.4f}  {c['wirelength_cost']:7.4f}"
        f"  {c['density_cost']:7.4f}  {c['congestion_cost']:7.4f}"
        f"  {c['overlap_count']:8d}"
    )
    sys.stdout.flush()
