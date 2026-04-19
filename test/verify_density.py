"""
Verify that _density_cost() in gradient_placer.py matches plc_client_os.get_density_cost()
exactly on the initial positions of each benchmark.

Usage:
    uv run python test/verify_density.py
    uv run python test/verify_density.py -b ibm01
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import _set_placement
from submissions.gradient_placer import _density_cost

TESTCASE_ROOT = "external/MacroPlacement/Testcases/ICCAD04"
IBM_BENCHMARKS = [
    "ibm01", "ibm02", "ibm03", "ibm04", "ibm06", "ibm07", "ibm08",
    "ibm09", "ibm10", "ibm11", "ibm12", "ibm13", "ibm14", "ibm15",
    "ibm16", "ibm17", "ibm18",
]


def verify_one(name: str) -> bool:
    benchmark_dir = f"{TESTCASE_ROOT}/{name}"
    if not Path(benchmark_dir).exists():
        print(f"  [{name}] SKIP — {benchmark_dir} not found")
        return True

    benchmark, plc = load_benchmark_from_dir(benchmark_dir)
    placement = benchmark.macro_positions.clone().float()

    # Ground truth: evaluator on initial positions
    _set_placement(plc, placement, benchmark)
    gt = plc.get_density_cost()

    # Our implementation: pass all macros (hard + soft)
    all_pos   = benchmark.macro_positions.float()
    all_sizes = benchmark.macro_sizes.float()
    ours = _density_cost(
        all_pos, all_sizes,
        benchmark.canvas_width, benchmark.canvas_height,
        benchmark.grid_rows, benchmark.grid_cols,
    ).item()

    rel_err = abs(ours - gt) / (abs(gt) + 1e-12)
    ok = rel_err < 1e-5
    status = "OK      " if ok else "MISMATCH"
    print(f"  [{name:8s}] {status}  gt={gt:.8f}  ours={ours:.8f}  rel_err={rel_err:.2e}")
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", "-b", default=None)
    args = parser.parse_args()

    names = [args.benchmark] if args.benchmark else IBM_BENCHMARKS
    results = [verify_one(n) for n in names]
    passed = sum(results)
    print(f"\n{passed}/{len(results)} benchmarks match.")
    if not all(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
