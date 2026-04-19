"""
Initial Positions Visualizer + Congestion Perturbation Study

Returns macros at their initial positions (no movement).
Runs 100 perturbations of movable macros around initial positions
and prints congestion cost for each to study local sensitivity.

Usage:
    uv run evaluate submissions/initial_positions.py -b ibm01
    uv run evaluate submissions/initial_positions.py --all
"""

import torch
import numpy as np
from macro_place.benchmark import Benchmark
from macro_place.loader import load_benchmark_from_dir, load_benchmark
from macro_place.objective import compute_proxy_cost
import os
import glob


def _find_and_load_plc(benchmark: Benchmark):
    """Reconstruct plc from benchmark name by searching known directories."""
    name = benchmark.name
    
    # Search paths
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [
        os.path.join(base, "external/MacroPlacement/Testcases/ICCAD04", name),
    ]
    # NG45 designs
    ng45_base = os.path.join(base, "external/MacroPlacement/Testcases")
    for design_dir in glob.glob(os.path.join(ng45_base, "*/NanGate45")):
        parent = os.path.basename(os.path.dirname(design_dir))
        if parent.startswith(name) or name.endswith("_ng45"):
            candidates.append(design_dir)
    
    for d in candidates:
        netlist = os.path.join(d, "netlist.pb.txt")
        if os.path.exists(netlist):
            plc_file = os.path.join(d, "initial.plc")
            if not os.path.exists(plc_file):
                plc_file = None
            _, plc = load_benchmark(netlist, plc_file, name=name)
            return plc
    return None


class InitialPositions:
    def place(self, benchmark: Benchmark) -> torch.Tensor:
        placement = benchmark.macro_positions.clone().float()
        movable = benchmark.get_movable_mask() & benchmark.get_hard_macro_mask()
        sizes = benchmark.macro_sizes.float()
        movable_indices = movable.nonzero(as_tuple=True)[0]

        plc = _find_and_load_plc(benchmark)
        if plc is None:
            print(f"  [WARN] Could not find plc for {benchmark.name}, skipping perturbation study")
            return benchmark.macro_positions.clone()

        # Baseline congestion cost
        base_costs = compute_proxy_cost(placement, benchmark, plc)
        base_cong = base_costs["congestion_cost"]
        base_wl = base_costs["wirelength_cost"]
        base_den = base_costs["density_cost"]
        print(f"\n  === Congestion perturbation study (100 trials) ===")
        print(f"  Baseline: wl={base_wl:.6f}  den={base_den:.6f}  cong={base_cong:.6f}")
        print(f"  Movable hard macros: {len(movable_indices)}")
        print(f"  {'Trial':>5}  {'Scale':>8}  {'Congestion':>10}  {'Delta':>10}  {'WL':>10}  {'Density':>10}")
        print(f"  {'-'*5}  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")

        perturbation_scales = [0.01, 0.1, 10.0, 25.0, 50.0]
        results_by_scale = {s: [] for s in perturbation_scales}

        trial = 0
        for scale in perturbation_scales:
            for _ in range(20):
                trial += 1
                p = placement.clone()
                noise = torch.randn(len(movable_indices), 2) * scale
                p[movable_indices] += noise

                # Clamp to canvas
                half_w = sizes[movable_indices, 0] / 2.0
                half_h = sizes[movable_indices, 1] / 2.0
                p[movable_indices, 0] = p[movable_indices, 0].clamp(half_w, benchmark.canvas_width - half_w)
                p[movable_indices, 1] = p[movable_indices, 1].clamp(half_h, benchmark.canvas_height - half_h)

                costs = compute_proxy_cost(p, benchmark, plc)
                cong = costs["congestion_cost"]
                delta = cong - base_cong
                results_by_scale[scale].append(cong)
                print(f"  {trial:>5}  {scale:>6.4f}µm  {cong:>10.6f}  {delta:>+10.6f}  {costs['wirelength_cost']:>10.6f}  {costs['density_cost']:>10.6f}")

        print(f"\n  === Summary by perturbation scale ===")
        print(f"  {'Scale':>10}  {'Mean Cong':>10}  {'Std':>10}  {'Min':>10}  {'Max':>10}  {'Mean Delta':>10}")
        for scale in perturbation_scales:
            vals = np.array(results_by_scale[scale])
            print(f"  {scale:>8.1f}µm  {vals.mean():>10.6f}  {vals.std():>10.6f}  {vals.min():>10.6f}  {vals.max():>10.6f}  {vals.mean()-base_cong:>+10.6f}")

        return benchmark.macro_positions.clone()
