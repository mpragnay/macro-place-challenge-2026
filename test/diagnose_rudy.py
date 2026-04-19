"""
Diagnostic: compare RUDY congestion proxy with ground-truth evaluator on ibm01.

Measures:
  1. Per-cell RUDY values vs actual congestion map from plc evaluator
  2. Scale ratio: RUDY abu(5%) vs actual congestion cost
  3. Correlation between RUDY and actual per-cell congestion
"""

import sys
import os
import math
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from macro_place.loader import load_benchmark
from macro_place.objective import compute_proxy_cost, _set_placement
from macro_place.benchmark import Benchmark


def compute_rudy_map(
    positions: torch.Tensor,
    benchmark: Benchmark,
    *,
    eps: float = 0.1,
) -> tuple:
    """
    Compute RUDY H and V utilization maps.

    Returns:
        H_util [G, C], V_util [G, C], combined [2*G*C]
    """
    device = positions.device
    dtype = positions.dtype

    G = benchmark.grid_rows
    C = benchmark.grid_cols
    W = benchmark.canvas_width
    H = benchmark.canvas_height
    cell_w = W / C
    cell_h = H / G

    # Routing capacity per cell edge (tracks per cell)
    grid_h_routes = cell_h * benchmark.hroutes_per_micron
    grid_v_routes = cell_w * benchmark.vroutes_per_micron

    # Combined node positions: macros + ports
    if benchmark.port_positions.shape[0] > 0:
        port_pos = benchmark.port_positions.to(device=device, dtype=dtype)
        all_pos = torch.cat([positions, port_pos], dim=0)
    else:
        all_pos = positions

    # Grid cell boundaries
    col_edges = torch.arange(C + 1, dtype=dtype, device=device) * cell_w
    row_edges = torch.arange(G + 1, dtype=dtype, device=device) * cell_h
    cell_x_lo = col_edges[:-1]   # [C]
    cell_x_hi = col_edges[1:]    # [C]
    cell_y_lo = row_edges[:-1]   # [G]
    cell_y_hi = row_edges[1:]    # [G]

    H_util = torch.zeros(G, C, dtype=dtype, device=device)
    V_util = torch.zeros(G, C, dtype=dtype, device=device)

    for net_nodes in benchmark.net_nodes:
        node_pos = all_pos[net_nodes]
        x_min = node_pos[:, 0].min().item()
        x_max = node_pos[:, 0].max().item()
        y_min = node_pos[:, 1].min().item()
        y_max = node_pos[:, 1].max().item()

        bbox_w = max(x_max - x_min, eps)
        bbox_h = max(y_max - y_min, eps)

        # Overlap of bbox with each cell column/row
        ol_x = torch.clamp(
            torch.minimum(cell_x_hi, torch.tensor(x_max, dtype=dtype))
            - torch.maximum(cell_x_lo, torch.tensor(x_min, dtype=dtype)),
            min=0.0,
        )  # [C]
        ol_y = torch.clamp(
            torch.minimum(cell_y_hi, torch.tensor(y_max, dtype=dtype))
            - torch.maximum(cell_y_lo, torch.tensor(y_min, dtype=dtype)),
            min=0.0,
        )  # [G]

        outer = ol_y[:, None] * ol_x[None, :]  # [G, C] overlap area
        H_util = H_util + outer / (bbox_h * grid_h_routes)
        V_util = V_util + outer / (bbox_w * grid_v_routes)

    combined = torch.cat([H_util.flatten(), V_util.flatten()])
    return H_util, V_util, combined


def abu(values, frac=0.05):
    """Average of top frac of values (including zeros)."""
    arr = sorted(values, reverse=True)
    k = max(1, math.floor(len(arr) * frac))
    return sum(arr[:k]) / k


def main():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    netlist = os.path.join(base, "external/MacroPlacement/Testcases/ICCAD04/ibm01/netlist.pb.txt")
    plc_file = os.path.join(base, "external/MacroPlacement/Testcases/ICCAD04/ibm01/initial.plc")

    print("Loading ibm01...")
    bench, plc = load_benchmark(netlist, plc_file, name="ibm01")

    positions = bench.macro_positions.clone().float()

    print(f"\nBenchmark: {bench}")
    print(f"  hroutes_per_micron = {bench.hroutes_per_micron:.3f}")
    print(f"  vroutes_per_micron = {bench.vroutes_per_micron:.3f}")
    print(f"  cell_w = {bench.canvas_width/bench.grid_cols:.4f} µm, "
          f"cell_h = {bench.canvas_height/bench.grid_rows:.4f} µm")
    print(f"  grid_h_routes = {(bench.canvas_height/bench.grid_rows)*bench.hroutes_per_micron:.2f} tracks/cell")
    print(f"  grid_v_routes = {(bench.canvas_width/bench.grid_cols)*bench.vroutes_per_micron:.2f} tracks/cell")

    # Ground-truth congestion
    print("\nComputing ground-truth costs...")
    costs = compute_proxy_cost(positions, bench, plc)
    print(f"  proxy_cost     = {costs['proxy_cost']:.6f}")
    print(f"  wirelength     = {costs['wirelength_cost']:.6f}")
    print(f"  density        = {costs['density_cost']:.6f}")
    print(f"  congestion     = {costs['congestion_cost']:.6f}  ← ground truth")
    print(f"  overlaps       = {costs['overlap_count']}")

    # Extract raw H/V congestion maps from evaluator
    plc_H = list(plc.H_routing_cong)
    plc_V = list(plc.V_routing_cong)
    plc_combined = plc_H + plc_V

    print(f"\nEvaluator congestion map stats:")
    print(f"  H map: min={min(plc_H):.4f}  max={max(plc_H):.4f}  "
          f"mean={sum(plc_H)/len(plc_H):.4f}  nonzero={sum(1 for v in plc_H if v>0)}/{len(plc_H)}")
    print(f"  V map: min={min(plc_V):.4f}  max={max(plc_V):.4f}  "
          f"mean={sum(plc_V)/len(plc_V):.4f}  nonzero={sum(1 for v in plc_V if v>0)}/{len(plc_V)}")
    print(f"  combined abu(5%) = {abu(plc_combined, 0.05):.6f}  ← should match congestion_cost")

    # RUDY map
    print("\nComputing RUDY map...")
    with torch.no_grad():
        H_util, V_util, rudy_combined = compute_rudy_map(positions, bench)

    rudy_H = H_util.numpy().flatten().tolist()
    rudy_V = V_util.numpy().flatten().tolist()
    rudy_all = rudy_combined.numpy().tolist()

    rudy_abu5 = abu(rudy_all, 0.05)
    plc_abu5 = abu(plc_combined, 0.05)

    print(f"\nRUDY map stats:")
    print(f"  H map: min={min(rudy_H):.4f}  max={max(rudy_H):.4f}  "
          f"mean={sum(rudy_H)/len(rudy_H):.4f}  nonzero={sum(1 for v in rudy_H if v>0)}/{len(rudy_H)}")
    print(f"  V map: min={min(rudy_V):.4f}  max={max(rudy_V):.4f}  "
          f"mean={sum(rudy_V)/len(rudy_V):.4f}  nonzero={sum(1 for v in rudy_V if v>0)}/{len(rudy_V)}")
    print(f"  combined abu(5%) = {rudy_abu5:.6f}")

    print(f"\nScale comparison:")
    print(f"  Ground-truth congestion cost  = {plc_abu5:.6f}")
    print(f"  RUDY abu(5%)                  = {rudy_abu5:.6f}")
    print(f"  Scale ratio (GT / RUDY)       = {plc_abu5 / rudy_abu5:.4f}x")
    print(f"  → To match GT scale, multiply RUDY by {plc_abu5/rudy_abu5:.4f}")

    # Correlation between RUDY and evaluator per-cell
    rudy_arr = np.array(rudy_all)
    plc_arr = np.array(plc_combined)
    corr = np.corrcoef(rudy_arr, plc_arr)[0, 1]
    print(f"\nPer-cell correlation (RUDY vs evaluator): {corr:.4f}")

    # Top-cell comparison
    print(f"\nTop 10 cells by evaluator congestion:")
    top_idx = sorted(range(len(plc_combined)), key=lambda i: plc_combined[i], reverse=True)[:10]
    for rank, i in enumerate(top_idx):
        tag = "H" if i < len(plc_H) else "V"
        cell = i if i < len(plc_H) else i - len(plc_H)
        r, c = cell // bench.grid_cols, cell % bench.grid_cols
        plc_val = plc_combined[i]
        rudy_val = rudy_all[i]
        print(f"  [{rank+1:2d}] {tag}[r={r:2d},c={c:2d}]  plc={plc_val:.4f}  rudy={rudy_val:.4f}  "
              f"ratio={plc_val/max(rudy_val,1e-9):.2f}x")


if __name__ == "__main__":
    main()
