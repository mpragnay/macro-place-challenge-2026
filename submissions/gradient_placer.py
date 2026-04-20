"""
Gradient Placer - SA Congestion + Gradient Overlap Fine-tuning

Phase 1: Simulated Annealing targeting congestion cost using the
         ground-truth plc evaluator with tiny perturbations (~0.01µm).
Phase 2: Gradient descent on differentiable overlap loss to resolve
         any remaining macro overlaps.

Logs proxy_cost and individual cost terms throughout.

Usage:
    uv run evaluate submissions/gradient_placer.py -b ibm01
    uv run evaluate submissions/gradient_placer.py --all
"""

import torch
import math
import os
import glob
import numpy as np
from macro_place.benchmark import Benchmark
from macro_place.loader import load_benchmark, load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost


# ── RUDY Congestion Proxy ─────────────────────────────────────────────────────

class RudyPrecompute:
    """
    Pre-splits nets into fixed (no movable hard macros) vs variable (has at
    least one movable hard macro), and pre-computes the fixed contribution to
    the RUDY map.  Call once; then use in the training loop cheaply.

    The variable-net contribution is fully vectorized via scatter_reduce +
    batch matmul, so each forward+backward pass is O(N_var × (G+C)) tensor
    ops with no Python loop.
    """

    def __init__(
        self,
        benchmark: Benchmark,
        movable_mask: torch.Tensor,
        *,
        pin_weight: bool = False,
    ):
        """
        Args:
            pin_weight: if True, scale each net's demand by (pin_count - 1),
                        giving multi-pin nets proportionally more routing weight.
        """
        G = benchmark.grid_rows
        C = benchmark.grid_cols
        W = benchmark.canvas_width
        H_canvas = benchmark.canvas_height
        self.G = G
        self.C = C
        self.cell_w = W / C
        self.cell_h = H_canvas / G
        self.grid_h_routes = self.cell_h * benchmark.hroutes_per_micron
        self.grid_v_routes = self.cell_w * benchmark.vroutes_per_micron
        self.pin_weight = pin_weight

        port_pos = benchmark.port_positions.float()
        self.has_ports = port_pos.shape[0] > 0
        self.port_positions = port_pos
        self.num_macros = benchmark.num_macros

        movable_idx_set = set(movable_mask.nonzero(as_tuple=True)[0].tolist())

        # Grid edges
        self.cell_x_lo = torch.arange(C, dtype=torch.float32) * self.cell_w
        self.cell_x_hi = self.cell_x_lo + self.cell_w
        self.cell_y_lo = torch.arange(G, dtype=torch.float32) * self.cell_h
        self.cell_y_hi = self.cell_y_lo + self.cell_h

        # Split into fixed / variable nets + record pin counts
        fixed_nets_nodes = []
        variable_nets_nodes = []
        var_pin_counts = []
        for net_nodes in benchmark.net_nodes:
            has_movable = any(n.item() in movable_idx_set for n in net_nodes)
            if has_movable:
                variable_nets_nodes.append(net_nodes)
                var_pin_counts.append(len(net_nodes))
            else:
                fixed_nets_nodes.append(net_nodes)

        # ── Pre-compute fixed map (no autograd needed) ──
        if self.has_ports:
            fixed_all_pos = torch.cat(
                [benchmark.macro_positions.float(), port_pos], dim=0
            )
        else:
            fixed_all_pos = benchmark.macro_positions.float()

        H_fixed, V_fixed = self._build_rudy_map(
            fixed_all_pos, fixed_nets_nodes, requires_grad=False
        )
        self.H_fixed = H_fixed  # [G, C]
        self.V_fixed = V_fixed  # [G, C]

        # ── Build flat index arrays for variable nets ──
        N_var = len(variable_nets_nodes)
        self.N_var = N_var
        flat_indices = []
        net_ids_flat = []
        for i, nodes in enumerate(variable_nets_nodes):
            flat_indices.append(nodes)
            net_ids_flat.append(torch.full((len(nodes),), i, dtype=torch.long))

        if N_var > 0:
            self.flat_node_indices = torch.cat(flat_indices)
            self.net_ids_flat = torch.cat(net_ids_flat)
            # (pin_count - 1) weights, clamped to at least 1
            pc = torch.tensor(var_pin_counts, dtype=torch.float32)
            self.pin_counts = (pc - 1.0).clamp(min=1.0)  # [N_var]
        else:
            self.flat_node_indices = torch.zeros(0, dtype=torch.long)
            self.net_ids_flat = torch.zeros(0, dtype=torch.long)
            self.pin_counts = torch.zeros(0, dtype=torch.float32)

    def _build_rudy_map(self, all_pos, nets_list, *, requires_grad=False):
        """Compute H_util and V_util via Python loop (used for fixed nets once)."""
        G, C = self.G, self.C
        # Clamp bbox to cell size — caps density at 1/cell_area, avoids explosion
        min_w = self.cell_w
        min_h = self.cell_h
        H_util = torch.zeros(G, C)
        V_util = torch.zeros(G, C)
        for net_nodes in nets_list:
            node_pos = all_pos[net_nodes]
            x_min = node_pos[:, 0].min().item()
            x_max = node_pos[:, 0].max().item()
            y_min = node_pos[:, 1].min().item()
            y_max = node_pos[:, 1].max().item()
            bbox_w = max(x_max - x_min, min_w)
            bbox_h = max(y_max - y_min, min_h)
            w = max(len(net_nodes) - 1, 1) if self.pin_weight else 1.0
            ol_x = torch.clamp(
                torch.minimum(self.cell_x_hi, torch.tensor(x_max))
                - torch.maximum(self.cell_x_lo, torch.tensor(x_min)),
                min=0.0,
            )
            ol_y = torch.clamp(
                torch.minimum(self.cell_y_hi, torch.tensor(y_max))
                - torch.maximum(self.cell_y_lo, torch.tensor(y_min)),
                min=0.0,
            )
            outer = ol_y[:, None] * ol_x[None, :]
            H_util = H_util + w * outer / (bbox_h * self.grid_h_routes)
            V_util = V_util + w * outer / (bbox_w * self.grid_v_routes)
        return H_util, V_util

    def compute(
        self,
        positions: torch.Tensor,
        *,
        temperature: float = 20.0,
        top_frac: float = 0.05,
    ) -> torch.Tensor:
        """
        Differentiable RUDY congestion proxy (smooth abu top-5%).

        Normalization (option 3+4 from user analysis):
          - bbox clamped to cell_w/cell_h floor → caps density at 1/cell_area
          - H/V separated with capacity normalization
          - optional (pin_count - 1) weighting

        Variable nets: scatter_reduce (batched min/max) + batch matmul.
        """
        device = positions.device
        dtype = positions.dtype
        N_var = self.N_var
        G, C = self.G, self.C

        H_util = self.H_fixed.to(device=device, dtype=dtype)
        V_util = self.V_fixed.to(device=device, dtype=dtype)

        if N_var > 0:
            if self.has_ports:
                port_pos = self.port_positions.to(device=device, dtype=dtype)
                all_pos = torch.cat([positions, port_pos], dim=0)
            else:
                all_pos = positions

            flat_idx = self.flat_node_indices.to(device)
            net_ids = self.net_ids_flat.to(device)

            flat_pos = all_pos[flat_idx]  # [total_nodes, 2]
            flat_x = flat_pos[:, 0]
            flat_y = flat_pos[:, 1]

            NEG_INF = torch.full((N_var,), -1e9, dtype=dtype, device=device)
            POS_INF = torch.full((N_var,), +1e9, dtype=dtype, device=device)

            x_max = NEG_INF.scatter_reduce(0, net_ids, flat_x, reduce='amax', include_self=True)
            x_min = POS_INF.scatter_reduce(0, net_ids, flat_x, reduce='amin', include_self=True)
            y_max = NEG_INF.scatter_reduce(0, net_ids, flat_y, reduce='amax', include_self=True)
            y_min = POS_INF.scatter_reduce(0, net_ids, flat_y, reduce='amin', include_self=True)

            # Clamp to cell size (option 1/3: cap density at 1/cell_area)
            bbox_w = (x_max - x_min).clamp(min=self.cell_w)  # [N_var]
            bbox_h = (y_max - y_min).clamp(min=self.cell_h)  # [N_var]

            cell_x_lo = self.cell_x_lo.to(device=device, dtype=dtype)
            cell_x_hi = self.cell_x_hi.to(device=device, dtype=dtype)
            cell_y_lo = self.cell_y_lo.to(device=device, dtype=dtype)
            cell_y_hi = self.cell_y_hi.to(device=device, dtype=dtype)

            ol_x = torch.clamp(
                torch.minimum(cell_x_hi[:, None], x_max[None, :])
                - torch.maximum(cell_x_lo[:, None], x_min[None, :]),
                min=0.0,
            )  # [C, N_var]
            ol_y = torch.clamp(
                torch.minimum(cell_y_hi[:, None], y_max[None, :])
                - torch.maximum(cell_y_lo[:, None], y_min[None, :]),
                min=0.0,
            )  # [G, N_var]

            # Pin-count weights (option 2)
            if self.pin_weight:
                pc = self.pin_counts.to(device=device, dtype=dtype)  # [N_var]
            else:
                pc = torch.ones(N_var, dtype=dtype, device=device)

            h_coeff = pc / (bbox_h * self.grid_h_routes)  # [N_var]
            v_coeff = pc / (bbox_w * self.grid_v_routes)  # [N_var]

            H_util = H_util + ol_y @ (ol_x * h_coeff[None, :]).T   # [G,C]
            V_util = V_util + ol_y @ (ol_x * v_coeff[None, :]).T   # [G,C]

        combined = torch.cat([H_util.flatten(), V_util.flatten()])
        k = max(1, int(combined.numel() * top_frac))
        top_vals, _ = torch.topk(combined, k)
        loss = torch.logsumexp(temperature * top_vals, dim=0) / temperature
        return loss


# ── Gaussian RUDY Congestion Proxy ───────────────────────────────────────────

class GaussianRudyPrecompute:
    """
    RUDY congestion proxy with per-macro Gaussian (bell) spreading.

    Unlike hard-bbox RUDY where only bbox-boundary macros get gradient,
    each macro spreads routing demand via a Gaussian kernel over grid cells.
    Every macro gets a non-zero gradient, including those deep inside a net bbox.

    Per-net demand at cell (r,c):
        demand += (1/k) * sum_{i in net} [ gauss(cx - xi, σx) * gauss(cy - yi, σy) ]

    where σ = sigma_scale × cell_size, giving ~1-cell spread by default.
    """

    def __init__(self, benchmark: Benchmark, movable_mask: torch.Tensor, *, sigma_scale: float = 1.0):
        G = benchmark.grid_rows
        C = benchmark.grid_cols
        W = benchmark.canvas_width
        H_canvas = benchmark.canvas_height
        self.G = G
        self.C = C
        self.cell_w = W / C
        self.cell_h = H_canvas / G
        self.sigma_x = self.cell_w * sigma_scale
        self.sigma_y = self.cell_h * sigma_scale

        port_pos = benchmark.port_positions.float()
        self.has_ports = port_pos.shape[0] > 0
        self.port_positions = port_pos

        movable_idx_set = set(movable_mask.nonzero(as_tuple=True)[0].tolist())

        # Cell centers (used for Gaussian evaluation)
        self.cell_cx = (torch.arange(C, dtype=torch.float32) + 0.5) * self.cell_w  # [C]
        self.cell_cy = (torch.arange(G, dtype=torch.float32) + 0.5) * self.cell_h  # [G]

        # Split fixed/variable nets
        fixed_nets = []
        variable_nets = []
        var_pin_counts = []
        for net_nodes in benchmark.net_nodes:
            has_movable = any(n.item() in movable_idx_set for n in net_nodes)
            if has_movable:
                variable_nets.append(net_nodes)
                var_pin_counts.append(len(net_nodes))
            else:
                fixed_nets.append(net_nodes)

        # Pre-compute fixed map (one-time, no grad)
        if self.has_ports:
            fixed_all_pos = torch.cat([benchmark.macro_positions.float(), port_pos], dim=0)
        else:
            fixed_all_pos = benchmark.macro_positions.float()

        self.fixed_map = self._build_gaussian_map(fixed_all_pos, fixed_nets)  # [G, C]

        self.N_var = len(variable_nets)
        if self.N_var > 0:
            flat_indices, net_ids = [], []
            for i, nodes in enumerate(variable_nets):
                flat_indices.append(nodes)
                net_ids.append(torch.full((len(nodes),), i, dtype=torch.long))
            self.flat_node_indices = torch.cat(flat_indices)
            self.net_ids_flat = torch.cat(net_ids)
            self.pin_counts = torch.tensor(var_pin_counts, dtype=torch.float32)
        else:
            self.flat_node_indices = torch.zeros(0, dtype=torch.long)
            self.net_ids_flat = torch.zeros(0, dtype=torch.long)
            self.pin_counts = torch.zeros(0, dtype=torch.float32)

    def _build_gaussian_map(self, all_pos: torch.Tensor, nets_list: list) -> torch.Tensor:
        """Build Gaussian demand map for a list of nets (Python loop, used once for fixed nets)."""
        G, C = self.G, self.C
        demand = torch.zeros(G, C)
        for net_nodes in nets_list:
            node_pos = all_pos[net_nodes].float()  # [k, 2]
            k = len(net_nodes)
            dx = self.cell_cx[None, :] - node_pos[:, 0:1]   # [k, C]
            dy = self.cell_cy[None, :] - node_pos[:, 1:2]   # [k, G]
            bell_x = torch.exp(-dx ** 2 / (2 * self.sigma_x ** 2))  # [k, C]
            bell_y = torch.exp(-dy ** 2 / (2 * self.sigma_y ** 2))  # [k, G]
            # sum_i [bell_y_i ⊗ bell_x_i] = bell_y.T @ bell_x  [G, C]
            demand = demand + bell_y.T @ bell_x / k
        return demand

    def compute(
        self,
        positions: torch.Tensor,
        *,
        temperature: float = 20.0,
        top_frac: float = 0.05,
        blocking_offset: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Differentiable Gaussian-RUDY congestion proxy (smooth abu top-frac%).

        blocking_offset: optional [G, C] precomputed macro blocking map (no grad).
            When provided, the smooth-max is taken over (net_routing + blocking),
            matching the ground-truth congestion formula. High-blocking cells are
            pre-penalized so gradients steer macros away from those cells.
        """
        device = positions.device
        dtype = positions.dtype
        G, C = self.G, self.C

        demand = self.fixed_map.to(device=device, dtype=dtype)

        if self.N_var > 0:
            if self.has_ports:
                port_pos = self.port_positions.to(device=device, dtype=dtype)
                all_pos = torch.cat([positions, port_pos], dim=0)
            else:
                all_pos = positions

            flat_idx = self.flat_node_indices.to(device)
            net_ids  = self.net_ids_flat.to(device)
            flat_pos = all_pos[flat_idx]  # [total_pins, 2]

            cell_cx = self.cell_cx.to(device=device, dtype=dtype)
            cell_cy = self.cell_cy.to(device=device, dtype=dtype)

            dx = cell_cx[None, :] - flat_pos[:, 0:1]  # [total_pins, C]
            dy = cell_cy[None, :] - flat_pos[:, 1:2]  # [total_pins, G]
            bell_x = torch.exp(-dx ** 2 / (2 * self.sigma_x ** 2))  # [total_pins, C]
            bell_y = torch.exp(-dy ** 2 / (2 * self.sigma_y ** 2))  # [total_pins, G]

            # Per-pin footprint [total_pins, G, C] → flatten to [total_pins, G*C]
            footprint = (bell_y[:, :, None] * bell_x[:, None, :]).reshape(-1, G * C)

            # Scatter-sum per net: [N_var, G*C]
            pin_counts = self.pin_counts.to(device=device, dtype=dtype)
            per_net = torch.zeros(self.N_var, G * C, dtype=dtype, device=device)
            per_net.scatter_add_(0, net_ids[:, None].expand(-1, G * C), footprint)
            per_net = per_net / pin_counts[:, None]  # normalize by pin count per net

            demand = demand + per_net.sum(dim=0).reshape(G, C)

        # Add macro blocking offset (precomputed constant) so smooth-max selects
        # cells hot in (net_routing + blocking) — matching ground-truth formula
        if blocking_offset is not None:
            combined = demand + blocking_offset.to(device=device, dtype=dtype)
        else:
            combined = demand

        flat = combined.flatten()
        k = max(1, int(flat.numel() * top_frac))
        top_vals, _ = torch.topk(flat, k)
        return torch.logsumexp(temperature * top_vals, dim=0) / temperature


# ── Gaussian Hard Macro Blocking Loss ────────────────────────────────────────

def _gaussian_hard_blocking_loss(
    positions: torch.Tensor,
    sizes: torch.Tensor,
    num_hard: int,
    canvas_w: float,
    canvas_h: float,
    grid_rows: int,
    grid_cols: int,
    *,
    sigma_scale: float = 1.0,
    top_frac: float = 0.05,
    temperature: float = 20.0,
) -> torch.Tensor:
    """
    Gaussian-smoothed hard macro blocking pressure map.

    Each hard macro spreads its blocking contribution via a Gaussian kernel,
    giving non-zero gradient everywhere (unlike the exact geometric version
    which has zero gradient when a macro is fully inside a cell).

    blocking_map[r, c] = sum_m [ (macro_area_m / cell_area)
                                 × gauss(cx - x_m, σx)
                                 × gauss(cy - y_m, σy) ]

    Loss = smooth-max of top-frac cells (logsumexp).
    """
    G, C = grid_rows, grid_cols
    device = positions.device
    dtype  = positions.dtype

    cell_w = canvas_w / C
    cell_h = canvas_h / G
    sigma_x = cell_w * sigma_scale
    sigma_y = cell_h * sigma_scale
    cell_area = cell_w * cell_h

    cell_cx = ((torch.arange(C, device=device, dtype=dtype) + 0.5) * cell_w)  # [C]
    cell_cy = ((torch.arange(G, device=device, dtype=dtype) + 0.5) * cell_h)  # [G]

    hard_pos   = positions[:num_hard]                           # [N, 2]
    hard_sizes = sizes[:num_hard].to(device=device, dtype=dtype)
    macro_area = hard_sizes[:, 0] * hard_sizes[:, 1]           # [N]

    dx = cell_cx[None, :] - hard_pos[:, 0:1]   # [N, C]
    dy = cell_cy[None, :] - hard_pos[:, 1:2]   # [N, G]

    bell_x = torch.exp(-dx ** 2 / (2 * sigma_x ** 2))  # [N, C]
    bell_y = torch.exp(-dy ** 2 / (2 * sigma_y ** 2))  # [N, G]

    # Scale each macro's footprint by its area relative to cell area
    scaled_bell_x = bell_x * (macro_area[:, None] / cell_area)  # [N, C]

    # block_map[r, c] = bell_y[:, r] · scaled_bell_x[:, c]  →  [G, C]
    block_map = bell_y.T @ scaled_bell_x  # [G, C]

    flat = block_map.flatten()
    k = max(1, int(flat.numel() * top_frac))
    top_vals, _ = torch.topk(flat, k)
    return torch.logsumexp(temperature * top_vals, dim=0) / temperature


# ── Hard Macro Blocking Loss ─────────────────────────────────────────────────

def _hard_macro_blocking_map(
    positions: torch.Tensor,
    sizes: torch.Tensor,
    num_hard: int,
    canvas_w: float,
    canvas_h: float,
    grid_rows: int,
    grid_cols: int,
    hroutes_per_micron: float,
    vroutes_per_micron: float,
) -> torch.Tensor:
    """
    Differentiable hard-macro routing blocking map.

    Per-cell blocking = sum_macros[ ol_x * ol_y ] * (1/(cell_h * grid_v_routes)
                                                    + 1/(cell_w * grid_h_routes))

    This mirrors the evaluator's V_macro_routing_cong / H_macro_routing_cong
    computation. Using the product ol_x * ol_y makes it fully differentiable
    and ensures macros only contribute to cells they actually overlap.

    Returns:
        [grid_rows, grid_cols] blocking map
    """
    G, C = grid_rows, grid_cols
    device = positions.device
    dtype = positions.dtype

    cell_w = canvas_w / C
    cell_h = canvas_h / G
    grid_v_routes = cell_w * vroutes_per_micron
    grid_h_routes = cell_h * hroutes_per_micron

    cell_x_lo = torch.arange(C, dtype=dtype, device=device) * cell_w
    cell_x_hi = cell_x_lo + cell_w
    cell_y_lo = torch.arange(G, dtype=dtype, device=device) * cell_h
    cell_y_hi = cell_y_lo + cell_h

    hard_pos = positions[:num_hard]
    hard_sz = sizes[:num_hard]
    hw = hard_sz[:, 0] / 2.0
    hh = hard_sz[:, 1] / 2.0
    mx_lo = hard_pos[:, 0] - hw  # [N_hard]
    mx_hi = hard_pos[:, 0] + hw
    my_lo = hard_pos[:, 1] - hh
    my_hi = hard_pos[:, 1] + hh

    # ol_x: [C, N_hard]
    ol_x = torch.clamp(
        torch.minimum(cell_x_hi[:, None], mx_hi[None, :])
        - torch.maximum(cell_x_lo[:, None], mx_lo[None, :]),
        min=0.0,
    )
    # ol_y: [G, N_hard]
    ol_y = torch.clamp(
        torch.minimum(cell_y_hi[:, None], my_hi[None, :])
        - torch.maximum(cell_y_lo[:, None], my_lo[None, :]),
        min=0.0,
    )

    # overlap_area[r, c] = ol_y[r, :] @ ol_x[c, :].T  →  [G, C] via matmul
    overlap_area = ol_y @ ol_x.T  # [G, C]

    norm = 1.0 / (cell_h * grid_v_routes) + 1.0 / (cell_w * grid_h_routes)
    return overlap_area * norm


def _hard_macro_blocking_loss(
    positions: torch.Tensor,
    sizes: torch.Tensor,
    num_hard: int,
    canvas_w: float,
    canvas_h: float,
    grid_rows: int,
    grid_cols: int,
    hroutes_per_micron: float,
    vroutes_per_micron: float,
    top_frac: float = 0.05,
    temperature: float = 20.0,
    top_cells: torch.Tensor = None,
) -> torch.Tensor:
    """
    Smooth loss on top-5% hard-macro blocking cells.

    If top_cells is provided (list of (r,c) pairs), only those cells contribute.
    Otherwise all cells are used with soft-top-k via logsumexp.
    """
    bmap = _hard_macro_blocking_map(
        positions, sizes, num_hard,
        canvas_w, canvas_h, grid_rows, grid_cols,
        hroutes_per_micron, vroutes_per_micron,
    )  # [G, C]

    if top_cells is not None:
        # Only penalize the pre-identified hot cells
        rows, cols = top_cells
        cell_vals = bmap[rows, cols]
        loss = torch.logsumexp(temperature * cell_vals, dim=0) / temperature
    else:
        flat = bmap.flatten()
        k = max(1, int(flat.numel() * top_frac))
        top_vals, _ = torch.topk(flat, k)
        loss = torch.logsumexp(temperature * top_vals, dim=0) / temperature

    return loss


# ── Helpers ──────────────────────────────────────────────────────────────────

def _find_and_load_plc(benchmark: Benchmark):
    """Reconstruct plc from benchmark name by searching known directories."""
    name = benchmark.name
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [
        os.path.join(base, "external/MacroPlacement/Testcases/ICCAD04", name),
    ]
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


def _overlap_loss(
    positions: torch.Tensor,
    sizes: torch.Tensor,
    num_hard: int,
    eps: float = 1e-10,
) -> torch.Tensor:
    """
    Differentiable pairwise overlap loss over hard macros only.
    """
    pos = positions[:num_hard]
    sz = sizes[:num_hard]
    hw = sz[:, 0] / 2.0
    hh = sz[:, 1] / 2.0

    dx = pos[:, 0].unsqueeze(1) - pos[:, 0].unsqueeze(0)
    dy = pos[:, 1].unsqueeze(1) - pos[:, 1].unsqueeze(0)

    min_sep_x = hw.unsqueeze(1) + hw.unsqueeze(0) + eps
    min_sep_y = hh.unsqueeze(1) + hh.unsqueeze(0) + eps

    overlap_x = torch.clamp(min_sep_x - dx.abs(), min=0.0)
    overlap_y = torch.clamp(min_sep_y - dy.abs(), min=0.0)

    mask = torch.triu(torch.ones(num_hard, num_hard, dtype=torch.bool,
                                 device=positions.device), diagonal=1)
    return (overlap_x * overlap_y)[mask].sum()


def _get_congestion_cost_only(plc, placement, benchmark):
    """Evaluate only congestion cost (faster than full proxy)."""
    from macro_place.objective import _set_placement
    _set_placement(plc, placement, benchmark)
    return plc.get_congestion_cost()


def _update_single_macro(plc, benchmark, tensor_idx, x, y):
    """
    Update only one macro + its pins in the plc, then mark congestion dirty.
    Much faster than _set_placement which updates ALL macros.
    """
    num_hard = benchmark.num_hard_macros
    if tensor_idx < num_hard:
        macro_plc_idx = benchmark.hard_macro_indices[tensor_idx]
    else:
        macro_plc_idx = benchmark.soft_macro_indices[tensor_idx - num_hard]

    node = plc.modules_w_pins[macro_plc_idx]
    node.set_pos(x, y)

    # Build pin map cache if needed
    if not hasattr(plc, '_macro_pin_map'):
        pin_map = {}
        for idx, mod in enumerate(plc.modules_w_pins):
            if mod.get_type() == 'MACRO_PIN' and hasattr(mod, 'get_macro_name'):
                name = mod.get_macro_name()
                if name not in pin_map:
                    pin_map[name] = []
                pin_map[name].append(idx)
        plc._macro_pin_map = pin_map

    for pin_idx in plc._macro_pin_map.get(node.get_name(), []):
        pin = plc.modules_w_pins[pin_idx]
        pin.set_pos(x + pin.x_offset, y + pin.y_offset)

    plc.FLAG_UPDATE_CONGESTION = True


# ── Phase 1: Simulated Annealing on Congestion ──────────────────────────────

def _sa_congestion(
    placement: torch.Tensor,
    benchmark: Benchmark,
    plc,
    movable_indices: torch.Tensor,
    sizes: torch.Tensor,
    *,
    num_iters: int = 2000,
    T0: float = 0.0005,
    T_min: float = 1e-7,
    alpha: float = 0.995,
    perturbation_start: float = 2.0,
    perturbation_end: float = 0.1,
    log_interval: int = 50,
) -> torch.Tensor:
    """
    Simulated annealing that minimises congestion cost via small perturbations.

    Args:
        placement:       [N, 2] current positions
        benchmark:       Benchmark object
        plc:             PlacementCost object for ground-truth congestion eval
        movable_indices: [M] indices of movable hard macros
        sizes:           [N, 2] macro sizes
        num_iters:       SA iterations
        T0:              initial temperature
        T_min:           minimum temperature floor
        alpha:           geometric cooling factor
        perturbation_start: initial perturbation scale (µm)
        perturbation_end:   final perturbation scale (µm)
        log_interval:    print every N steps

    Returns:
        Optimised [N, 2] placement tensor.
    """
    best = placement.clone()
    current = placement.clone()

    # Canvas bounds per movable macro
    half_w = sizes[movable_indices, 0] / 2.0
    half_h = sizes[movable_indices, 1] / 2.0
    x_lo = half_w
    x_hi = benchmark.canvas_width - half_w
    y_lo = half_h
    y_hi = benchmark.canvas_height - half_h

    # Baseline costs
    from macro_place.objective import _set_placement
    _set_placement(plc, current, benchmark)
    base_costs = compute_proxy_cost(current, benchmark, plc)
    current_cong = base_costs["congestion_cost"]
    best_cong = current_cong

    print(f"\n  ── SA Phase: Congestion Optimisation ({num_iters} iters) ──")
    print(f"  Initial: proxy={base_costs['proxy_cost']:.6f}  "
          f"wl={base_costs['wirelength_cost']:.6f}  "
          f"den={base_costs['density_cost']:.6f}  "
          f"cong={base_costs['congestion_cost']:.6f}")
    print(f"  {'Step':>6}  {'T':>10}  {'Scale':>8}  {'Cong':>10}  "
          f"{'Best':>10}  {'Accept':>6}  {'Macro':>6}")

    T = T0
    n_accept = 0
    n_moves = len(movable_indices)

    for step in range(num_iters):
        # Linearly decay perturbation scale
        frac = step / max(num_iters - 1, 1)
        scale = perturbation_start + (perturbation_end - perturbation_start) * frac

        # Pick a random movable macro
        mi = torch.randint(n_moves, (1,)).item()
        macro_idx = movable_indices[mi].item()

        # Save old position
        old_x = current[macro_idx, 0].item()
        old_y = current[macro_idx, 1].item()

        # Perturb
        dx = torch.randn(1).item() * scale
        dy = torch.randn(1).item() * scale
        new_x = max(x_lo[mi].item(), min(x_hi[mi].item(), old_x + dx))
        new_y = max(y_lo[mi].item(), min(y_hi[mi].item(), old_y + dy))

        # Update only this macro in plc
        _update_single_macro(plc, benchmark, macro_idx, new_x, new_y)
        trial_cong = plc.get_congestion_cost()
        delta = trial_cong - current_cong

        # Accept / reject
        accepted = False
        if delta < 0:
            accepted = True
        elif T > 0:
            p = math.exp(-delta / T) if delta / T < 500 else 0.0
            if torch.rand(1).item() < p:
                accepted = True

        if accepted:
            current[macro_idx, 0] = new_x
            current[macro_idx, 1] = new_y
            current_cong = trial_cong
            n_accept += 1
            if trial_cong < best_cong:
                best = current.clone()
                best_cong = trial_cong
        else:
            # Revert the macro in plc
            _update_single_macro(plc, benchmark, macro_idx, old_x, old_y)

        # Cool
        T = max(T * alpha, T_min)

        # Log
        if step % log_interval == 0 or step == num_iters - 1:
            print(f"  {step:>6}  {T:>10.6f}  {scale:>6.3f}µm  "
                  f"{current_cong:>10.6f}  {best_cong:>10.6f}  "
                  f"{n_accept:>6}  {macro_idx:>6}")

    # Final cost summary
    final_costs = compute_proxy_cost(best, benchmark, plc)
    print(f"\n  SA result: proxy={final_costs['proxy_cost']:.6f}  "
          f"wl={final_costs['wirelength_cost']:.6f}  "
          f"den={final_costs['density_cost']:.6f}  "
          f"cong={final_costs['congestion_cost']:.6f}  "
          f"overlaps={final_costs['overlap_count']}")
    print(f"  Accepted {n_accept}/{num_iters} moves "
          f"({100*n_accept/max(num_iters,1):.1f}%)")

    return best


# ── Phase 2: Greedy Congestion Local Search ──────────────────────────────────

def _greedy_congestion_search(
    placement: torch.Tensor,
    benchmark: Benchmark,
    plc,
    movable_indices: torch.Tensor,
    sizes: torch.Tensor,
    *,
    num_rounds: int = 10,
    trials_per_macro: int = 10,
    top_k_macros: int = 10,
    perturbation: float = 0.01,
    log_interval: int = 1,
) -> torch.Tensor:
    """
    Greedy local search on congestion cost, RUDY-gradient-directed.

    Each round:
      1. Rank movable macros by their cell's congestion (hottest first).
      2. For each top-k macro, compute the RUDY gradient at its position.
      3. Propose moves along the negative gradient direction at a few step
         sizes; also try a few random perturbations as fallback.
      4. Accept any move that strictly improves ground-truth congestion.

    Using the RUDY gradient as a proposal direction is better than pure
    random because even with correlation 0.52 the gradient points away from
    high-demand regions more often than a random vector would.

    Args:
        perturbation:    base step size in microns
        num_rounds:      passes over the hot-macro list
        trials_per_macro: gradient step sizes to try + random fallbacks
        top_k_macros:    only perturb the top-k hottest macros per round
    """
    from macro_place.objective import _set_placement

    current = placement.clone()
    _set_placement(plc, current, benchmark)
    current_cong = plc.get_congestion_cost()
    best_cong = current_cong
    best = current.clone()

    # Canvas bounds per movable macro
    half_w = sizes[movable_indices, 0] / 2.0
    half_h = sizes[movable_indices, 1] / 2.0
    x_lo = half_w
    x_hi = benchmark.canvas_width - half_w
    y_lo = half_h
    y_hi = benchmark.canvas_height - half_h

    grid_w = benchmark.canvas_width / benchmark.grid_cols
    grid_h = benchmark.canvas_height / benchmark.grid_rows
    n_moves = len(movable_indices)
    num_hard = benchmark.num_hard_macros

    # Pre-build RUDY helper (fixed nets pre-computed once)
    movable_mask = benchmark.get_movable_mask() & benchmark.get_hard_macro_mask()
    rudy = RudyPrecompute(benchmark, movable_mask)

    # Step sizes to try along gradient direction (multipliers of perturbation)
    grad_scales = [1.0, 2.0, 0.5]
    n_random = max(1, trials_per_macro - len(grad_scales))

    print(f"\n  ── Greedy Congestion Search ({num_rounds} rounds, "
          f"top_k={top_k_macros}, {len(grad_scales)} grad + {n_random} rand trials, "
          f"σ={perturbation}µm) ──")
    print(f"  Initial congestion: {current_cong:.6f}")

    total_improvements = 0

    for rnd in range(num_rounds):
        # Compute per-cell congestion map and score each macro
        plc.get_routing()
        H_cong = list(plc.H_routing_cong)
        V_cong = list(plc.V_routing_cong)

        macro_scores = []
        for mi in range(n_moves):
            macro_idx = movable_indices[mi].item()
            x = current[macro_idx, 0].item()
            y = current[macro_idx, 1].item()
            col = min(int(x / grid_w), benchmark.grid_cols - 1)
            row = min(int(y / grid_h), benchmark.grid_rows - 1)
            cell = row * benchmark.grid_cols + col
            macro_scores.append((H_cong[cell] + V_cong[cell], mi))

        macro_scores.sort(reverse=True)
        top_macros = macro_scores[:top_k_macros]

        n_improved = 0
        for score, mi in top_macros:
            macro_idx = movable_indices[mi].item()
            old_x = current[macro_idx, 0].item()
            old_y = current[macro_idx, 1].item()

            # ── Compute RUDY gradient for this macro ──
            pos_req = current.clone().requires_grad_(True)
            soft_pos = current[num_hard:].detach()
            full_pos = torch.cat([pos_req[:num_hard], soft_pos], dim=0)
            rudy_loss = rudy.compute(full_pos, temperature=20.0, top_frac=0.05)
            rudy_loss.backward()
            grad = pos_req.grad[macro_idx]  # [2]: (dx, dy)
            grad_norm = grad.norm().item()

            # Normalised negative gradient direction (move away from high RUDY)
            if grad_norm > 1e-8:
                gx = -grad[0].item() / grad_norm
                gy = -grad[1].item() / grad_norm
            else:
                gx, gy = 0.0, 0.0

            # Build candidate moves: gradient-directed first, then random
            candidates = []
            for scale in grad_scales:
                step = perturbation * scale
                candidates.append((old_x + gx * step, old_y + gy * step))
            for _ in range(n_random):
                dx = torch.randn(1).item() * perturbation
                dy = torch.randn(1).item() * perturbation
                candidates.append((old_x + dx, old_y + dy))

            for cx, cy in candidates:
                new_x = max(x_lo[mi].item(), min(x_hi[mi].item(), cx))
                new_y = max(y_lo[mi].item(), min(y_hi[mi].item(), cy))

                _update_single_macro(plc, benchmark, macro_idx, new_x, new_y)
                trial_cong = plc.get_congestion_cost()

                if trial_cong < current_cong:
                    current[macro_idx, 0] = new_x
                    current[macro_idx, 1] = new_y
                    current_cong = trial_cong
                    old_x, old_y = new_x, new_y
                    n_improved += 1
                    if trial_cong < best_cong:
                        best = current.clone()
                        best_cong = trial_cong
                else:
                    _update_single_macro(plc, benchmark, macro_idx, old_x, old_y)

        total_improvements += n_improved
        if rnd % log_interval == 0 or rnd == num_rounds - 1:
            print(f"  round {rnd+1:3d}/{num_rounds}  cong={current_cong:.6f}  "
                  f"best={best_cong:.6f}  improved={n_improved} macros")

    print(f"  Total improvements: {total_improvements} across {num_rounds} rounds")
    return best


# ── Phase 3: Gradient Overlap Fine-tuning ────────────────────────────────────

def _gradient_overlap_finetune(
    placement: torch.Tensor,
    benchmark: Benchmark,
    sizes: torch.Tensor,
    movable: torch.Tensor,
    *,
    lr: float = 0.01,
    num_steps: int = 9000,
    cong_weight: float = 0.0,
    cong_temperature: float = 20.0,
    blocking_weight: float = 0.0,
    blocking_temperature: float = 20.0,
    blocking_top_frac: float = 0.05,
    gauss_blocking_weight: float = 0.0,
    gauss_blocking_sigma: float = 1.0,
    optimize_soft: bool = False,
    soft_only_rudy: bool = False,
    log_interval: int = 500,
) -> torch.Tensor:
    """
    Gradient descent on overlap loss (+ optional Gaussian RUDY congestion).

    Args:
        cong_weight:           Weight for Gaussian RUDY congestion term (0 = disabled).
        optimize_soft:         If True, co-optimize soft macro positions via RUDY gradient.
        blocking_weight:       Weight for exact geometric hard-macro blocking loss.
        gauss_blocking_weight: Weight for Gaussian-smoothed hard-macro blocking loss.
        gauss_blocking_sigma:  σ multiplier (in cell units) for Gaussian blocking spread.
    """
    num_hard = benchmark.num_hard_macros

    # ── Hard macro setup ──────────────────────────────────────────────────
    hard_half_w = sizes[:num_hard, 0] / 2.0
    hard_half_h = sizes[:num_hard, 1] / 2.0
    x_lo = hard_half_w
    x_hi = torch.full_like(hard_half_w, benchmark.canvas_width) - hard_half_w
    y_lo = hard_half_h
    y_hi = torch.full_like(hard_half_h, benchmark.canvas_height) - hard_half_h

    hard_movable = movable[:num_hard]
    movable_idx  = hard_movable.nonzero(as_tuple=True)[0]
    hard_base    = placement[:num_hard].detach()
    hard_sizes   = sizes[:num_hard]
    free_pos     = hard_base[hard_movable].clone().requires_grad_(True)

    # ── Soft macro setup (optional) ───────────────────────────────────────
    soft_base = placement[num_hard:].detach()
    free_pos_soft = None
    soft_movable_idx = None
    soft_x_lo = soft_x_hi = soft_y_lo = soft_y_hi = None

    if optimize_soft and benchmark.num_soft_macros > 0:
        soft_movable_mask = ~benchmark.macro_fixed[num_hard:]
        soft_movable_idx  = soft_movable_mask.nonzero(as_tuple=True)[0]
        if len(soft_movable_idx) > 0:
            soft_half_w = sizes[num_hard:][soft_movable_mask, 0] / 2.0
            soft_half_h = sizes[num_hard:][soft_movable_mask, 1] / 2.0
            soft_x_lo = soft_half_w
            soft_x_hi = torch.full_like(soft_half_w, benchmark.canvas_width) - soft_half_w
            soft_y_lo = soft_half_h
            soft_y_hi = torch.full_like(soft_half_h, benchmark.canvas_height) - soft_half_h
            free_pos_soft = soft_base[soft_movable_mask].clone().requires_grad_(True)

    opt_params = [free_pos] + ([free_pos_soft] if free_pos_soft is not None else [])
    optimizer  = torch.optim.Adam(opt_params, lr=lr)

    # ── RUDY pre-compute ──────────────────────────────────────────────────
    rudy = None
    if cong_weight > 0.0:
        # Expand movable mask to include soft macros so their nets become variable
        broad_movable = benchmark.get_movable_mask() if optimize_soft else movable
        print(f"  Pre-computing Gaussian RUDY (optimize_soft={optimize_soft})...")
        rudy = GaussianRudyPrecompute(benchmark, broad_movable, sigma_scale=1.0)
        print(f"  Variable nets: {rudy.N_var} / {benchmark.num_nets}")

    hot_blocking_cells = None
    blocking_refresh_interval = max(1, num_steps // 5)

    print(f"\n  ── Overlap Fine-tuning Phase ({num_steps} steps, "
          f"cong_weight={cong_weight}, optimize_soft={optimize_soft}) ──")

    for step in range(num_steps):
        optimizer.zero_grad()

        # Reconstruct full hard position tensor
        full_hard_pos = torch.index_put(hard_base, (movable_idx,), free_pos)

        # Reconstruct full soft position tensor
        if free_pos_soft is not None:
            full_soft_pos = torch.index_put(soft_base, (soft_movable_idx,), free_pos_soft)
        else:
            full_soft_pos = soft_base

        full_pos = torch.cat([full_hard_pos, full_soft_pos], dim=0)

        overlap_loss = _overlap_loss(full_hard_pos, hard_sizes, num_hard)
        cong_loss        = torch.tensor(0.0)
        block_loss       = torch.tensor(0.0)
        gauss_block_loss = torch.tensor(0.0)

        if rudy is not None:
            # soft_only_rudy: detach hard positions so RUDY gradient flows only to soft macros
            rudy_pos = torch.cat([full_hard_pos.detach(), full_soft_pos], dim=0) \
                       if soft_only_rudy else full_pos
            cong_loss = rudy.compute(rudy_pos, temperature=cong_temperature, top_frac=0.05)

        if blocking_weight > 0.0:
            if step % blocking_refresh_interval == 0:
                with torch.no_grad():
                    bmap = _hard_macro_blocking_map(
                        full_hard_pos.detach(), hard_sizes, num_hard,
                        benchmark.canvas_width, benchmark.canvas_height,
                        benchmark.grid_rows, benchmark.grid_cols,
                        benchmark.hroutes_per_micron, benchmark.vroutes_per_micron,
                    )
                    flat = bmap.flatten()
                    k = max(1, int(flat.numel() * blocking_top_frac))
                    _, top_idx = torch.topk(flat, k)
                    rows = top_idx // benchmark.grid_cols
                    cols = top_idx % benchmark.grid_cols
                    hot_blocking_cells = (rows, cols)

            block_loss = _hard_macro_blocking_loss(
                full_hard_pos, hard_sizes, num_hard,
                benchmark.canvas_width, benchmark.canvas_height,
                benchmark.grid_rows, benchmark.grid_cols,
                benchmark.hroutes_per_micron, benchmark.vroutes_per_micron,
                temperature=blocking_temperature,
                top_cells=hot_blocking_cells,
            )

        if gauss_blocking_weight > 0.0:
            gauss_block_loss = _gaussian_hard_blocking_loss(
                full_hard_pos, hard_sizes, num_hard,
                benchmark.canvas_width, benchmark.canvas_height,
                benchmark.grid_rows, benchmark.grid_cols,
                sigma_scale=gauss_blocking_sigma,
                top_frac=0.05,
                temperature=20.0,
            )

        loss = (overlap_loss
                + cong_weight * cong_loss
                + blocking_weight * block_loss
                + gauss_blocking_weight * gauss_block_loss)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            free_pos[:, 0].clamp_(x_lo[hard_movable], x_hi[hard_movable])
            free_pos[:, 1].clamp_(y_lo[hard_movable], y_hi[hard_movable])
            if free_pos_soft is not None:
                free_pos_soft[:, 0].clamp_(soft_x_lo, soft_x_hi)
                free_pos_soft[:, 1].clamp_(soft_y_lo, soft_y_hi)

        if step % log_interval == 0 or step == num_steps - 1:
            print(f"  step {step:5d}  overlap={overlap_loss.item():.4e}  "
                  f"rudy={cong_loss.item():.4f}  "
                  f"gblk={gauss_block_loss.item():.4f}")

    with torch.no_grad():
        placement[:num_hard][hard_movable] = free_pos
        if free_pos_soft is not None:
            placement[num_hard:][soft_movable_idx] = free_pos_soft

    fixed_mask = benchmark.macro_fixed
    placement[fixed_mask] = benchmark.macro_positions[fixed_mask]

    return placement


# ── Main Placer ──────────────────────────────────────────────────────────────

class GradientPlacer:
    """
    Three-phase placer:
      1. Gradient descent on overlap loss
      2. Greedy local search on congestion (tiny perturbations, ground-truth evaluator)
      3. SA on congestion (optional, disabled by default)
    """

    def __init__(
        self,
        sa_iters: int = 0,
        overlap_steps: int = 9000,
        lr: float = 0.01,
        cong_weight: float = 0.0,
        cong_temperature: float = 20.0,
        blocking_weight: float = 0.0,
        blocking_temperature: float = 20.0,
        blocking_top_frac: float = 0.05,
        optimize_soft: bool = False,
        greedy_rounds: int = 20,
        greedy_trials: int = 10,
        greedy_top_k_macros: int = 10,
        greedy_perturbation: float = 0.01,
        verbose: bool = False,
    ):
        self.sa_iters = sa_iters
        self.overlap_steps = overlap_steps
        self.lr = lr
        self.cong_weight = cong_weight
        self.cong_temperature = cong_temperature
        self.blocking_weight = blocking_weight
        self.blocking_temperature = blocking_temperature
        self.blocking_top_frac = blocking_top_frac
        self.optimize_soft = optimize_soft
        self.greedy_rounds = greedy_rounds
        self.greedy_trials = greedy_trials
        self.greedy_top_k_macros = greedy_top_k_macros
        self.greedy_perturbation = greedy_perturbation
        self.verbose = verbose

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        placement = benchmark.macro_positions.clone().float()
        sizes = benchmark.macro_sizes.float()
        movable = benchmark.get_movable_mask() & benchmark.get_hard_macro_mask()
        movable_indices = movable.nonzero(as_tuple=True)[0]

        # ── Phase 1: Gradient overlap resolution ──
        placement = _gradient_overlap_finetune(
            placement, benchmark, sizes, movable,
            lr=self.lr, num_steps=self.overlap_steps,
            cong_weight=self.cong_weight,
            cong_temperature=self.cong_temperature,
            blocking_weight=self.blocking_weight,
            blocking_temperature=self.blocking_temperature,
            blocking_top_frac=self.blocking_top_frac,
            optimize_soft=self.optimize_soft,
        )

        plc = _find_and_load_plc(benchmark)
        if plc is None:
            print(f"  [WARN] Could not load plc for {benchmark.name}, skipping congestion phases")
            return placement

        costs = compute_proxy_cost(placement, benchmark, plc)
        print(f"\n  Post-overlap: proxy={costs['proxy_cost']:.6f}  "
              f"wl={costs['wirelength_cost']:.6f}  "
              f"den={costs['density_cost']:.6f}  "
              f"cong={costs['congestion_cost']:.6f}  "
              f"overlaps={costs['overlap_count']}")

        # ── Phase 2: Greedy congestion local search ──
        if self.greedy_rounds > 0:
            placement = _greedy_congestion_search(
                placement, benchmark, plc, movable_indices, sizes,
                num_rounds=self.greedy_rounds,
                trials_per_macro=self.greedy_trials,
                top_k_macros=self.greedy_top_k_macros,
                perturbation=self.greedy_perturbation,
            )

        # ── Phase 3: SA on congestion (optional) ──
        if self.sa_iters > 0:
            placement = _sa_congestion(
                placement, benchmark, plc, movable_indices, sizes,
                num_iters=self.sa_iters,
            )

        costs = compute_proxy_cost(placement, benchmark, plc)
        print(f"\n  Final: proxy={costs['proxy_cost']:.6f}  "
              f"wl={costs['wirelength_cost']:.6f}  "
              f"den={costs['density_cost']:.6f}  "
              f"cong={costs['congestion_cost']:.6f}  "
              f"overlaps={costs['overlap_count']}")

        return placement
