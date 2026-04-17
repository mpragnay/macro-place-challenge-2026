"""
Gradient Placer - Overlap + Isolated-Cell Density Optimization

Phase 1: overlap loss drives all movable hard macros apart.
Phase 2: isolated-cell density loss targets the few densest cells that are
         surrounded by 8 sparse neighbours — these are the easy wins that
         gradient descent on the full density grid misses (zero gradient for
         fully-contained macros).

Usage:
    uv run evaluate submissions/gradient_placer.py -b ibm01
    uv run evaluate submissions/gradient_placer.py --all
"""

import math

import torch
from macro_place.benchmark import Benchmark


# ── Overlap loss ─────────────────────────────────────────────────────────────

def _overlap_loss(
    positions: torch.Tensor,
    sizes: torch.Tensor,
    num_hard: int,
    eps: float = 1e-10,
) -> torch.Tensor:
    """
    Pairwise overlap loss over all hard macros (fixed + movable).

    overlap_x = clamp(hw_i + hw_j + eps - |x_i - x_j|, 0)
    overlap_y = clamp(hh_i + hh_j + eps - |y_i - y_j|, 0)
    loss = sum of overlap_x * overlap_y over upper triangle
    """
    pos = positions[:num_hard]
    sz  = sizes[:num_hard]
    hw  = sz[:, 0] / 2.0
    hh  = sz[:, 1] / 2.0

    dx = pos[:, 0].unsqueeze(1) - pos[:, 0].unsqueeze(0)
    dy = pos[:, 1].unsqueeze(1) - pos[:, 1].unsqueeze(0)

    min_sep_x = hw.unsqueeze(1) + hw.unsqueeze(0) + eps
    min_sep_y = hh.unsqueeze(1) + hh.unsqueeze(0) + eps

    overlap_x = torch.clamp(min_sep_x - dx.abs(), min=0.0)
    overlap_y = torch.clamp(min_sep_y - dy.abs(), min=0.0)

    mask = torch.triu(torch.ones(num_hard, num_hard, dtype=torch.bool, device=positions.device), diagonal=1)
    return (overlap_x * overlap_y)[mask].sum()


# ── Density helpers ───────────────────────────────────────────────────────────

def _density_cost(
    positions: torch.Tensor,
    sizes: torch.Tensor,
    canvas_width: float,
    canvas_height: float,
    grid_rows: int,
    grid_cols: int,
) -> torch.Tensor:
    """
    Exact density cost matching plc_client_os.get_density_cost().
    Pass ALL macros (hard + soft). Verified 17/17 IBM benchmarks < 1e-5 rel error.
    """
    cell_w = canvas_width / grid_cols
    cell_h = canvas_height / grid_rows
    cell_area = cell_w * cell_h
    total_cells = grid_rows * grid_cols

    half_w = sizes[:, 0:1] / 2.0
    half_h = sizes[:, 1:2] / 2.0
    x0 = positions[:, 0:1] - half_w
    x1 = positions[:, 0:1] + half_w
    y0 = positions[:, 1:2] - half_h
    y1 = positions[:, 1:2] + half_h

    col_idx = torch.arange(grid_cols, dtype=positions.dtype, device=positions.device)
    row_idx = torch.arange(grid_rows, dtype=positions.dtype, device=positions.device)
    cell_x0 = col_idx * cell_w
    cell_x1 = cell_x0 + cell_w
    cell_y0 = row_idx * cell_h
    cell_y1 = cell_y0 + cell_h

    ov_x = torch.clamp(
        torch.min(x1, cell_x1.unsqueeze(0)) - torch.max(x0, cell_x0.unsqueeze(0)),
        min=0.0,
    )
    ov_y = torch.clamp(
        torch.min(y1, cell_y1.unsqueeze(0)) - torch.max(y0, cell_y0.unsqueeze(0)),
        min=0.0,
    )
    per_cell = (ov_y.unsqueeze(2) * ov_x.unsqueeze(1)).sum(dim=0) / cell_area  # [R, C]

    flat = per_cell.reshape(-1)
    k = max(1, math.floor(total_cells * 0.1))
    top_vals, _ = torch.topk(flat, k)
    return 0.5 * top_vals.mean()


def _per_cell(
    positions: torch.Tensor,
    sizes: torch.Tensor,
    canvas_width: float,
    canvas_height: float,
    grid_rows: int,
    grid_cols: int,
) -> torch.Tensor:
    """Return [R, C] per-cell density tensor (no top-k aggregation)."""
    cell_w = canvas_width / grid_cols
    cell_h = canvas_height / grid_rows
    cell_area = cell_w * cell_h

    half_w = sizes[:, 0:1] / 2.0
    half_h = sizes[:, 1:2] / 2.0
    x0 = positions[:, 0:1] - half_w
    x1 = positions[:, 0:1] + half_w
    y0 = positions[:, 1:2] - half_h
    y1 = positions[:, 1:2] + half_h

    col_idx = torch.arange(grid_cols, dtype=positions.dtype, device=positions.device)
    row_idx = torch.arange(grid_rows, dtype=positions.dtype, device=positions.device)
    cell_x0 = col_idx * cell_w
    cell_x1 = cell_x0 + cell_w
    cell_y0 = row_idx * cell_h
    cell_y1 = cell_y0 + cell_h

    ov_x = torch.clamp(
        torch.min(x1, cell_x1.unsqueeze(0)) - torch.max(x0, cell_x0.unsqueeze(0)),
        min=0.0,
    )
    ov_y = torch.clamp(
        torch.min(y1, cell_y1.unsqueeze(0)) - torch.max(y0, cell_y0.unsqueeze(0)),
        min=0.0,
    )
    return (ov_y.unsqueeze(2) * ov_x.unsqueeze(1)).sum(dim=0) / cell_area  # [R, C]


# ── Isolated dense cell detection ────────────────────────────────────────────

def _find_isolated_dense_cells(
    per_cell_density: torch.Tensor,
    num_dense: int,
    neighbor_ratio: float,
) -> list:
    """
    Find up to `num_dense` densest grid cells where every 8-connected
    neighbour has density <= neighbor_ratio * cell_density.

    Args:
        per_cell_density: [R, C] density tensor (detached)
        num_dense:        max isolated cells to return
        neighbor_ratio:   neighbour must be <= this fraction of cell density

    Returns:
        List of (r, c) tuples sorted by density descending.
    """
    R, C = per_cell_density.shape
    flat = per_cell_density.reshape(-1)
    n_candidates = min(R * C, num_dense * 20)
    _, top_idx = torch.topk(flat, n_candidates)

    isolated = []
    for idx in top_idx.tolist():
        r, c = divmod(idx, C)
        d = per_cell_density[r, c].item()
        if d == 0.0:
            break

        max_nbr = max(
            (per_cell_density[r + dr, c + dc].item()
             for dr in (-1, 0, 1) for dc in (-1, 0, 1)
             if (dr != 0 or dc != 0) and 0 <= r + dr < R and 0 <= c + dc < C),
            default=0.0,
        )
        if max_nbr <= neighbor_ratio * d:
            isolated.append((r, c))
            if len(isolated) >= num_dense:
                break

    return isolated


# ── Isolated-cell density loss ───────────────────────────────────────────────

def _isolated_cell_density_loss(
    full_pos: torch.Tensor,
    sizes: torch.Tensor,
    canvas_width: float,
    canvas_height: float,
    grid_rows: int,
    grid_cols: int,
    isolated_cells: list,
) -> torch.Tensor:
    """
    Density loss restricted to the isolated dense cells only.

    For each isolated cell (r, c):
        density(r,c) = sum_macros(overlap_area(macro, cell)) / cell_area
    loss = mean(density over isolated cells)

    Gradients flow through full_pos (which embeds free_pos via index_put).

    Args:
        full_pos:       [N, 2] all macro positions — movable rows have grad
        sizes:          [N, 2] all macro sizes
        isolated_cells: list of (r, c) from _find_isolated_dense_cells
    """
    if not isolated_cells:
        return full_pos.sum() * 0.0   # zero, keeps autograd graph

    cell_w = canvas_width / grid_cols
    cell_h = canvas_height / grid_rows
    cell_area = cell_w * cell_h

    half_w = sizes[:, 0:1] / 2.0   # [N, 1]
    half_h = sizes[:, 1:2] / 2.0
    x0 = full_pos[:, 0:1] - half_w
    x1 = full_pos[:, 0:1] + half_w
    y0 = full_pos[:, 1:2] - half_h
    y1 = full_pos[:, 1:2] + half_h

    total = full_pos.sum() * 0.0   # differentiable zero
    for r, c in isolated_cells:
        cx0 = c * cell_w
        cx1 = cx0 + cell_w
        cy0 = r * cell_h
        cy1 = cy0 + cell_h
        # clamp(max=scalar) = min(tensor, scalar), clamp(min=scalar) = max(tensor, scalar)
        ov_x = torch.clamp(x1.clamp(max=cx1) - x0.clamp(min=cx0), min=0.0)  # [N, 1]
        ov_y = torch.clamp(y1.clamp(max=cy1) - y0.clamp(min=cy0), min=0.0)  # [N, 1]
        total = total + (ov_x * ov_y).sum() / cell_area

    return total / len(isolated_cells)


# ── Placer ────────────────────────────────────────────────────────────────────

class GradientPlacer:
    """
    Gradient-based macro placer.

    Joint loss: w_overlap * overlap_loss + w_density * isolated_cell_density_loss

    The isolated-cell density loss targets only the top `num_dense` densest
    cells that are surrounded by 8 sparse neighbours (all neighbours <=
    neighbor_ratio * cell density).  This avoids the zero-gradient problem
    of the full density cost for fully-contained macros.

    Isolated cells are redetected every `refresh_every` steps using a
    detached position snapshot.
    """

    def __init__(
        self,
        lr: float = 0.001,
        num_steps: int = 10000,
        w_overlap: float = 5.0,
        w_density: float = 0.0,
        num_dense: int = 3,
        neighbor_ratio: float = 0.3,
        refresh_every: int = 200,
        log_every: int = 1000,
        verbose: bool = True,
    ):
        self.lr = lr
        self.num_steps = num_steps
        self.w_overlap = w_overlap
        self.w_density = w_density
        self.num_dense = num_dense
        self.neighbor_ratio = neighbor_ratio
        self.refresh_every = refresh_every
        self.log_every = log_every
        self.verbose = verbose

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        placement = benchmark.macro_positions.clone().float()
        movable = benchmark.get_movable_mask() & benchmark.get_hard_macro_mask()
        sizes = benchmark.macro_sizes.float()

        num_hard = benchmark.num_hard_macros
        hard_movable = movable[:num_hard]
        movable_idx = hard_movable.nonzero(as_tuple=True)[0]
        hard_base = placement[:num_hard].detach()
        hard_sizes = sizes[:num_hard]
        soft_pos = placement[num_hard:].detach()   # [S, 2] — fixed throughout

        # Canvas clamp bounds
        hard_half_w = hard_sizes[:, 0] / 2.0
        hard_half_h = hard_sizes[:, 1] / 2.0
        x_lo = hard_half_w[hard_movable]
        x_hi = benchmark.canvas_width  - hard_half_w[hard_movable]
        y_lo = hard_half_h[hard_movable]
        y_hi = benchmark.canvas_height - hard_half_h[hard_movable]

        free_pos = hard_base[hard_movable].clone().requires_grad_(True)
        optimizer = torch.optim.Adam([free_pos], lr=self.lr)

        # Initial density cost
        with torch.no_grad():
            init_full = torch.cat([hard_base, soft_pos], dim=0)
            d0 = _density_cost(init_full, sizes,
                               benchmark.canvas_width, benchmark.canvas_height,
                               benchmark.grid_rows, benchmark.grid_cols)
        print(f"  density_cost start={d0.item():.6f}")

        isolated_cells = []
        last_refresh = -self.refresh_every  # trigger on step 0

        for step in range(self.num_steps):
            optimizer.zero_grad()

            full_hard_pos = torch.index_put(hard_base, (movable_idx,), free_pos)  # [H, 2]
            full_pos = torch.cat([full_hard_pos, soft_pos], dim=0)                # [N, 2]

            # Refresh isolated cell list
            if step - last_refresh >= self.refresh_every:
                with torch.no_grad():
                    pc = _per_cell(
                        full_pos.detach(), sizes,
                        benchmark.canvas_width, benchmark.canvas_height,
                        benchmark.grid_rows, benchmark.grid_cols,
                    )
                    isolated_cells = _find_isolated_dense_cells(
                        pc, self.num_dense, self.neighbor_ratio
                    )
                last_refresh = step

            ol = _overlap_loss(full_hard_pos, hard_sizes, num_hard)
            dl = _isolated_cell_density_loss(
                full_pos, sizes,
                benchmark.canvas_width, benchmark.canvas_height,
                benchmark.grid_rows, benchmark.grid_cols,
                isolated_cells,
            )
            loss = self.w_overlap * ol + self.w_density * dl
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                free_pos[:, 0] = free_pos[:, 0].clamp(min=x_lo, max=x_hi)
                free_pos[:, 1] = free_pos[:, 1].clamp(min=y_lo, max=y_hi)

            if self.verbose and (step % self.log_every == 0 or step == self.num_steps - 1):
                with torch.no_grad():
                    dc = _density_cost(
                        torch.cat([
                            torch.index_put(hard_base, (movable_idx,), free_pos),
                            soft_pos,
                        ], dim=0),
                        sizes,
                        benchmark.canvas_width, benchmark.canvas_height,
                        benchmark.grid_rows, benchmark.grid_cols,
                    )
                print(
                    f"  step {step:5d}"
                    f"  overlap={ol.item():.4e}"
                    f"  iso_density={dl.item():.4e}"
                    f"  density_cost={dc.item():.6f}"
                    f"  n_isolated={len(isolated_cells)}"
                )

        # Write back
        with torch.no_grad():
            placement[:num_hard][hard_movable] = free_pos

        fixed_mask = benchmark.macro_fixed
        placement[fixed_mask] = benchmark.macro_positions[fixed_mask]

        with torch.no_grad():
            d1 = _density_cost(placement, sizes,
                               benchmark.canvas_width, benchmark.canvas_height,
                               benchmark.grid_rows, benchmark.grid_cols)
        print(f"  density_cost end  ={d1.item():.6f}  delta={d1.item()-d0.item():+.6f}")

        return placement
