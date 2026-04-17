"""
Initial Positions Visualizer

Returns macros at their initial positions (no movement).
Use this to visualize the starting state of a benchmark.

Usage:
    uv run evaluate submissions/examples/initial_positions.py --vis
    uv run evaluate submissions/examples/initial_positions.py -b ibm03 --vis
"""

import torch
import math
from macro_place.benchmark import Benchmark


def _density_cost(
    positions: torch.Tensor,
    sizes: torch.Tensor,
    canvas_width: float,
    canvas_height: float,
    grid_rows: int,
    grid_cols: int,
) -> torch.Tensor:
    """
    Exact differentiable density cost matching plc_client_os.get_density_cost().

    Pass ALL macros (hard + soft, fixed + movable) in benchmark order.
    Fixed/soft positions should be passed as detached tensors so they
    contribute to the cost without receiving gradients.

    Formula (exact match to evaluator):
        per_cell_density[r,c] = sum_macros(overlap_area(macro, cell[r,c])) / cell_area
        k = floor(total_cells * 0.1)
        cost = 0.5 * mean(top-k per_cell_density values)
             = 0.5 * sum(top-k) / k

    Note: torch.topk over the full flat grid (including zero cells) replicates
    the evaluator's behaviour of dividing by k even when fewer than k cells
    are occupied.

    Args:
        positions:      [N, 2] macro centers — hard first, then soft
        sizes:          [N, 2] macro (width, height)
        canvas_width:   canvas width in microns
        canvas_height:  canvas height in microns
        grid_rows:      grid rows (benchmark.grid_rows)
        grid_cols:      grid columns (benchmark.grid_cols)

    Returns:
        Scalar density cost (differentiable w.r.t. positions that have requires_grad).
    """
    cell_w = canvas_width / grid_cols
    cell_h = canvas_height / grid_rows
    cell_area = cell_w * cell_h
    total_cells = grid_rows * grid_cols

    # Macro bounding boxes: [N, 1]
    half_w = sizes[:, 0:1] / 2.0
    half_h = sizes[:, 1:2] / 2.0
    x0 = positions[:, 0:1] - half_w   # [N, 1]
    x1 = positions[:, 0:1] + half_w
    y0 = positions[:, 1:2] - half_h
    y1 = positions[:, 1:2] + half_h

    # Grid cell boundaries
    col_idx = torch.arange(grid_cols, dtype=positions.dtype, device=positions.device)  # [C]
    row_idx = torch.arange(grid_rows, dtype=positions.dtype, device=positions.device)  # [R]
    cell_x0 = col_idx * cell_w          # [C]
    cell_x1 = cell_x0 + cell_w
    cell_y0 = row_idx * cell_h          # [R]
    cell_y1 = cell_y0 + cell_h

    # Overlap in x: [N, 1] vs [C] → [N, C]
    ov_x = torch.clamp(
        torch.min(x1, cell_x1.unsqueeze(0)) - torch.max(x0, cell_x0.unsqueeze(0)),
        min=0.0,
    )
    # Overlap in y: [N, 1] vs [R] → [N, R]
    ov_y = torch.clamp(
        torch.min(y1, cell_y1.unsqueeze(0)) - torch.max(y0, cell_y0.unsqueeze(0)),
        min=0.0,
    )

    # Overlap area per (macro, row, col): [N, R, C] via outer product over R and C
    overlap_area = ov_y.unsqueeze(2) * ov_x.unsqueeze(1)  # [N, R, C]

    # Per-cell density: sum over macros, normalise by cell area → [R, C]
    per_cell_density = overlap_area.sum(dim=0) / cell_area

    flat = per_cell_density.reshape(-1)   # [R*C]
    k = max(1, math.floor(total_cells * 0.1))
    top_vals, _ = torch.topk(flat, k)    # k largest (zeros included if needed)

    return 0.5 * top_vals.mean()


def _print_density_grid(
    positions: torch.Tensor,
    sizes: torch.Tensor,
    canvas_width: float,
    canvas_height: float,
    grid_rows: int,
    grid_cols: int,
    label: str = "",
) -> torch.Tensor:
    """
    Compute and print per-cell density as a 2D grid.

    Returns the [grid_rows, grid_cols] density tensor.
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
    threshold = torch.topk(flat, k).values[-1].item()   # min value in top-k

    header = f"  Density grid ({grid_rows}r x {grid_cols}c){f'  [{label}]' if label else ''}"
    print(header)
    print(f"  top-10% threshold={threshold:.4f}  scale: . <0.5  : 0.5-1  # 1-2  @ >2  * top-10%")
    # rows printed top-to-bottom = high y first (matches visual canvas orientation)
    for r in range(grid_rows - 1, -1, -1):
        row_str = "  |"
        for c in range(grid_cols):
            d = per_cell[r, c].item()
            if d >= threshold and d > 0:
                ch = "*"
            elif d > 2.0:
                ch = "@"
            elif d > 1.0:
                ch = "#"
            elif d > 0.5:
                ch = ":"
            elif d > 0.0:
                ch = "."
            else:
                ch = " "
            row_str += ch
        row_str += "|"
        print(row_str)
    print()
    return per_cell


def _per_cell_density(
    positions: torch.Tensor,
    sizes: torch.Tensor,
    canvas_width: float,
    canvas_height: float,
    grid_rows: int,
    grid_cols: int,
    detach: bool = True,
) -> torch.Tensor:
    """Return [grid_rows, grid_cols] per-cell density tensor."""
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


def _spread_isolated_dense_cells(
    placement: torch.Tensor,
    sizes: torch.Tensor,
    movable_hard_mask: torch.Tensor,
    canvas_width: float,
    canvas_height: float,
    grid_rows: int,
    grid_cols: int,
    margin: float = 0.5,
    isolation_ratio: float = 2.0,
) -> torch.Tensor:
    """
    One pass: find isolated top-10% density cells and push their movable macros
    into the lowest-density adjacent neighbor.

    A cell is "isolated" if its density divided by the mean of its non-zero
    neighbours is >= isolation_ratio.  Only 4-connected neighbours are checked.

    Macros are assigned to cells by center position.  The smallest displacement
    that moves the center into the target cell is applied (plus `margin` µm).

    Args:
        placement:         [N, 2] all macro positions (modified in-place clone)
        sizes:             [N, 2] all macro sizes
        movable_hard_mask: [N] bool — only these macros are moved
        margin:            extra µm past the cell boundary to avoid sitting exactly on edge
        isolation_ratio:   min density / mean(neighbour densities) to qualify

    Returns:
        Updated [N, 2] placement tensor.
    """
    placement = placement.clone()

    cell_w = canvas_width / grid_cols
    cell_h = canvas_height / grid_rows

    # Neighbour offsets: (dr, dc)
    neighbours = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    per_cell = _per_cell_density(
        placement, sizes, canvas_width, canvas_height, grid_rows, grid_cols
    )  # [R, C]

    total_cells = grid_rows * grid_cols
    k = max(1, math.floor(total_cells * 0.1))
    flat = per_cell.reshape(-1)
    threshold = torch.topk(flat, k).values[-1].item()

    # Find isolated top-10% cells, sorted by isolation ratio descending.
    candidates = []
    for r in range(grid_rows):
        for c in range(grid_cols):
            d = per_cell[r, c].item()
            if d < threshold:
                continue
            nbr_densities = [
                per_cell[r + dr, c + dc].item()
                for dr, dc in neighbours
                if 0 <= r + dr < grid_rows and 0 <= c + dc < grid_cols
            ]
            if not nbr_densities:
                continue
            mean_nbr = sum(nbr_densities) / len(nbr_densities)
            if mean_nbr == 0:
                ratio = float("inf")
            else:
                ratio = d / mean_nbr
            if ratio >= isolation_ratio:
                candidates.append((ratio, r, c))

    candidates.sort(reverse=True)

    moved = 0
    for _, r, c in candidates:
        # Recompute density after previous moves.
        per_cell = _per_cell_density(
            placement, sizes, canvas_width, canvas_height, grid_rows, grid_cols
        )

        # Skip if this cell is no longer in top-10% after earlier moves.
        if per_cell[r, c].item() < threshold:
            continue

        # Find best neighbour (lowest density).
        best_nbr = None
        best_d = float("inf")
        for dr, dc in neighbours:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < grid_rows and 0 <= nc < grid_cols):
                continue
            nd = per_cell[nr, nc].item()
            if nd < best_d:
                best_d = nd
                best_nbr = (nr, nc, dr, dc)

        if best_nbr is None:
            continue
        nr, nc, dr, dc = best_nbr

        # Find movable macros whose center is in cell (r, c).
        cell_x0 = c * cell_w
        cell_x1 = cell_x0 + cell_w
        cell_y0 = r * cell_h
        cell_y1 = cell_y0 + cell_h

        in_cell = (
            movable_hard_mask
            & (placement[:, 0] >= cell_x0) & (placement[:, 0] < cell_x1)
            & (placement[:, 1] >= cell_y0) & (placement[:, 1] < cell_y1)
        )
        macro_indices = in_cell.nonzero(as_tuple=True)[0]
        if macro_indices.numel() == 0:
            continue

        # Sort by area descending — move the largest macro first (most density impact).
        areas = sizes[macro_indices, 0] * sizes[macro_indices, 1]
        order = areas.argsort(descending=True)
        macro_indices = macro_indices[order]

        # Move the largest macro; minimum displacement to enter the target cell.
        idx = macro_indices[0].item()
        xi, yi = placement[idx, 0].item(), placement[idx, 1].item()

        if dr == 0:   # horizontal move
            if dc == 1:   # move right
                new_x = cell_x1 + margin
            else:         # move left
                new_x = c * cell_w - margin
            new_y = yi
        else:             # vertical move
            new_x = xi
            if dr == 1:   # move up
                new_y = cell_y1 + margin
            else:         # move down
                new_y = r * cell_h - margin

        # Clamp to canvas so we don't push outside.
        hw = sizes[idx, 0].item() / 2.0
        hh = sizes[idx, 1].item() / 2.0
        new_x = max(hw, min(canvas_width  - hw, new_x))
        new_y = max(hh, min(canvas_height - hh, new_y))

        placement[idx, 0] = new_x
        placement[idx, 1] = new_y
        moved += 1

    print(f"  [spread] moved {moved} macros from {len(candidates)} isolated dense cells")
    return placement


class InitialPositions:
    def place(self, benchmark: Benchmark) -> torch.Tensor:
        # Start from the benchmark's initial positions (connectivity-aware).
        placement = benchmark.macro_positions.clone().float()

        # Mask of positions we are allowed to move.
        movable = (
            benchmark.get_movable_mask() & benchmark.get_hard_macro_mask()
        )  # [N] bool

        sizes = benchmark.macro_sizes.float()  # fixed, no grad
        
        # Print density cost at start (initial positions).
        with torch.no_grad():
            d0 = _density_cost(
                placement,
                sizes,
                benchmark.canvas_width, benchmark.canvas_height,
                benchmark.grid_rows, benchmark.grid_cols,
            )
        print(f"  density_cost  start={d0.item():.6f}")
        _print_density_grid(
            placement, sizes,
            benchmark.canvas_width, benchmark.canvas_height,
            benchmark.grid_rows, benchmark.grid_cols,
        )
        return benchmark.macro_positions.clone()
