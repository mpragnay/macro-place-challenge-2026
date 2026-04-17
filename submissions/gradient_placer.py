"""
Gradient Placer - Differentiable Density Optimization

Optimizes macro positions by backpropagating through a differentiable
density cost. Uses the same grid-based density formula as the ground
truth evaluator:

    density_cost = 0.5 * mean(top 10% of per-cell density values)

where per-cell density = sum of macro overlap areas with that cell / cell area.

All operations use torch.max / torch.min so gradients flow cleanly
through autograd.

Usage:
    uv run evaluate submissions/gradient_placer.py -b ibm01
    uv run evaluate submissions/gradient_placer.py --all
"""

import math

import torch
from macro_place.benchmark import Benchmark


def _overlap_loss(
    positions: torch.Tensor,
    sizes: torch.Tensor,
    num_hard: int,
    eps: float = 1e-10,
) -> torch.Tensor:
    """
    Differentiable pairwise overlap loss over hard macros only.

    For each pair (i, j) of hard macros:
        overlap_x = clamp(half_w_i + half_w_j + eps - |x_i - x_j|, min=0)
        overlap_y = clamp(half_h_i + half_h_j + eps - |y_i - y_j|, min=0)
        loss += overlap_x * overlap_y

    The eps margin forces macros to stay at least eps microns apart, which
    keeps the loss (and its gradient) above float32 underflow even when
    macros are nearly touching. Without eps, near-touching pairs produce
    overlap areas ~1e-13 um², which underflow to zero when squared.

    Args:
        positions:  [N, 2] macro centers (requires_grad on movable subset)
        sizes:      [N, 2] macro (width, height), fixed
        num_hard:   number of hard macros (first num_hard rows used)
        eps:        minimum required separation in microns (default 0.1)

    Returns:
        Scalar total overlap loss (sum over all pairs within eps of each other).
    """
    pos  = positions[:num_hard]          # [H, 2]
    sz   = sizes[:num_hard]              # [H, 2]
    hw   = sz[:, 0] / 2.0               # half-widths  [H]
    hh   = sz[:, 1] / 2.0               # half-heights [H]

    # Pairwise distance in x and y: [H, H]
    dx = pos[:, 0].unsqueeze(1) - pos[:, 0].unsqueeze(0)   # x_i - x_j
    dy = pos[:, 1].unsqueeze(1) - pos[:, 1].unsqueeze(0)   # y_i - y_j

    # Minimum separation required: sum of half-sizes + margin
    min_sep_x = hw.unsqueeze(1) + hw.unsqueeze(0) + eps    # [H, H]
    min_sep_y = hh.unsqueeze(1) + hh.unsqueeze(0) + eps    # [H, H]

    # Overlap amount: positive when boxes are within eps of each other
    overlap_x = torch.clamp(min_sep_x - dx.abs(), min=0.0)  # [H, H]
    overlap_y = torch.clamp(min_sep_y - dy.abs(), min=0.0)  # [H, H]

    # Sum upper triangle only (each pair counted once).
    mask = torch.triu(torch.ones(num_hard, num_hard, dtype=torch.bool, device=positions.device), diagonal=1)
    return (overlap_x * overlap_y)[mask].sum()


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


class GradientPlacer:
    """
    Gradient-based macro placer using differentiable density cost.

    Initialises from the benchmark's own initial positions (which are
    near-optimal for wirelength but contain hard-macro overlaps) and
    descends the density gradient to spread macros apart.

    Only movable hard macros are optimised; fixed macros and soft macros
    keep their initial positions throughout.
    """

    def __init__(
        self,
        lr: float = 0.01,
        num_steps: int = 1000,
        verbose: bool = False,
    ):
        self.lr = lr
        self.num_steps = num_steps
        self.verbose = verbose

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        """
        Run gradient descent and return optimised placement.

        Args:
            benchmark: Benchmark with circuit data.

        Returns:
            [num_macros, 2] tensor of (x, y) center positions.
        """
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

        # Canvas bounds for clamping — computed only for hard macros, then filtered to movable.
        hard_half_w = sizes[:benchmark.num_hard_macros, 0] / 2.0
        hard_half_h = sizes[:benchmark.num_hard_macros, 1] / 2.0
        x_lo = hard_half_w
        x_hi = torch.full_like(hard_half_w, benchmark.canvas_width) - hard_half_w
        y_lo = hard_half_h
        y_hi = torch.full_like(hard_half_h, benchmark.canvas_height) - hard_half_h

        num_hard = benchmark.num_hard_macros

        # Work only with hard macro rows — soft macros never enter the overlap loss.
        hard_movable = movable[:num_hard]                          # [H] bool
        movable_idx = hard_movable.nonzero(as_tuple=True)[0]      # [M] int indices
        hard_base = placement[:num_hard].detach()                  # [H, 2] fixed reference
        hard_sizes = sizes[:num_hard]                              # [H, 2]

        # Optimise only the movable positions as a free parameter.
        free_pos = hard_base[hard_movable].clone().requires_grad_(True)  # [M, 2]
        optimizer = torch.optim.Adam([free_pos], lr=self.lr)

        for step in range(self.num_steps):
            optimizer.zero_grad()

            # index_put (non-inplace) builds a new [H, 2] tensor in the autograd graph
            # without cloning the full [N, 2] placement each step.
            full_hard_pos = torch.index_put(hard_base, (movable_idx,), free_pos)

            loss = _overlap_loss(full_hard_pos, hard_sizes, num_hard)

            loss.backward()
            optimizer.step()

            # Per-macro canvas clamp — use individual bounds, not global min/max,
            # so macros near the boundary don't drift into each other.
            with torch.no_grad():
                free_pos[:, 0] = torch.max(
                    torch.min(free_pos[:, 0], x_hi[hard_movable]),
                    x_lo[hard_movable],
                )
                free_pos[:, 1] = torch.max(
                    torch.min(free_pos[:, 1], y_hi[hard_movable]),
                    y_lo[hard_movable],
                )

            if step % 1000 == 0 or step == self.num_steps - 1:
                print(f"  step {step:5d}  overlap_loss={loss.item():.6e}")

        # Write optimised positions back (no extra clamp — already enforced above).
        with torch.no_grad():
            placement[:num_hard][hard_movable] = free_pos

        # Restore fixed macros to their original positions.
        fixed_mask = benchmark.macro_fixed
        placement[fixed_mask] = benchmark.macro_positions[fixed_mask]

        # Print density cost at end on the final placement (all macros: hard + soft).
        with torch.no_grad():
            d1 = _density_cost(
                placement,
                sizes,
                benchmark.canvas_width, benchmark.canvas_height,
                benchmark.grid_rows, benchmark.grid_cols,
            )
        print(f"  density_cost  end  ={d1.item():.6f}  delta={d1.item()-d0.item():+.6f}")

        return placement
