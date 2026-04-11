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
    Differentiable density cost.

    Uses log-sum-exp over all grid cells as a smooth surrogate for the top-k
    density cost. LSE has nonzero gradient everywhere (unlike mean, which is
    constant w.r.t. position for fully in-canvas macros) and naturally
    emphasises the hottest cells (like top-k) without hard selection.

    Per-cell density = sum of clamped overlap areas between each macro
    and the cell, divided by cell area.

    All intermediate ops use torch.min / torch.max so autograd can
    differentiate through them w.r.t. positions.

    Args:
        positions:      [N, 2] macro center (x, y) — must have requires_grad
        sizes:          [N, 2] macro (width, height) — fixed, no grad needed
        canvas_width:   canvas width in microns
        canvas_height:  canvas height in microns
        grid_rows:      number of grid rows
        grid_cols:      number of grid columns
        top_k_fraction: fraction of cells used for averaging (default 0.1 = top 10%)

    Returns:
        Scalar density cost tensor (differentiable w.r.t. positions).
    """
    N = positions.shape[0]
    cell_w = canvas_width / grid_cols   # width of one grid cell
    cell_h = canvas_height / grid_rows  # height of one grid cell
    cell_area = cell_w * cell_h

    # --- macro bounding boxes ---
    # [N, 1]: left / right / bottom / top edges of each macro
    half_w = sizes[:, 0:1] / 2.0   # [N, 1]
    half_h = sizes[:, 1:2] / 2.0   # [N, 1]
    macro_x0 = positions[:, 0:1] - half_w  # [N, 1]
    macro_x1 = positions[:, 0:1] + half_w  # [N, 1]
    macro_y0 = positions[:, 1:2] - half_h  # [N, 1]
    macro_y1 = positions[:, 1:2] + half_h  # [N, 1]

    # --- grid cell bounding boxes ---
    # col_idx: [1, grid_cols], row_idx: [grid_rows, 1]
    col_idx = torch.arange(grid_cols, dtype=positions.dtype, device=positions.device).unsqueeze(0)  # [1, C]
    row_idx = torch.arange(grid_rows, dtype=positions.dtype, device=positions.device).unsqueeze(1)  # [R, 1]

    cell_x0 = col_idx * cell_w                # [1, C]
    cell_x1 = cell_x0 + cell_w                # [1, C]
    cell_y0 = row_idx * cell_h                # [R, 1]
    cell_y1 = cell_y0 + cell_h                # [R, 1]

    # --- overlap area: macro i with each grid cell (r, c) ---
    # Broadcast to [N, R, C] by inserting dims appropriately.
    # macro tensors: [N, 1, 1], cell tensors: [1, R, C]
    macro_x0_3d = macro_x0.unsqueeze(2)       # [N, 1, 1]
    macro_x1_3d = macro_x1.unsqueeze(2)       # [N, 1, 1]
    macro_y0_3d = macro_y0.unsqueeze(2)       # [N, 1, 1]
    macro_y1_3d = macro_y1.unsqueeze(2)       # [N, 1, 1]

    cell_x0_3d = cell_x0.unsqueeze(0)         # [1, 1, C]
    cell_x1_3d = cell_x1.unsqueeze(0)         # [1, 1, C]
    cell_y0_3d = cell_y0.unsqueeze(0)         # [1, R, 1]
    cell_y1_3d = cell_y1.unsqueeze(0)         # [1, R, 1]

    # Overlap in x: max(0, min(macro_x1, cell_x1) - max(macro_x0, cell_x0))
    overlap_x = torch.clamp(
        torch.min(macro_x1_3d, cell_x1_3d) - torch.max(macro_x0_3d, cell_x0_3d),
        min=0.0,
    )  # [N, R, C]  — clamp keeps the zero floor differentiable via relu gradient

    # Overlap in y: max(0, min(macro_y1, cell_y1) - max(macro_y0, cell_y0))
    overlap_y = torch.clamp(
        torch.min(macro_y1_3d, cell_y1_3d) - torch.max(macro_y0_3d, cell_y0_3d),
        min=0.0,
    )  # [N, R, C]

    overlap_area = overlap_x * overlap_y      # [N, R, C]

    # --- per-cell density ---
    # Sum contributions from all macros, normalise by cell area → [R, C]
    per_cell_density = overlap_area.sum(dim=0) / cell_area  # [R, C]

    # --- top-k LSE matching ground truth formula ---
    flat = per_cell_density.reshape(-1)       # [R*C]

    # Ground truth: sort non-zero cells descending, take top floor(total*0.1).
    # k is computed from total cells (including zeros), matching plc_client_os.
    k = max(1, int(flat.numel() * 0.1))

    # Select top-k values — differentiable via torch.topk.
    top_vals, _ = torch.topk(flat, k)         # [k], descending

    # Log-sum-exp over top-k: smooth approximation to mean(top_k).
    # Avoids the sparsity/whack-a-mole problem of a hard mean while
    # staying close to the ground truth formula.
    # γ=10: tight enough to closely track the top cells.
    gamma = 10.0
    density_cost = 0.5 * torch.logsumexp(gamma * top_vals, dim=0) / gamma

    return density_cost


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
        num_steps: int = 9000,
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

        # Canvas bounds for clamping after each step.
        half_w = sizes[:, 0] / 2.0
        half_h = sizes[:, 1] / 2.0
        x_lo = half_w
        x_hi = torch.full_like(half_w, benchmark.canvas_width) - half_w
        y_lo = half_h
        y_hi = torch.full_like(half_h, benchmark.canvas_height) - half_h

        # Optimise only the movable positions as a free parameter.
        free_pos = placement[movable].clone().requires_grad_(True)
        optimizer = torch.optim.Adam([free_pos], lr=self.lr)

        for step in range(self.num_steps):
            optimizer.zero_grad()

            # Assemble full position tensor (detached fixed rows + grad-tracked free rows).
            full_pos = placement.clone()
            full_pos[movable] = free_pos

            loss = _overlap_loss(full_pos, sizes, benchmark.num_hard_macros)

            loss.backward()
            optimizer.step()

            # Per-macro canvas clamp — use individual bounds, not global min/max,
            # so macros near the boundary don't drift into each other.
            with torch.no_grad():
                free_pos[:, 0] = torch.max(
                    torch.min(free_pos[:, 0], x_hi[movable]),
                    x_lo[movable],
                )
                free_pos[:, 1] = torch.max(
                    torch.min(free_pos[:, 1], y_hi[movable]),
                    y_lo[movable],
                )

            if step % 1000 == 0 or step == self.num_steps - 1:
                print(f"  step {step:5d}  overlap_loss={loss.item():.6e}")

        # Write optimised positions back (no extra clamp — already enforced above).
        with torch.no_grad():
            placement[movable] = free_pos

        # Restore fixed macros to their original positions.
        fixed_mask = benchmark.macro_fixed
        placement[fixed_mask] = benchmark.macro_positions[fixed_mask]

        return placement
