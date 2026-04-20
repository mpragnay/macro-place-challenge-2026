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
from macro_place.objective import compute_proxy_cost

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


# ── Bell potential density loss ───────────────────────────────────────────────

def _bell_density_loss(
    positions: torch.Tensor,
    sizes: torch.Tensor,
    canvas_width: float,
    canvas_height: float,
    grid_rows: int,
    grid_cols: int,
    top_frac: float = 0.1,
    sigma_scale: float = 1.0,
) -> torch.Tensor:
    """
    Gaussian bell potential density loss (ePlace/RePlAce-style).

    Each macro contributes to every bin via a Gaussian kernel so the gradient
    is non-zero everywhere — including macros fully contained in one cell.

    σ_x = sigma_scale * (half_macro_w + half_cell_w)
    σ_y = sigma_scale * (half_macro_h + half_cell_h)

    density[r,c] = Σ_i exp(-dx²/2σx²) * exp(-dy²/2σy²)
    loss = 0.5 * mean(top top_frac bins)   [matches the actual density cost formula]
    """
    cell_w = canvas_width  / grid_cols
    cell_h = canvas_height / grid_rows

    col_cx = (torch.arange(grid_cols, dtype=positions.dtype, device=positions.device) + 0.5) * cell_w
    row_cy = (torch.arange(grid_rows, dtype=positions.dtype, device=positions.device) + 0.5) * cell_h

    xi = positions[:, 0:1]          # [N, 1]
    yi = positions[:, 1:2]          # [N, 1]
    sigma_x = sigma_scale * (sizes[:, 0:1] / 2.0 + cell_w / 2.0)  # [N, 1]
    sigma_y = sigma_scale * (sizes[:, 1:2] / 2.0 + cell_h / 2.0)  # [N, 1]

    phi_x = torch.exp(-(xi - col_cx.unsqueeze(0)) ** 2 / (2.0 * sigma_x ** 2))  # [N, C]
    phi_y = torch.exp(-(yi - row_cy.unsqueeze(0)) ** 2 / (2.0 * sigma_y ** 2))  # [N, R]

    density = (phi_y.unsqueeze(2) * phi_x.unsqueeze(1)).sum(dim=0)  # [R, C]

    flat = density.reshape(-1)
    k = max(1, int(flat.numel() * top_frac))
    top_vals, _ = torch.topk(flat, k)
    return 0.5 * top_vals.mean()


# ── Isolated dense cell detection ────────────────────────────────────────────

def _find_isolated_dense_cells(
    per_cell_density: torch.Tensor,
    num_dense: int,
    neighbor_ratio: float,
    min_sparse_neighbors: int = 5,
) -> list:
    """
    Find up to `num_dense` densest grid cells where at least
    `min_sparse_neighbors` of the 8-connected neighbours have density
    <= neighbor_ratio * cell_density (partial isolation criterion).

    Args:
        per_cell_density:    [R, C] density tensor (detached)
        num_dense:           max isolated cells to return
        neighbor_ratio:      neighbour is sparse if <= this fraction of cell density
        min_sparse_neighbors: minimum number of sparse neighbours required (default 5, i.e. >4/8)

    Returns:
        List of (r, c) tuples sorted by density descending.
    """
    R, C = per_cell_density.shape
    flat = per_cell_density.reshape(-1)
    _, top_idx = torch.topk(flat, R * C)

    isolated = []
    for idx in top_idx.tolist():
        r, c = divmod(idx, C)
        d = per_cell_density[r, c].item()
        if d == 0.0:
            break

        nbrs = [per_cell_density[r + dr, c + dc].item()
                for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                if (dr != 0 or dc != 0) and 0 <= r + dr < R and 0 <= c + dc < C]
        sparse_count = sum(1 for n in nbrs if n <= neighbor_ratio * d)
        if sparse_count >= min_sparse_neighbors:
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

_LR_CANDIDATES = [0.001, 0.003, 0.005, 0.01]
_WD_CANDIDATES = [0.01, 0.05, 0.1]
_WR_CANDIDATES = [0.2, 0.5, 1.0, 2.0]
_PILOT_STEPS   = 500


def _pilot_joint_sweep(
    free_pos_init: torch.Tensor,
    hard_base: torch.Tensor,
    movable_idx: torch.Tensor,
    hard_sizes: torch.Tensor,
    num_hard: int,
    soft_base: torch.Tensor,
    soft_movable: torch.Tensor,
    soft_movable_idx: torch.Tensor,
    sizes: torch.Tensor,
    x_lo: torch.Tensor,
    x_hi: torch.Tensor,
    y_lo: torch.Tensor,
    y_hi: torch.Tensor,
    sx_lo: torch.Tensor,
    sx_hi: torch.Tensor,
    sy_lo: torch.Tensor,
    sy_hi: torch.Tensor,
    benchmark,
    plc,
    lr_candidates: list,
    wd_candidates: list,
    wr_candidates: list,
    pilot_steps: int,
    w_overlap: float = 5.0,
    verbose: bool = True,
    dev: torch.device = None,
) -> tuple:
    if dev is None:
        dev = _DEVICE
    """
    Joint sweep over (lr, w_density, w_reg) using the plc ground-truth evaluator.
    Runs pilot_steps of full loss (overlap + bell density + L2 reg) for each of
    the len(lr)*len(wd)*len(wr) configs and returns (lr, w_density, w_reg) with
    lowest proxy cost.  When w_density==0, w_reg has no effect on hard macros
    (soft macros just stay near init), so those configs collapse to overlap-only —
    still swept for completeness.
    """
    best_params = (lr_candidates[0], 0.0, wr_candidates[0])
    best_score  = float("inf")
    total = len(lr_candidates) * len(wd_candidates) * len(wr_candidates)
    run   = 0

    for lr in lr_candidates:
        for wd in wd_candidates:
            for wr in wr_candidates:
                run += 1
                free_pos  = free_pos_init.clone().detach().requires_grad_(True)
                free_soft = soft_base[soft_movable].clone().detach().requires_grad_(True)
                opt = torch.optim.Adam([free_pos, free_soft], lr=lr)

                for _ in range(pilot_steps):
                    opt.zero_grad()
                    full_hard = torch.index_put(hard_base, (movable_idx,), free_pos)
                    full_soft = torch.index_put(soft_base, (soft_movable_idx,), free_soft)
                    full_pos  = torch.cat([full_hard, full_soft], dim=0)

                    ol   = _overlap_loss(full_hard, hard_sizes, num_hard)
                    dl   = _bell_density_loss(
                        full_pos, sizes,
                        benchmark.canvas_width, benchmark.canvas_height,
                        benchmark.grid_rows, benchmark.grid_cols,
                    )
                    rl   = ((free_soft - soft_base[soft_movable]) ** 2).mean()
                    loss = w_overlap * ol + wd * dl + wr * rl
                    loss.backward()
                    opt.step()

                    with torch.no_grad():
                        free_pos[:, 0].clamp_(x_lo, x_hi)
                        free_pos[:, 1].clamp_(y_lo, y_hi)
                        free_soft[:, 0].clamp_(sx_lo, sx_hi)
                        free_soft[:, 1].clamp_(sy_lo, sy_hi)

                with torch.no_grad():
                    full_hard = torch.index_put(hard_base, (movable_idx,), free_pos)
                    full_soft = torch.index_put(soft_base, (soft_movable_idx,), free_soft)
                    full_pos  = torch.cat([full_hard, full_soft], dim=0)
                    full_pos[benchmark.macro_fixed.to(dev)] = benchmark.macro_positions[benchmark.macro_fixed].float().to(dev)

                costs = compute_proxy_cost(full_pos.cpu(), benchmark, plc)
                score = costs["proxy_cost"]

                if verbose:
                    print(
                        f"  [{run:3d}/{total}] lr={lr}  wd={wd}  wr={wr}"
                        f"  proxy={score:.6f}  overlaps={costs['overlap_count']}"
                    )

                if score < best_score:
                    best_score  = score
                    best_params = (lr, wd, wr)

    if verbose:
        lr_, wd_, wr_ = best_params
        print(f"  → selected lr={lr_}  w_density={wd_}  w_reg={wr_}  (proxy={best_score:.6f})")
    return best_params


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
        w_density: float = 0.1,
        w_reg: float = 0.5,
        num_dense: int = 20,
        neighbor_ratio: float = 0.3,
        refresh_every: int = 200,
        log_every: int = 1000,
        verbose: bool = True,
    ):
        self.lr = lr
        self.num_steps = num_steps
        self.w_overlap = w_overlap
        self.w_density = w_density
        self.w_reg = w_reg
        self.num_dense = num_dense
        self.neighbor_ratio = neighbor_ratio
        self.refresh_every = refresh_every
        self.log_every = log_every
        self.verbose = verbose

    def place(self, benchmark: Benchmark, plc=None) -> torch.Tensor:
        dev = _DEVICE
        placement = benchmark.macro_positions.clone().float().to(dev)
        movable = (benchmark.get_movable_mask() & benchmark.get_hard_macro_mask()).to(dev)
        sizes = benchmark.macro_sizes.float().to(dev)

        num_hard = benchmark.num_hard_macros
        hard_movable = movable[:num_hard]
        movable_idx = hard_movable.nonzero(as_tuple=True)[0]
        hard_base = placement[:num_hard].detach()
        hard_sizes = sizes[:num_hard]
        soft_base = placement[num_hard:].detach()

        # Soft movable macros (for density loss)
        soft_movable = (~benchmark.macro_fixed[num_hard:]).to(dev)
        soft_movable_idx = soft_movable.nonzero(as_tuple=True)[0]
        soft_sizes = sizes[num_hard:]
        soft_half_w = soft_sizes[:, 0] / 2.0
        soft_half_h = soft_sizes[:, 1] / 2.0
        sx_lo = soft_half_w[soft_movable]
        sx_hi = benchmark.canvas_width  - soft_half_w[soft_movable]
        sy_lo = soft_half_h[soft_movable]
        sy_hi = benchmark.canvas_height - soft_half_h[soft_movable]

        # Canvas clamp bounds for hard macros
        hard_half_w = hard_sizes[:, 0] / 2.0
        hard_half_h = hard_sizes[:, 1] / 2.0
        x_lo = hard_half_w[hard_movable]
        x_hi = benchmark.canvas_width  - hard_half_w[hard_movable]
        y_lo = hard_half_h[hard_movable]
        y_hi = benchmark.canvas_height - hard_half_h[hard_movable]

        # Joint pilot sweep: select (lr, w_density, w_reg) via plc ground-truth evaluator
        n_configs = len(_LR_CANDIDATES) * len(_WD_CANDIDATES) * len(_WR_CANDIDATES)
        if self.verbose:
            print(f"  Running joint pilot sweep ({n_configs} configs × {_PILOT_STEPS} steps)...")
        if plc is not None:
            lr, w_density, w_reg = _pilot_joint_sweep(
                hard_base[hard_movable], hard_base, movable_idx,
                hard_sizes, num_hard,
                soft_base, soft_movable, soft_movable_idx, sizes,
                x_lo, x_hi, y_lo, y_hi,
                sx_lo, sx_hi, sy_lo, sy_hi,
                benchmark, plc,
                _LR_CANDIDATES, _WD_CANDIDATES, _WR_CANDIDATES,
                _PILOT_STEPS, self.w_overlap, self.verbose, dev,
            )
        else:
            # No plc: fall back to overlap-only with default weights
            lr       = _LR_CANDIDATES[2]  # 0.005 safe default
            w_density = self.w_density
            w_reg     = self.w_reg
            if self.verbose:
                print(f"  No plc available — using defaults lr={lr}  w_density={w_density}  w_reg={w_reg}")
        num_steps = self.num_steps

        free_pos  = hard_base[hard_movable].clone().requires_grad_(True)
        free_soft = soft_base[soft_movable].clone().requires_grad_(True)
        optimizer = torch.optim.Adam([free_pos, free_soft], lr=lr)

        # Initial density cost
        with torch.no_grad():
            init_full = torch.cat([hard_base, soft_base], dim=0)
            d0 = _density_cost(init_full, sizes,
                               benchmark.canvas_width, benchmark.canvas_height,
                               benchmark.grid_rows, benchmark.grid_cols)
        print(f"  density_cost start={d0.item():.6f}")

        _ES_CHECK = 500   # evaluate proxy every N steps
        _ES_WINDOW = 1000  # look back this many steps for proxy change
        _ES_TOL = 1e-3    # stop if proxy improved < 0.1% over window
        _proxy_history: list[float] = []
        _best_proxy = float("inf")
        _best_free_pos = free_pos.detach().clone()
        _best_free_soft = free_soft.detach().clone()

        for step in range(num_steps):
            optimizer.zero_grad()

            full_hard_pos = torch.index_put(hard_base, (movable_idx,), free_pos)
            full_soft_pos = torch.index_put(soft_base, (soft_movable_idx,), free_soft)
            full_pos = torch.cat([full_hard_pos, full_soft_pos], dim=0)

            ol = _overlap_loss(full_hard_pos, hard_sizes, num_hard)
            dl = _bell_density_loss(
                full_pos, sizes,
                benchmark.canvas_width, benchmark.canvas_height,
                benchmark.grid_rows, benchmark.grid_cols,
            )
            # L2 regularization pulls soft macros toward initial positions
            rl = ((free_soft - soft_base[soft_movable]) ** 2).mean()
            loss = self.w_overlap * ol + w_density * dl + w_reg * rl
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                free_pos[:, 0].clamp_(x_lo, x_hi)
                free_pos[:, 1].clamp_(y_lo, y_hi)
                free_soft[:, 0].clamp_(sx_lo, sx_hi)
                free_soft[:, 1].clamp_(sy_lo, sy_hi)

            if self.verbose and (step % self.log_every == 0 or step == num_steps - 1):
                with torch.no_grad():
                    fh = torch.index_put(hard_base, (movable_idx,), free_pos)
                    fs = torch.index_put(soft_base, (soft_movable_idx,), free_soft)
                    dc = _density_cost(torch.cat([fh, fs], dim=0), sizes,
                                       benchmark.canvas_width, benchmark.canvas_height,
                                       benchmark.grid_rows, benchmark.grid_cols)
                print(
                    f"  step {step:5d}"
                    f"  overlap={ol.item():.4e}"
                    f"  bell_density={dl.item():.4e}"
                    f"  density_cost={dc.item():.6f}"
                )

            # Checkpoint + early stopping every _ES_CHECK steps (only when valid)
            if step % _ES_CHECK == 0 and step > 0 and ol.item() == 0.0:
                with torch.no_grad():
                    fh = torch.index_put(hard_base, (movable_idx,), free_pos)
                    fs = torch.index_put(soft_base, (soft_movable_idx,), free_soft)
                    fp = torch.cat([fh, fs], dim=0)
                    fp[benchmark.macro_fixed.to(dev)] = benchmark.macro_positions[benchmark.macro_fixed].float().to(dev)
                costs = compute_proxy_cost(fp.cpu(), benchmark, plc)
                proxy_val = costs["proxy_cost"]
                overlap_count = costs.get("overlap_count", 0)

                if overlap_count == 0 and proxy_val < _best_proxy:
                    _best_proxy = proxy_val
                    _best_free_pos  = free_pos.detach().clone()
                    _best_free_soft = free_soft.detach().clone()
                    if self.verbose:
                        print(f"  checkpoint step {step}  proxy={proxy_val:.6f}")

                _proxy_history.append(proxy_val)
                window_steps = _ES_WINDOW // _ES_CHECK
                if len(_proxy_history) >= window_steps:
                    past = _proxy_history[-window_steps]
                    if past - proxy_val < _ES_TOL * past:
                        if self.verbose:
                            print(f"  early stop at step {step} (proxy plateau, overlap=0)")
                        break

        # Restore best checkpointed state if we found a valid one
        if _best_proxy < float("inf"):
            free_pos  = _best_free_pos
            free_soft = _best_free_soft
            if self.verbose:
                print(f"  restoring best checkpoint  proxy={_best_proxy:.6f}")

        # Write back
        with torch.no_grad():
            placement[:num_hard][hard_movable] = free_pos
            placement[num_hard:][soft_movable] = free_soft

        fixed_mask = benchmark.macro_fixed  # keep on CPU for indexing benchmark tensors
        placement[fixed_mask.to(dev)] = benchmark.macro_positions[fixed_mask].float().to(dev)

        with torch.no_grad():
            d1 = _density_cost(placement, sizes,
                               benchmark.canvas_width, benchmark.canvas_height,
                               benchmark.grid_rows, benchmark.grid_cols)
        print(f"  density_cost end  ={d1.item():.6f}  delta={d1.item()-d0.item():+.6f}")

        return placement.cpu()
