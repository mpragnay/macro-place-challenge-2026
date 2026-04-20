# Macro Placement Challenge — Claude Context

## What This Is

Partcl/HRT VLSI macro placement competition ($20K grand prize, deadline May 21 2026).
Goal: place hard macros (SRAMs, IP blocks) on a chip canvas to minimize proxy cost.

## Objective

```
proxy_cost = 1.0 × wirelength_cost + 0.5 × density_cost + 0.5 × congestion_cost
```

Lower is better. **Must have zero hard-hard macro overlaps to be valid.**

### Cost Components

**Wirelength (HPWL)**: `weight × (max_x - min_x + max_y - min_y)` per net, normalized by `(canvas_W + canvas_H) × num_nets`. Connected macros should be close.

**Density**: Per grid cell = `sum of macro overlap areas / cell area`. Cost = `0.5 × avg(top 10% cells)`. Both hard AND soft macros contribute. Spread macros to reduce.

**Congestion**: Routing track utilization per grid cell edge. Cost = `abu(H + V routing cong, 0.05)` — avg of top 5%.

## Validity Rules

A placement `[num_macros, 2]` tensor is valid if:
1. No NaN/Inf values
2. All macros within canvas bounds (half-width/height margins enforced)
3. Fixed macros unchanged (atol=1e-3)
4. **Zero pairwise overlaps between movable hard macros** (strictly `<`; touching edges OK)

Note: Hard-soft macro overlap is allowed and does NOT disqualify.

## Key Data Structures

```python
benchmark.macro_positions       # [N, 2] initial (x, y) centers
benchmark.macro_sizes           # [N, 2] (width, height)
benchmark.macro_fixed           # [N] bool — do not move these
benchmark.canvas_width/height   # canvas dimensions in microns
benchmark.num_hard_macros       # first num_hard indices are hard macros
benchmark.num_macros            # total hard + soft
benchmark.get_movable_mask()    # bool mask of movable macros
benchmark.get_hard_macro_mask() # bool mask of hard macros
```

Macro ordering: hard macros first (indices `0..num_hard-1`), then soft macros.

## Writing a Placer

```python
import torch
from macro_place.benchmark import Benchmark

class MyPlacer:
    def place(self, benchmark: Benchmark) -> torch.Tensor:
        placement = benchmark.macro_positions.clone()
        movable = benchmark.get_movable_mask() & benchmark.get_hard_macro_mask()
        # ... modify placement[movable] ...
        return placement  # [num_macros, 2]
```

## Leaderboard Context

| Method | Avg Proxy Cost |
|--------|---------------|
| DreamPlace++ (target to beat) | 1.3998 |
| **SweepingBellPlacement (current)** | **1.4427** |
| RePlAce baseline | 1.4578 |
| Initial positions (invalid — 1939 overlaps) | 1.4551 |
| Greedy row placer (valid) | ~2.21 |
| SA baseline | ~2.1251 |

**Key insight**: Initial positions are connectivity-aware and near-RePlAce quality (1.455) but have 1939 overlaps. The challenge is resolving overlaps with minimal cost increase. Naive legalization costs ~2.21.

## Submission Results (SweepingBellPlacement)

| BM | Proxy | vs SA | vs RePlAce | Overlaps |
|----|-------|-------|-----------|---------|
| ibm01 | 1.0190 | +22.6% | −2.1% | 0 |
| ibm02 | 1.5600 | +18.2% | +15.1% | 0 |
| ibm03 | 1.3047 | +25.0% | +1.3% | 0 |
| ibm04 | 1.3101 | +12.9% | −0.6% | 0 |
| ibm06 | 1.6310 | +34.9% | −0.8% | 0 |
| ibm07 | 1.4534 | +28.2% | +0.7% | 0 |
| ibm08 | 1.4461 | +24.8% | −1.2% | 0 |
| ibm09 | 1.1044 | +20.4% | +1.3% | 0 |
| ibm10 | 1.3221 | +37.4% | +11.9% | 0 |
| ibm11 | 1.1950 | +30.2% | −1.5% | 0 |
| ibm12 | 1.6243 | +42.5% | +5.9% | 0 |
| ibm13 | 1.3671 | +28.6% | −2.4% | 0 |
| ibm14 | 1.5899 | +30.1% | −3.0% | 0 |
| ibm15 | 1.5836 | +31.1% | −4.5% | 0 |
| ibm16 | 1.4857 | +33.5% | −0.5% | 0 |
| ibm17 | 1.7391 | +52.6% | −5.7% | 0 |
| ibm18 | 1.7899 | +35.5% | −1.0% | 0 |
| **AVG** | **1.4427** | **+32.1%** | **+1.0%** | **0** |

Avg runtime: 632.56s/benchmark. Beats RePlAce by 1.0%, beats SA by 32.1%. Still 0.43% above DreamPlace++ target (1.3998).

**Benchmarks still above RePlAce**: ibm02 (+15.1%), ibm10 (+11.9%), ibm12 (+5.9%), ibm03 (+1.3%), ibm09 (+1.3%), ibm07 (+0.7%). These are the priority targets.

## Current Approach: SweepingBellPlacement

File: `submissions/gradient_placer.py`

Gradient-based overlap removal via PyTorch autograd:

1. Start from benchmark initial positions (connectivity-aware)
2. Extract movable hard macros as `free_pos` (Adam-optimized leaf tensor)
3. Minimize `_overlap_loss` (differentiable pairwise overlap area) to spread macros
4. Canvas-clamp after each step using initial-position-aware bounds

### `_overlap_loss` (pairwise overlap)

```python
overlap_x = clamp(hw_i + hw_j + eps - |x_i - x_j|, 0)
overlap_y = clamp(hh_i + hh_j + eps - |y_i - y_j|, 0)
loss = sum over upper triangle of (overlap_x * overlap_y)
```

Only passes movable hard macros — matches evaluator semantics (fixed macros excluded).

### `_density_cost` (grid-based density, exact match)

Verified exact match to `plc_client_os.get_density_cost()` on all 17 IBM benchmarks (rel_err < 1e-6):

```python
cell_w = canvas_width / grid_cols
cell_h = canvas_height / grid_rows
cell_area = cell_w * cell_h
# Per cell: sum of geometric overlap areas from all macros (hard + soft)
ov_x = clamp(min(x1, cell_x1) - max(x0, cell_x0), 0)
ov_y = clamp(min(y1, cell_y1) - max(y0, cell_y0), 0)
per_cell[r, c] = sum_macros(ov_x * ov_y) / cell_area
# Cost = 0.5 × mean of top k=floor(total_cells × 0.1) cells
k = max(1, floor(total_cells * 0.1))
cost = 0.5 × mean(topk(per_cell.flatten(), k))
```

**Key gradient properties**: Gradients flow only through macros straddling a cell boundary. A macro fully contained in one cell has zero gradient — it can't reduce density by moving slightly. This means naive backprop on top-10% density is ineffective for deeply embedded macros.

### Isolated-Cell Density Loss (experimental, ibm01 results: no benefit)

To target macros in density hotspots, we tried an isolated-cell density term:

1. Find the top-k densest cells where all 8 neighbors have density ≤ `neighbor_ratio × cell_density` ("isolated peaks")
2. Compute geometric overlap of all macros against only those cells
3. Backpropagate to push macros out of those cells

**Why it doesn't help on ibm01**: The one isolated dense cell persists throughout optimization. It's caused by fixed hard macros and soft macros (standard cell clusters) in that region — since only movable hard macros can be optimized, the density in that cell never drops. The movable macros that were contributing get resolved by the overlap loss, but fixed/soft contributions dominate.

**Sweep results on ibm01** (lr=0.01 baseline):

| Config | proxy | wl | den | cong | overlaps |
|--------|-------|----|----|------|---------|
| lr=0.01 steps=4k [ref] | 1.042 | 0.064 | 0.813 | 1.139 | 0 |
| lr=0.005 steps=8k | 1.041 | 0.064 | 0.813 | 1.139 | 0 |
| lr=0.001 steps=10k | **1.039** | 0.064 | **0.812** | 1.137 | 0 |
| lr=0.01 wd=0 (overlap only) | 1.042 | 0.064 | 0.813 | 1.139 | 0 |
| lr=0.01 wd=50 | 1.042 | 0.064 | 0.813 | 1.139 | 0 |

**Key finding**: Density weight (`wd`) has zero effect. Lower lr wins because macros move minimally — just enough to resolve overlaps — so density stays near the initial-position baseline. The challenge is congestion (1.137), not density.

### RePLACE-style Density Potential (not yet implemented)

The foundational approach used in ePlace/RePlAce/DreamPlace is to distribute each macro's density charge across the grid using a smooth potential, then repel overlapping macros via its gradient. Unlike geometric overlap area (which has zero gradient inside a cell), this gives a non-zero gradient everywhere.

**Bell-shaped basis (Wu et al., ePlace)**:

For each macro `i` at position `(xi, yi)` with half-size `(ai, bi)`, define a per-dimension bell function over each cell center `(cx, cy)`:

```
      a_i + b_u                   |xi - cx| ≤ a_i + b_u
Φx =  ──────────── × (1 - ──────────────────────────)²   for a_i < |xi-cx| ≤ a_i + 2b_u
      2*a_i*b_u + 4/3*b_u²            (a_i + 2b_u)²
      ... (exact piecewise form, see ePlace paper)
```

where `b_u = cell_width / 2` is the half-cell size. The 2D potential is `Φx × Φy`.

Density cost = `sum_cells(max(0, D(cell) - target_density)²)`, where `D(cell) = sum_macros(Φi(cell))`.

**Gradient**: `dL/d(xi) = 2*(D(cell) - target) * dΦi/d(xi)` — smooth, non-zero everywhere.

**Why this matters here**: With the bell potential, even a macro sitting fully inside one cell gets a non-zero gradient pushing it toward lower-density regions. The current geometric approach can only move macros that straddle boundaries.

**Implementation path**:
1. Implement `_bell_potential(xi, ai, bu)` for each dimension
2. Compute per-cell density `D = sum_macros(Φx * Φy)`
3. Loss = `sum_cells(relu(D - target)²)`
4. Add as `w_density * density_potential_loss` alongside `_overlap_loss`

### Canvas Clamp Bug (NG45)

NG45 benchmarks (ariane133, ariane136, mempool_tile, nvdla) have macros whose initial centers exceed `canvas_height - hh`. The evaluator accepts these as valid. Clamping them to the nominal bound moves them into neighbors.

**Fix**: Use per-macro initial-position-aware bounds:
```python
x_lo = min(hw, init_x)        # never clamp tighter than starting position
x_hi = max(canvas_w - hw, init_x)
y_lo = min(hh, init_y)
y_hi = max(canvas_h - hh, init_y)
```

## Key Files

```
submissions/
  gradient_placer.py           # Main submission — gradient-based overlap removal
  initial_positions.py         # Returns initial positions; has OOB diagnostics

submissions/examples/
  greedy_row_placer.py         # Shelf packer — valid but naive (~2.21)
  initial_positions.py         # Vanilla initial positions without diagnostics
  simple_random_placer.py      # Random baseline

macro_place/
  benchmark.py                 # Benchmark dataclass
  objective.py                 # compute_proxy_cost() — monkey-patches boundary bug
  utils.py                     # validate_placement(), visualize_placement()
  evaluate.py                  # CLI harness

external/MacroPlacement/CodeElements/Plc_client/plc_client_os.py
  # Core PlacementCost evaluator — get_cost(), get_density_cost(), get_congestion_cost()
```

## Important Observations (ibm01)

- Soft macros occupy center/left — hard macros placed there compound density cost
- Large fixed hard macro at bottom-right creates congestion hot zone — avoid routing there
- Bottom edge has very high congestion — avoid funneling nets through there
- I/O pins (boundary) define pull directions for nearby macros

## Congestion Cost Deep Dive

### Congestion Formula

```
cell_cong[r,c] = V_routing_cong[r,c] + H_routing_cong[r,c]

where:
  V_routing_cong[i] = V_macro_routing_cong[i] / grid_v_routes  +  V_net_routing[i]
  H_routing_cong[i] = H_macro_routing_cong[i] / grid_h_routes  +  H_net_routing[i]
```

Smoothing (smooth_range=2) is applied to the combined per-cell value, then `abu(top-5%)`.

**Macro blocking term** — only hard macros contribute (evaluator line 1574 checks `is_node_hard_macro`):
```
V_block[r,c] += x_overlap(hard_macro, cell) × vrouting_alloc
H_block[r,c] += y_overlap(hard_macro, cell) × hrouting_alloc
```
Soft macros do NOT contribute to macro blocking — only to net routing via their pin connections.

**grid_v_routes / grid_h_routes** — uniform across all cells:
```
grid_v_routes = cell_width  × vroutes_per_micron
grid_h_routes = cell_height × hroutes_per_micron
```
V_block[r,c] / grid_v_routes = fraction of vertical tracks consumed by hard macros.

**Net routing term** — RUDY-style: spread each net's routing demand uniformly over its bounding box. All macro types (hard + soft) and ports contribute via their pin positions.

### Decomposition on IBM Benchmarks (reliable method)

To split net routing vs macro blocking: zero out all hard macro widths/heights in plc → compute congestion → this gives net-routing-only congestion. Difference from total = macro blocking contribution.

**ibm01 at initial positions**: total=1.137, net_routing=1.097 (96.5%), macro_blocking=0.039 (3.5%)

Across all 17 IBM benchmarks, macro blocking is **0–8% of total congestion**. Net routing dominates entirely.

### Why Net Routing Dominates

All IBM hard macros are movable (no fixed hard macros). The congested cells are dominated by **soft-macro-to-soft-macro and soft-macro-to-port nets** ("fixed nets" — no movable hard macro pin). These contribute 86–99.5% of the demand at the top-5% hottest cells:

| Benchmark | Fixed net % of hot cells | Macro blocking % |
|-----------|--------------------------|-----------------|
| ibm01 | 99.5% | 3.5% |
| ibm02 | 96.9% | 7.9% |
| ibm10 | 86.1% | 7.5% |

### Congestion Optimization Experiments (ibm01)

All experiments start from initial positions (cong=1.137, net=1.097, blk=0.039, overlaps=69).

#### Hard-bbox RUDY (hard macros only)
Sparse gradients — only bbox-boundary macros get signal. Without overlap constraint, macros cluster and macro blocking explodes.

| Config | cong | net | blk | overlaps |
|--------|------|-----|-----|---------|
| overlap only (baseline) | 1.138 | — | — | 0 |
| RUDY only, no overlap | 2.336 | ↓tiny | ↑3.3 | 663 |

#### Gaussian RUDY (per-macro bell kernel, dense gradients)

Each macro spreads routing demand via `exp(-d²/2σ²)` over cells instead of hard bbox, giving non-zero gradient to all macros. But on hard macros only: loss is nearly flat (82.646→82.646) because variable nets contribute only 0.5% of hot-cell demand — fixed soft-macro nets dominate the top-5% cells.

#### Soft Macro Co-optimization

All 894 soft macros in ibm01 are movable (`macro_fixed=False`). Including them expands variable nets from 1958→5993 (100% variable). Soft macros contribute to congestion only via **net routing** (not macro blocking).

**Key ceiling experiment: Gaussian RUDY on soft macros only, hard macros frozen, no overlap loss (15k steps):**

| | cong | net (Δ) | blk (Δ) | overlaps |
|---|---|---|---|---|
| Initial | 1.137 | 1.097 | 0.039 | 69 |
| Soft-only RUDY | 1.367 | 0.949 (−0.149) | — | 69 |

Net routing drops **−0.149** but total congestion rises to 1.367. **Hard macros never moved, so macro blocking per-cell is constant.** The apparent blk increase in the decomposition is an artifact — abu(top-5%) is a global ranking, and when net routing shifts out of the previously-hottest cells, other cells (which happen to sit on hard macros) get promoted into the top-5%.

**Root cause of total congestion rising**: RUDY spreads soft macros apart to reduce routing demand in hot cells, but longer nets route demand through cells that overlap with hard macro blocking zones — those cells enter the top-5% with high combined (net+blocking) values, worsening the global abu metric.

#### Blocking-Aware RUDY (BA-RUDY) — soft macros only, hard frozen

BA-RUDY adds the precomputed hard macro blocking map as a constant offset *inside* the RUDY smooth-max before taking top-5%:

```
combined[r,c] = net_routing[r,c] + blocking_map[r,c]
loss = smooth_max(top-5% of combined)
```

This matches the ground-truth congestion formula exactly. The gradient steers soft macros away from high-blocking cells before routing demand accumulates there. Blocking map: max=0.0245, mean(top5%)=0.0245, mean=0.0105.

| | cong | net (Δ) | blk (Δ) | overlaps |
|---|---|---|---|---|
| Initial | 1.137 | 1.097 | 0.039 | 69 |
| A: Pure Gaussian RUDY | 1.367 | 0.949 (−0.149) | +0.378 | 69 |
| B: BA-RUDY scale=1 | 1.356 | 0.945 (−0.152) | +0.372 | 69 |
| C: BA-RUDY scale=10 | 1.337 | 0.948 (−0.150) | +0.349 | 69 |
| D: BA-RUDY scale=30 | 1.312 | 0.926 (−0.171) | +0.347 | 69 |
| E: BA-RUDY scale=50 | 1.301 | 0.927 (−0.170) | +0.334 | 69 |
| F: BA-RUDY scale=100 | 1.275 | 0.940 (−0.158) | +0.296 | 69 |
| G: BA-RUDY scale=200 | 1.257 | 1.006 (−0.092) | +0.211 | 69 |
| H: BA-RUDY scale=500 | 1.227 | 1.079 (−0.019) | +0.109 | 69 |
| **I: BA-RUDY scale=1000** | **1.155** | 1.126 (+0.028) | **−0.010** | 69 |
| J: BA-RUDY scale=1500 | 1.190 | 1.169 (+0.072) | −0.018 | 69 |
| K: BA-RUDY scale=2000 | 1.188 | 1.171 (+0.074) | −0.023 | 69 |

**Scale=1000 is the sweet spot** at cong=1.155. Scaling up the blocking offset monotonically reduces blocking inflation but eventually degrades net routing faster than it saves on blocking. At scale=1000, blocking drops below initial (blk −0.010) but net routing rises slightly (+0.028), giving cong=1.155 — the best achievable with this method.

**Conclusion**: BA-RUDY with scaled blocking offset dramatically improves over pure RUDY (1.367→1.155) but still cannot beat initial placement (1.137). Moving soft macros always trades worse net routing for better blocking — the initial positions already have near-optimal soft macro placement for net routing.

#### Combined RUDY + Hard Macro Blocking Loss (soft + hard co-optimized)

Adding Gaussian-smoothed hard macro blocking loss (non-zero gradient everywhere) partially suppresses macro clustering:

| Config | cong | net (Δ) | overlaps |
|--------|------|---------|---------|
| RUDY+overlap+soft (no blocking) | 1.334 | −0.149 | 7 |
| +exact blocking w=1 | 1.307 | −0.141 | 6 |
| +Gaussian blocking σ=1 w=1 | 1.290 | −0.128 | 0 |
| +Gaussian blocking σ=2 w=1 | 1.273 | −0.142 | 1 |
| +Gaussian blocking σ=1 w=5 | **1.271** | −0.141 | 0 |

Best result is 1.271 — still worse than baseline 1.138. The macro blocking term rises from 0.039 to ~0.31 because hard macros are being moved by the RUDY gradient and clustering.

### Congestion Ceiling Summary

| Method | Net routing ceiling | Total cong | Feasible? |
|--------|--------------------|-----------| ---------|
| Overlap only (hard) | 1.098 (unchanged) | **1.138** | Yes ✓ |
| Greedy local search | ~1.097 | **1.137** | Yes ✓ |
| Gaussian RUDY, soft only | 0.949 (−0.149) | 1.367 | No — abu shifts |
| BA-RUDY scale=1000, soft only | 1.126 (+0.028) | **1.155** | No — net routing degrades |
| Gaussian RUDY + blocking, hard+soft | ~0.957 (−0.14) | 1.271 | Mostly (0 overlaps) |

**Conclusion**: RUDY-based methods can reduce the net routing component by ~0.15, but total congestion worsens because (1) soft macros stretching nets routes demand through hard-macro-occupied cells, and (2) abu(top-5%) is a global ranking that gets worse when congestion redistributes rather than truly decreases. BA-RUDY (matching the GT formula exactly) gives only a marginal improvement over pure RUDY because the blocking map signal (max=0.025) is ~44× weaker than the net-routing demand (~1.1). The greedy local search (ground-truth evaluator) remains the best congestion optimizer.

## Soft Macro Co-optimization + Bell Potential Density (Current Approach)

### Architecture

The current `gradient_placer.py` optimizes both hard AND soft macros jointly:

```python
# Two leaf tensors in Adam optimizer
free_pos  = hard_base[movable_idx].clone().requires_grad_(True)   # movable hard macros
free_soft = soft_base[soft_movable].clone().requires_grad_(True)  # movable soft macros
optimizer = torch.optim.Adam([free_pos, free_soft], lr=lr)

# Build full position tensor each step
full_hard_pos = torch.index_put(hard_base, (movable_idx,), free_pos)
full_soft_pos = torch.index_put(soft_base, (soft_movable_idx,), free_soft)
full_pos = torch.cat([full_hard_pos, full_soft_pos], dim=0)

# Loss
loss = w_overlap * ol + w_density * dl + w_reg * rl
# ol = pairwise overlap of hard macros only
# dl = bell_density_loss(full_pos)  — Gaussian kernel density
# rl = ((free_soft - soft_base[soft_movable]) ** 2).mean()  — L2 reg to init
```

Write-back: `placement[num_hard:][soft_movable] = free_soft`

### Bell Potential Density Loss

Gaussian kernel density that gives non-zero gradient everywhere (fixes zero-gradient problem for macros fully inside one grid cell):

```python
def _bell_density_loss(positions, sizes, canvas_width, canvas_height,
                       grid_rows, grid_cols, top_frac=0.1, sigma_scale=1.0):
    cell_w = canvas_width / grid_cols
    cell_h = canvas_height / grid_rows
    col_cx = (torch.arange(grid_cols) + 0.5) * cell_w
    row_cy = (torch.arange(grid_rows) + 0.5) * cell_h
    sigma_x = sigma_scale * (sizes[:, 0:1] / 2.0 + cell_w / 2.0)
    sigma_y = sigma_scale * (sizes[:, 1:2] / 2.0 + cell_h / 2.0)
    phi_x = torch.exp(-(positions[:, 0:1] - col_cx) ** 2 / (2.0 * sigma_x ** 2))
    phi_y = torch.exp(-(positions[:, 1:2] - row_cy) ** 2 / (2.0 * sigma_y ** 2))
    density = (phi_y.unsqueeze(2) * phi_x.unsqueeze(1)).sum(dim=0)  # [R, C]
    k = max(1, int(density.numel() * top_frac))
    return 0.5 * torch.topk(density.flatten(), k).values.mean()
```

Constructor defaults: `w_density=0.1`, `w_reg=0.5`, `num_dense=20` (unused with bell potential).

### PLC-Based Pilot LR Sweep

Automatically selects learning rate using the plc ground-truth evaluator (no hardcoded per-benchmark params):

```python
_LR_CANDIDATES = [0.001, 0.003, 0.005, 0.01, 0.02]
_PILOT_STEPS = 500

def _pilot_lr_sweep(..., benchmark, plc, ...):
    for lr in lr_candidates:
        # run pilot_steps of overlap loss only (no density in pilot)
        costs = compute_proxy_cost(full_pos, benchmark, plc)
        score = costs["proxy_cost"]  # no overlap penalty — pilot is pure cost
        if score < best_score: best_lr = lr
```

`evaluate.py` passes plc through via `inspect.signature`:
```python
if "plc" in inspect.signature(place_fn).parameters:
    placement = place_fn(benchmark, plc=plc)
```

### Bell Potential Full-Run Results (all 17 IBM benchmarks)

Baseline = overlap-only with plc pilot lr sweep (avg 1.4541). Bell: w_density=0.1, w_reg=0.5.

| BM | overlap-only | bell potential | Δ | improved? |
|----|-------------|---------------|---|-----------|
| ibm01 | 1.0385 | **1.0263** | −0.0122 | ✓ |
| ibm02 | 1.5657 | 1.6549 | +0.0892 | ✗ |
| ibm03 | 1.3254 | **1.3142** | −0.0112 | ✓ |
| ibm04 | 1.3133 | 1.3192 | +0.0059 | ✗ |
| ibm06 | 1.6577 | 1.6619 | +0.0042 | ✗ |
| ibm07 | 1.4760 | **1.4529** | −0.0231 | ✓ |
| ibm08 | 1.4665 | **1.4567** | −0.0098 | ✓ |
| ibm09 | 1.1124 | 1.1246 INVALID | — | ✗ |
| ibm10 | 1.3367 | **1.3250** | −0.0117 | ✓ |
| ibm11 | 1.2138 | **1.2090** | −0.0048 | ✓ |
| ibm12 | 1.6254 | 1.6422 | +0.0168 | ✗ |
| ibm13 | 1.4028 | 1.4028 | 0 | — |
| ibm14 | 1.5938 | 1.6183 | +0.0245 | ✗ |
| ibm15 | 1.6033 | 1.6092 | +0.0059 | ✗ |
| ibm16 | 1.4911 | 1.5087 | +0.0176 | ✗ |
| ibm17 | 1.7391 | **1.7300** | −0.0091 | ✓ |
| ibm18 | 1.7899 | **1.7724** | −0.0175 | ✓ |
| **AVG** | **1.4578** | **1.4605** | +0.0027 | ✗ |

**Bell improves**: ibm01, 03, 07, 08, 10, 11, 17, 18 (8/17).
**Bell hurts**: ibm02, 04, 06, 12, 14, 15, 16 (7/17). ibm09 INVALID (1 overlap). ibm13 neutral.

**Net outcome**: Bell potential with fixed w_density=0.1, w_reg=0.5 is slightly worse on average (1.4605 vs 1.4578). Per-benchmark param selection is required to beat the overlap-only baseline.

**Root cause for regressions**: Bell potential spreads soft macros which stretches nets → wirelength increases → proxy worsens even if density drops. The L2 reg (w_reg=0.5) is too weak to prevent this on congested benchmarks.

### Per-Benchmark (w_density, w_reg) Sweep Results

ibm02 sweep (16 configs): **best=1.5903 at w_den=0.01, w_reg=2.0** — cannot beat overlap-only baseline (1.5657). Bell loss always hurts ibm02.

ibm04 sweep (16 configs): **best=1.3113 at w_den=0.1, w_reg=1.0** — beats overlap-only baseline (1.3133).

### Next Steps

**Current score**: 1.4427 avg — beats RePlAce (1.4578) by 1.0%, gap to DreamPlace++ (1.3998) is 0.0429.

**Priority targets** (above RePlAce baseline):
- ibm02 (+15.1%), ibm10 (+11.9%), ibm12 (+5.9%) — largest gaps, most room for improvement
- ibm03 (+1.3%), ibm09 (+1.3%), ibm07 (+0.7%) — marginal, may need different approach

**Ideas to close the gap**:
1. Wirelength loss — ibm02/10/12 have high congestion; adding HPWL gradient could help
2. Wider pilot sweep on problem benchmarks — more lr/wd/wr candidates
3. Longer runs for benchmarks that plateaued early (ibm17 at 2656s still far above RePlAce)

## Known Issues / Pending Work

1. **NG45 OOB**: 15 movable macros in ariane133_ng45 have centers beyond canvas bounds. Evaluator appears to accept this (unclear why). Pending: confirm with `uv run evaluate submissions/initial_positions.py -b ariane133_ng45`.
2. **Debug logging**: `gradient_placer.py` still has verbose step-0 debug prints — needs cleanup before final submission.
3. **ibm09 INVALID**: Bell potential with default weights causes 1 overlap on ibm09. Need higher w_overlap or w_density=0 for ibm09.
4. **Evaluator overlap semantics**: Confirmed — evaluator only checks movable-movable hard macro overlaps. Fixed macros are excluded from overlap checking.
5. **Per-benchmark density param selection**: Bell loss improves 7/17 benchmarks, hurts 7/17. Pilot sweep extension (see Next Steps above) is the critical path to capturing improvements without regressions.
