# Test Commands

## IBM Benchmarks (Tier 1 — Proxy Cost)

### Single benchmark
```bash
uv run evaluate submissions/gradient_placer.py -b ibm01
```

### All 17 IBM benchmarks
```bash
uv run evaluate submissions/gradient_placer.py --all
```

### Specific benchmarks
```bash
uv run evaluate submissions/gradient_placer.py -b ibm02
uv run evaluate submissions/gradient_placer.py -b ibm05
uv run evaluate submissions/gradient_placer.py -b ibm18
```

### With visualization
```bash
uv run evaluate submissions/gradient_placer.py -b ibm01 --vis
# Output saved to vis/ibm01.png
```

---

## NG45 Benchmarks (Tier 2 — Grand Prize path)

### All NG45 designs
```bash
uv run evaluate submissions/gradient_placer.py --ng45
```

### Individual NG45 designs
```bash
uv run evaluate submissions/gradient_placer.py -b ariane133_ng45
uv run evaluate submissions/gradient_placer.py -b ariane136_ng45
uv run evaluate submissions/gradient_placer.py -b mempool_tile_ng45
uv run evaluate submissions/gradient_placer.py -b nvdla_ng45
```

### With visualization
```bash
uv run evaluate submissions/gradient_placer.py -b ariane133_ng45 --vis
```

---

## Diagnostic: Initial Positions

Use `submissions/initial_positions.py` to check initial state (no movement, includes OOB diagnostics):

```bash
# Check initial positions validity and OOB status
uv run evaluate submissions/initial_positions.py -b ibm01
uv run evaluate submissions/initial_positions.py -b ariane133_ng45

# Check all IBM benchmarks
uv run evaluate submissions/initial_positions.py --all

# Check all NG45 benchmarks
uv run evaluate submissions/initial_positions.py --ng45
```

---

## Baselines / Reference

```bash
# Greedy row placer (valid, ~2.21 proxy cost)
uv run evaluate submissions/examples/greedy_row_placer.py -b ibm01
uv run evaluate submissions/examples/greedy_row_placer.py --all

# Vanilla initial positions (invalid — overlaps, but useful proxy cost reference ~1.455)
uv run evaluate submissions/examples/initial_positions.py -b ibm01
```

---

## Target Numbers

| Method | Avg Proxy Cost |
|--------|---------------|
| **DreamPlace++** (target to beat) | **1.3998** |
| RePlAce baseline | 1.4578 |
| Initial positions (invalid) | 1.4551 |
| Greedy row placer (valid) | ~2.21 |

A valid submission must be below **1.3998** to be competitive for the Grand Prize.
